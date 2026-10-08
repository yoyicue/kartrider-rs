//! `.rho5` archive unpacking. Header/file-name tables are read through a
//! Rho5Reader (64-byte block subtractive cipher, seekable, key swappable like the
//! C# Rho5DecryptStream); file data via the two-pass GetData (decrypt first 0x400
//! then the whole buffer, both with a fresh key, then zlib). See rust/docs/rho5-format.md.

use crate::archive::Entry;
use crate::rho5_key::{get_file_key_u1, get_packed_file_key};
use crate::rho5_keyprovider::Rho5KeyProvider;
use std::path::Path;

/// Region salt mixed into every key. CN client.
pub const SALT_CN: &str = "d$Bjgfc8@dH4TQ?k";
pub const SALT_KR: &str = "y&errfV6GRS!e8JL";
pub const SALT_TW: &str = "t5rHKg-g9BA7%=qD";

fn char_sum(name: &str) -> i32 {
    name.to_lowercase().chars().map(|c| c as i32).sum()
}

fn get_header_offset(name: &str) -> i32 {
    let sum = char_sum(name);
    let mpl = ((sum as i64).wrapping_mul(0xA41A_41A5i64) >> 32) as i32;
    let mut result = sum - mpl;
    result >>= 1;
    result += mpl;
    result >>= 8;
    result *= 0x138;
    sum - result + 0x1E
}

fn get_file_names_offset(name: &str) -> i32 {
    let sum = char_sum(name) * 3;
    let mpl = ((sum as i64).wrapping_mul(0x3521_CFB3i64) >> 32) as i32;
    let mut result = sum - mpl;
    result >>= 1;
    result += mpl;
    result >>= 7;
    result *= 0xD4;
    sum - result + 0x2A
}

/// Decrypt `data` in place: 64-byte blocks, each u32 word `-= next_sub_num()`.
fn rho5_decrypt(prov: &mut Rho5KeyProvider, data: &mut [u8]) {
    let len = data.len();
    let mut off = 0;
    while off < len {
        let n = (len - off).min(64);
        let words = n.div_ceil(4);
        for w in 0..words {
            let sub = prov.next_sub_num();
            let wo = off + w * 4;
            let take = (len - wo).min(4);
            let mut word = 0u32;
            for k in 0..take {
                word |= (data[wo + k] as u32) << (k * 8);
            }
            word = word.wrapping_sub(sub);
            for k in 0..take {
                data[wo + k] = (word >> (k * 8)) as u8;
            }
        }
        off += n;
    }
}

/// Seekable streaming reader mirroring C# Rho5DecryptStream (64-byte buffer).
struct Rho5Reader<'a> {
    data: &'a [u8],
    prov: Rho5KeyProvider,
    base_pos: usize, // next byte to pull from `data`
    seek_base: usize,
    consumed: usize,
    buf: [u8; 64],
    buf_pos: usize,
    buf_count: usize,
}

impl<'a> Rho5Reader<'a> {
    fn new(data: &'a [u8], prov: Rho5KeyProvider) -> Self {
        Rho5Reader {
            data,
            prov,
            base_pos: 0,
            seek_base: 0,
            consumed: 0,
            buf: [0; 64],
            buf_pos: 64,
            buf_count: 64,
        }
    }
    fn seek(&mut self, off: usize) {
        self.base_pos = off;
        self.seek_base = off;
        self.consumed = 0;
        self.buf_pos = 64;
        self.buf_count = 64;
    }
    fn position(&self) -> usize {
        self.seek_base + self.consumed
    }
    fn refresh(&mut self) -> bool {
        if self.base_pos >= self.data.len() {
            return false;
        }
        let n = (self.data.len() - self.base_pos).min(64);
        self.buf[..n].copy_from_slice(&self.data[self.base_pos..self.base_pos + n]);
        self.base_pos += n;
        let words = n.div_ceil(4);
        for w in 0..words {
            let sub = self.prov.next_sub_num();
            let off = w * 4;
            let take = (n - off).min(4); // real bytes; low bytes correct regardless of padding
            let mut word = 0u32;
            for k in 0..4 {
                if off + k < 64 {
                    word |= (self.buf[off + k] as u32) << (k * 8);
                }
            }
            word = word.wrapping_sub(sub);
            for k in 0..take {
                self.buf[off + k] = (word >> (k * 8)) as u8;
            }
        }
        self.buf_pos = 0;
        self.buf_count = n;
        true
    }
    fn read(&mut self, out: &mut [u8]) {
        let mut w = 0;
        while w < out.len() {
            if self.buf_pos >= self.buf_count && !self.refresh() {
                return;
            }
            let copy = (self.buf_count - self.buf_pos).min(out.len() - w);
            out[w..w + copy].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + copy]);
            self.buf_pos += copy;
            self.consumed += copy;
            w += copy;
        }
    }
    fn read_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        self.read(&mut v);
        v
    }
    fn read_u8(&mut self) -> u8 {
        let mut b = [0u8; 1];
        self.read(&mut b);
        b[0]
    }
    fn read_i32(&mut self) -> i32 {
        let mut b = [0u8; 4];
        self.read(&mut b);
        i32::from_le_bytes(b)
    }
    fn read_text_utf16(&mut self) -> String {
        let len = self.read_i32() as usize; // char count
        let bytes = self.read_bytes(len * 2);
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    }
}

/// Decrypt + unpack a `.rho5` file into its entries.
pub fn unpack_rho5(path: &Path, salt: &str) -> Result<Vec<Entry>, String> {
    let buf = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("bad file name")?;

    let header_off = get_header_offset(name);
    let fn_off = header_off + get_file_names_offset(name);
    if header_off < 0 || fn_off < 0 {
        return Err("negative offset".into());
    }

    // Header (header key).
    let mut hp = Rho5KeyProvider::new();
    hp.init_header_key(name, salt);
    let mut hr = Rho5Reader::new(&buf, hp);
    hr.seek(header_off as usize);
    let crc = hr.read_i32();
    let version = hr.read_u8();
    let file_count = hr.read_i32();
    if crc != version as i32 + file_count {
        return Err(format!(
            "rho5 header crc mismatch (crc={crc} ver={version} count={file_count})"
        ));
    }
    if !(0..=10_000_000).contains(&file_count) {
        return Err("absurd file count".into());
    }

    // File-name table (files-info key).
    let mut fp = Rho5KeyProvider::new();
    fp.init_files_info_key(name, salt);
    let mut fr = Rho5Reader::new(&buf, fp);
    fr.seek(fn_off as usize);

    struct F {
        path: String,
        offset: i32,
        decomp: usize,
        comp: usize,
        key: Vec<u8>,
    }
    let mut files = Vec::with_capacity(file_count as usize);
    for _ in 0..file_count {
        let path = fr.read_text_utf16();
        let _chksum = fr.read_i32();
        let _unknown = fr.read_i32();
        let offset = fr.read_i32();
        let decomp = fr.read_i32() as usize;
        let comp = fr.read_i32() as usize;
        let key = fr.read_bytes(0x10);
        files.push(F {
            path,
            offset,
            decomp,
            comp,
            key,
        });
    }
    let data_base = ((fr.position() + 0x3FF) >> 10) << 10;

    // Extract each file.
    let u1 = get_file_key_u1(salt);
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let dec_key = get_packed_file_key(&f.key, u1, &f.path);
        let start = f.offset as usize * 0x400 + data_base;
        if start + f.comp > buf.len() {
            return Err(format!("data past EOF for {}", f.path));
        }
        let mut data = buf[start..start + f.comp].to_vec();
        // pass 1: first up-to-0x400 bytes with a fresh key
        let mut p1 = Rho5KeyProvider::new();
        p1.init_from_key(&dec_key);
        let first = f.comp.min(0x400);
        rho5_decrypt(&mut p1, &mut data[0..first]);
        // pass 2: the whole buffer with a fresh key
        let mut p2 = Rho5KeyProvider::new();
        p2.init_from_key(&dec_key);
        rho5_decrypt(&mut p2, &mut data);
        // zlib inflate
        let mut outdata = miniz_oxide::inflate::decompress_to_vec_zlib(&data)
            .map_err(|e| format!("zlib {}: {e:?}", f.path))?;
        outdata.truncate(f.decomp);
        out.push(Entry {
            path: f.path,
            data: outdata,
        });
    }
    Ok(out)
}
