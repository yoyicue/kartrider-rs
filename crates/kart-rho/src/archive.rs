//! `.rho` archive unpacking (layer "Rh layer spec 1.1").
//!
//! Layout: 0x40-byte identifier · 0x80-byte archive-info @0x80 · block-info table
//! @0x100 (0x20 bytes each) · data blocks. Folders/files form a tree reached by
//! BFS from data index 0xFFFFFFFF. See KartLibrary RhoArchive for the reference.

use crate::crypto::*;
use std::collections::HashMap;
use std::path::Path;

const ARCHIVE_END_MAGIC: u32 = 0xfc1f_9778;

// RhoBlockProperty bit flags.
const COMPRESSED: u32 = 2;
const PARTIAL_ENCRYPTED: u32 = 4; // bit also set by FullEncrypted(5) / CompressedEncrypted(7)

/// Launcher `RhoFileProperty`, used when writing `.rho` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RhoFileProperty {
    None = 0,
    Compressed = 1,
    Encrypted = 4,
    PartialEncrypted = 5,
    CompressedEncrypted = 6,
}

/// Match Launcher's `RhoPacker.GetFileTypeByExtension`.
pub fn file_property_for_extension(ext: &str, file_size: usize) -> RhoFileProperty {
    let ext = ext.to_ascii_lowercase();
    match ext.as_str() {
        ".1s" | ".dds" | ".tga" | ".f30" | ".fft" | ".hdr" | ".wav" | ".bmh" | ".bmx" => {
            RhoFileProperty::Compressed
        }
        ".xml" | ".uset" => RhoFileProperty::Encrypted,
        ".bml" => RhoFileProperty::CompressedEncrypted,
        ".flac" | ".ogg" | ".jpg" | ".kap" | ".ksv" => RhoFileProperty::PartialEncrypted,
        ".png" if file_size > 256 => RhoFileProperty::PartialEncrypted,
        ".png" => RhoFileProperty::Encrypted,
        _ => RhoFileProperty::None,
    }
}

#[derive(Clone, Copy)]
struct DataInfo {
    offset: u64,
    data_size: usize,
    uncompressed_size: usize,
    block_property: u32,
}

/// One extracted file: archive-relative path (`/`-separated) and its bytes.
pub struct Entry {
    pub path: String,
    pub data: Vec<u8>,
}

/// Decrypt + unpack a `.rho` file into its contained entries.
pub fn unpack(rho_path: &Path) -> Result<Vec<Entry>, String> {
    let buf = std::fs::read(rho_path).map_err(|e| format!("read: {e}"))?;
    if buf.len() < 0x100 {
        return Err("file too small".into());
    }
    let stem = rho_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("bad file name")?;
    let rho_key = get_rho_key(stem);

    // Identifier — we only support layer 1.1 (every shipped archive).
    let ident = utf16le_string(&buf[0..0x22]); // "Rh layer spec 1.1" = 17 chars
    if !ident.starts_with("Rh layer spec 1.1") {
        return Err(format!("unsupported layer identifier: {ident:?}"));
    }

    // Archive info @0x80 (0x80 bytes, header cipher).
    let mut info = buf[0x80..0x100].to_vec();
    decrypt_header_info(&mut info, rho_key);
    let mut r = Cursor::new(&info);
    let info_chksum = r.u32();
    let verify = adler32(&info[4..0x80]);
    if info_chksum != verify {
        return Err("archive info checksum mismatch (modified or wrong key)".into());
    }
    let _version = r.i32();
    let data_info_count = r.i32();
    if !(0..=10_000_000).contains(&data_info_count) {
        return Err("absurd data-info count".into());
    }
    let whitening_key = r.u32();
    let mut block_key = whitening_key ^ rho_key;
    // layer 1.1: i32 u1, i32 u2, u32 dataChecksum, then end magic
    let _u1 = r.i32();
    let _u2 = r.i32();
    let _data_checksum = r.u32();
    let end_magic = r.u32();
    if end_magic != ARCHIVE_END_MAGIC {
        return Err("invalid archive-info end magic".into());
    }

    // Block-info table @0x100.
    let mut data_info: HashMap<u32, DataInfo> = HashMap::with_capacity(data_info_count as usize);
    let mut pos = 0x100usize;
    for _ in 0..data_info_count {
        if pos + 0x20 > buf.len() {
            return Err("block-info table runs past EOF".into());
        }
        let mut block = buf[pos..pos + 0x20].to_vec();
        decrypt_header_info(&mut block, block_key);
        block_key = block_key.wrapping_add(1);
        let mut br = Cursor::new(&block);
        let index = br.u32();
        let offset = (br.u32() as u64) << 8;
        let data_size = br.i32() as usize;
        let uncompressed_size = br.i32() as usize;
        let block_property = br.u32();
        let _checksum = br.u32();
        data_info.insert(
            index,
            DataInfo {
                offset,
                data_size,
                uncompressed_size,
                block_property,
            },
        );
        pos += 0x20;
    }

    let get_data = |index: u32, key: u32| -> Result<Vec<u8>, String> {
        let di = data_info.get(&index).ok_or("data index not found")?;
        if di.offset as usize + di.data_size > buf.len() {
            return Err("block data past EOF".into());
        }
        let mut out = buf[di.offset as usize..di.offset as usize + di.data_size].to_vec();
        if di.block_property & COMPRESSED != 0 {
            out = miniz_oxide::inflate::decompress_to_vec_zlib(&out)
                .map_err(|e| format!("zlib: {e:?}"))?;
            if out.len() != di.uncompressed_size {
                // not fatal, but a strong signal something's off
                return Err(format!(
                    "decompressed {} != expected {}",
                    out.len(),
                    di.uncompressed_size
                ));
            }
        }
        if di.block_property & PARTIAL_ENCRYPTED != 0 {
            decrypt_data(key, &mut out);
        }
        // Exactly PartialEncrypted(4): first block holds the encrypted head, the
        // unencrypted tail is the next block, stored contiguously after it.
        if di.block_property == PARTIAL_ENCRYPTED {
            if let Some(sec) = data_info.get(&(index + 1)) {
                let tail_start = di.offset as usize + di.data_size;
                let tail_end = tail_start + sec.data_size;
                if tail_end <= buf.len() {
                    out.extend_from_slice(&buf[tail_start..tail_end]);
                }
            }
        }
        Ok(out)
    };

    // BFS the folder tree from the root data index.
    let folder_key = get_directory_data_key(rho_key);
    let mut out_entries = Vec::new();
    let mut queue: Vec<(u32, String)> = vec![(0xFFFF_FFFF, String::new())];
    while let Some((folder_index, prefix)) = queue.pop() {
        let folder_data = get_data(folder_index, folder_key)?;
        let mut fr = Cursor::new(&folder_data);
        let folder_count = fr.i32();
        for _ in 0..folder_count {
            let name = fr.utf16_nullterm();
            let sub_index = fr.u32();
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            queue.push((sub_index, path));
        }
        let file_count = fr.i32();
        for _ in 0..file_count {
            let file_name = fr.utf16_nullterm();
            let ext_int = fr.u32();
            let _file_property = fr.i32();
            let data_index = fr.u32();
            let _file_size = fr.i32();
            let ext = ext_string(ext_int);
            let file_key = get_file_key(rho_key, &file_name, ext_int);
            let full = if ext.is_empty() {
                file_name.clone()
            } else {
                format!("{file_name}.{ext}")
            };
            let rel = if prefix.is_empty() {
                full
            } else {
                format!("{prefix}/{full}")
            };
            let data = get_data(data_index, file_key)?;
            out_entries.push(Entry { path: rel, data });
        }
    }
    Ok(out_entries)
}

fn ext_string(ext_int: u32) -> String {
    let bytes = ext_int.to_le_bytes();
    let s: String = bytes
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as char)
        .collect();
    s
}

fn utf16le_string(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .take_while(|&u| u != 0)
        .collect();
    String::from_utf16_lossy(&units)
}

/// Minimal little-endian cursor over a byte slice.
struct Cursor<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn new(d: &'a [u8]) -> Self {
        Cursor { d, p: 0 }
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes([
            self.d[self.p],
            self.d[self.p + 1],
            self.d[self.p + 2],
            self.d[self.p + 3],
        ]);
        self.p += 4;
        v
    }
    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
    fn utf16_nullterm(&mut self) -> String {
        let mut units = Vec::new();
        loop {
            let u = u16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
            self.p += 2;
            if u == 0 {
                break;
            }
            units.push(u);
        }
        String::from_utf16_lossy(&units)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_file_property_mapping() {
        assert_eq!(
            file_property_for_extension(".1s", 10),
            RhoFileProperty::Compressed
        );
        assert_eq!(
            file_property_for_extension(".BML", 10),
            RhoFileProperty::CompressedEncrypted
        );
        assert_eq!(
            file_property_for_extension(".xml", 10),
            RhoFileProperty::Encrypted
        );
        assert_eq!(
            file_property_for_extension(".uset", 10),
            RhoFileProperty::Encrypted
        );
        assert_eq!(
            file_property_for_extension(".kap", 10),
            RhoFileProperty::PartialEncrypted
        );
        assert_eq!(
            file_property_for_extension(".png", 256),
            RhoFileProperty::Encrypted
        );
        assert_eq!(
            file_property_for_extension(".png", 257),
            RhoFileProperty::PartialEncrypted
        );
        assert_eq!(
            file_property_for_extension(".bin", 10),
            RhoFileProperty::None
        );
    }
}
