//! .rho cryptographic primitives, ported clean-room from KartLibrary
//! (RhoKey / RhoEncrypt). Layer "Rh layer spec 1.1" is what every shipped
//! archive uses; the header/blockinfo path is `decrypt_header_info`, file/folder
//! data is the XOR stream `decrypt_data`.

use crate::vectors::ENCRYPT_VECTORS;

const ADLER_MOD: u32 = 65521;

/// Adler-32 over raw bytes (seed 0, matching KartLibrary's `Adler.Adler32(0, ...)`).
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (0u32, 0u32);
    for &x in data {
        a = (a + x as u32) % ADLER_MOD;
        b = (b + a) % ADLER_MOD;
    }
    (b << 16) | a
}

/// Standard/zlib Adler-32 over raw bytes (seed 1).
pub fn simple_adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % ADLER_MOD;
        b = (b + a) % ADLER_MOD;
    }
    (b << 16) | a
}

/// Adler-32 of a string encoded as UTF-16LE (KartRider hashes names this way).
pub fn adler32_utf16le(s: &str) -> u32 {
    let mut bytes = Vec::with_capacity(s.len() * 2);
    for u in s.encode_utf16() {
        bytes.push((u & 0xFF) as u8);
        bytes.push((u >> 8) as u8);
    }
    adler32(&bytes)
}

/// Archive key derived from the file name (without extension).
pub fn get_rho_key(name: &str) -> u32 {
    adler32_utf16le(name).wrapping_sub(0xa6ee_7565)
}

/// Key for folder/directory data blocks.
pub fn get_directory_data_key(rho_key: u32) -> u32 {
    rho_key.wrapping_add(0x2593_a9f1)
}

/// Key for a file's data block.
pub fn get_file_key(rho_key: u32, file_name: &str, ext_num: u32) -> u32 {
    adler32_utf16le(file_name)
        .wrapping_add(ext_num)
        .wrapping_add(rho_key.wrapping_sub(0x756d_e654))
}

/// 64-byte keystream expanded from a 32-bit key (XORed over data, period 64).
pub fn extend_key(key: u32) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut cur = key ^ 0x8473_fbc1;
    for i in 0..16 {
        out[i * 4..i * 4 + 4].copy_from_slice(&cur.to_le_bytes());
        cur = cur.wrapping_sub(0x7b8c_043f);
    }
    out
}

/// In-place XOR decrypt of file/folder data (symmetric; same as encrypt).
pub fn decrypt_data(key: u32, data: &mut [u8]) {
    let ek = extend_key(key);
    for (i, b) in data.iter_mut().enumerate() {
        *b ^= ek[i & 63];
    }
}

/// One step of the header/blockinfo cipher's keyed vector mix.
fn get_vector(value: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4 {
        out ^= ENCRYPT_VECTORS[i][((value >> (i * 8)) & 0xFF) as usize];
    }
    out
}

/// In-place decrypt of header / block-info data (CBC-like over u32 words).
/// `data.len()` must be a multiple of 4.
pub fn decrypt_header_info(data: &mut [u8], key: u32) {
    let mut cur_key = key;
    let mut a = 0u32;
    let n = data.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let enc = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let dec = enc ^ get_vector(cur_key) ^ a;
        data[off..off + 4].copy_from_slice(&dec.to_le_bytes());
        a = a.wrapping_add(dec);
        cur_key = cur_key.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values computed from KartLibrary's formulas (see立项 commit).
    #[test]
    fn rho_key_matches_reference() {
        assert_eq!(get_rho_key("character_bazzi"), 0xba41_90c7);
        assert_eq!(get_rho_key("kart"), 0x615f_8c4d);
        assert_eq!(get_rho_key("track_gold_I03"), 0xad47_8fc0);
    }

    #[test]
    fn simple_adler32_matches_zlib_seed() {
        assert_eq!(simple_adler32(b""), 0x0000_0001);
        assert_eq!(simple_adler32(b"hello kartrider"), 0x2f25_05fd);
    }

    #[test]
    fn directory_key_matches_reference() {
        assert_eq!(get_directory_data_key(0x615f_8c4d), 0x86f3_363e);
    }

    #[test]
    fn extend_key_matches_reference() {
        let ek = extend_key(0x615f_8c4d);
        assert_eq!(
            u32::from_le_bytes(ek[0..4].try_into().unwrap()),
            0xe52c_778c
        );
        assert_eq!(
            u32::from_le_bytes(ek[4..8].try_into().unwrap()),
            0x69a0_734d
        );
    }

    #[test]
    fn decrypt_data_is_involution() {
        let mut d = *b"hello kartrider rho test payload";
        let orig = d;
        decrypt_data(0x12345678, &mut d);
        assert_ne!(d, orig);
        decrypt_data(0x12345678, &mut d);
        assert_eq!(d, orig);
    }
}
