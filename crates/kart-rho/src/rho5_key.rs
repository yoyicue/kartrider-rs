//! Rho5Key — per-file key derivation for `.rho5`.
//!
//! `get_file_key_u1` is FNV-1a over the region salt. `get_packed_file_key` builds
//! the 0x80-byte file key. KartLibrary's C# computes one value via an SSE2 log
//! (`func_E321D0`), but the following loop divides that result's worth of decimal
//! digits out of u1 — so it is exactly `floor(log10(u1))`, i.e. the decimal
//! digit count minus one. We compute the digit count directly, sidestepping the
//! SSE2 log entirely (the reason the C# build needed Rosetta). Validated
//! end-to-end vs the C# RhoUnpacker output.

/// FNV-1a (32-bit) over the salt string's chars — KartLibrary `GetFileKey_U1`.
pub fn get_file_key_u1(another_data: &str) -> i32 {
    let mut out = 0x811C9DC5u32 as i32;
    for c in another_data.chars() {
        let temp = (c as i32) ^ out;
        out = temp.wrapping_mul(0x0100_0193);
    }
    out
}

fn decimal_digits(mut u: u32) -> u32 {
    if u == 0 {
        return 1;
    }
    let mut d = 0;
    while u > 0 {
        u /= 10;
        d += 1;
    }
    d
}

fn utf16le_bytes(s: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(s.len() * 2);
    for u in s.encode_utf16() {
        b.push((u & 0xFF) as u8);
        b.push((u >> 8) as u8);
    }
    b
}

/// KartLibrary `GetPackedFileKey` — the 0x80-byte decryption key for a file's data.
pub fn get_packed_file_key(file_chksum: &[u8], u1: i32, file_name: &str) -> [u8; 0x80] {
    let file_name_data = utf16le_bytes(file_name);
    let fn_units = file_name.encode_utf16().count();

    // temp_esi = floor(log10(u1)) = digit count - 1 (see module note).
    let mut temp_ebx = u1 as u32;
    let temp_esi = decimal_digits(temp_ebx) as i32 - 1;
    let mut arr2 = vec![0i32; (temp_esi + 1) as usize];
    let mut i = temp_esi;
    while temp_ebx != 0 {
        arr2[i as usize] = (temp_ebx % 10) as i32; // C# uses the 0xCCCCCCCD /10 magic
        temp_ebx /= 10;
        i -= 1;
    }

    let len = arr2.len();
    let mut output = [0u8; 0x80];
    for i in 0..0x80usize {
        let mut a = (arr2[i % len] as i8 as i32) & (0x8000_0001u32 as i32);
        if a < 0 {
            a = ((a - 1) | -2) + 1;
        }
        let b = arr2[(i + 1) % len] as i8 as i32;
        let mut c = ((arr2[(i + 2) % len] as i8 as i32) + i as i32) & (0x8000_000Fu32 as i32);
        if c < 0 {
            c = ((c - 1) | -10) + 1;
        }
        let mut out_byte = (b + i as i32) % 5;
        out_byte = ((out_byte + file_chksum[c as usize] as i32 + a) & 0xFF) as i8 as i32;
        out_byte = out_byte.wrapping_mul(file_name_data[(i % fn_units) << 1] as i8 as i32);
        out_byte = out_byte.wrapping_add(i as i32);
        output[i] = out_byte as u8;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fnv_matches_reference() {
        assert_eq!(get_file_key_u1("d$Bjgfc8@dH4TQ?k") as u32, 0x58e0_2546);
    }
}
