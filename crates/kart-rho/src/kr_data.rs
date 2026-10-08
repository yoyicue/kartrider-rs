//! KRData envelope used by `Data/aaa.pk`.
//!
//! Launcher reads `aaa.pk` as a little-endian length followed by a KRData block
//! (`KartLibrary.Data.DataProcessor.ReadKRData`). This is intentionally kept
//! separate from [`crate::encoded_block`], which is the PIN/config wrapper.

use crate::crypto::{adler32, decrypt_data};

pub const TAG: u8 = 0x53;
pub const FLAG_ZLIB: u8 = 0x01;
pub const FLAG_RHO_CRYPTO: u8 = 0x02;

/// Decode the `i32 total_length` + KRData container used by `Data/aaa.pk`.
pub fn decode_len_prefixed(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = Reader::new(input);
    let len = r.i32()?;
    if len < 0 {
        return Err("negative KRData length".into());
    }
    let block = r.bytes(len as usize)?;
    if !r.remaining().is_empty() {
        return Err(format!(
            "{} trailing byte(s) after KRData",
            r.remaining().len()
        ));
    }
    decode(block)
}

/// Decode a standalone KRData block.
pub fn decode(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = Reader::new(input);
    let tag = r.u8()?;
    if tag != TAG {
        return Err("invalid KRData tag".into());
    }
    let flags = r.u8()?;
    if flags & !(FLAG_ZLIB | FLAG_RHO_CRYPTO) != 0 {
        return Err(format!("unsupported KRData flags 0x{flags:02x}"));
    }
    let checksum = r.u32()?;
    let crypto_key = if flags & FLAG_RHO_CRYPTO != 0 {
        Some(r.u32()?)
    } else {
        None
    };
    let clear_len = if flags & FLAG_ZLIB != 0 {
        Some(r.i32()?)
    } else {
        None
    };
    if let Some(len) = clear_len {
        if len < 0 {
            return Err("negative KRData decoded length".into());
        }
    }

    let mut payload = r.remaining().to_vec();
    if let Some(key) = crypto_key {
        decrypt_data(key, &mut payload);
    }
    if let Some(len) = clear_len {
        let expected = len as usize;
        let clear = miniz_oxide::inflate::decompress_to_vec_zlib(&payload)
            .map_err(|e| format!("zlib: {e:?}"))?;
        if clear.len() != expected {
            return Err(format!(
                "decoded length {} != expected {}",
                clear.len(),
                expected
            ));
        }
        payload = clear;
    }

    let actual = adler32(&payload);
    if actual != checksum {
        return Err(format!(
            "KRData checksum mismatch: got 0x{actual:08x}, expected 0x{checksum:08x}"
        ));
    }
    Ok(payload)
}

/// Encode helper for fixtures and future `aaa.pk` writer work.
pub fn encode(input: &[u8], flags: u8, crypto_key: Option<u32>) -> Result<Vec<u8>, String> {
    if flags & !(FLAG_ZLIB | FLAG_RHO_CRYPTO) != 0 {
        return Err(format!("unsupported KRData flags 0x{flags:02x}"));
    }
    let mut payload = if flags & FLAG_ZLIB != 0 {
        miniz_oxide::deflate::compress_to_vec_zlib(input, 6)
    } else {
        input.to_vec()
    };
    let key = if flags & FLAG_RHO_CRYPTO != 0 {
        let key = crypto_key.unwrap_or_else(|| adler32(input));
        decrypt_data(key, &mut payload);
        Some(key)
    } else {
        None
    };

    let mut out = Vec::with_capacity(2 + 4 + 4 + 4 + payload.len());
    out.push(TAG);
    out.push(flags);
    out.extend_from_slice(&adler32(input).to_le_bytes());
    if let Some(key) = key {
        out.extend_from_slice(&key.to_le_bytes());
    }
    if flags & FLAG_ZLIB != 0 {
        out.extend_from_slice(&(input.len() as i32).to_le_bytes());
    }
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn encode_len_prefixed(
    input: &[u8],
    flags: u8,
    crypto_key: Option<u32>,
) -> Result<Vec<u8>, String> {
    let block = encode(input, flags, crypto_key)?;
    let mut out = Vec::with_capacity(4 + block.len());
    out.extend_from_slice(&(block.len() as i32).to_le_bytes());
    out.extend_from_slice(&block);
    Ok(out)
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }

    fn u8(&mut self) -> Result<u8, String> {
        let b = *self.d.get(self.p).ok_or("short read u8")?;
        self.p += 1;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        let bytes = self.bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("offset overflow")?;
        let bytes = self.d.get(self.p..end).ok_or("short read bytes")?;
        self.p = end;
        Ok(bytes)
    }

    fn remaining(&self) -> &'a [u8] {
        &self.d[self.p..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zlib_len_prefixed_krdata() {
        let clear = b"<PackFolder name=\"KartRider\" />";
        let wrapped = encode_len_prefixed(clear, FLAG_ZLIB, None).unwrap();
        assert_eq!(decode_len_prefixed(&wrapped).unwrap(), clear);
    }

    #[test]
    fn roundtrip_zlib_crypto_krdata() {
        let clear = b"aaa.pk binary xml payload";
        let wrapped =
            encode_len_prefixed(clear, FLAG_ZLIB | FLAG_RHO_CRYPTO, Some(0x1234_5678)).unwrap();
        assert_eq!(decode_len_prefixed(&wrapped).unwrap(), clear);
    }
}
