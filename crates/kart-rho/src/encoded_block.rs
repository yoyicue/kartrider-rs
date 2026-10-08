//! `KREncodedBlock` envelope used by launcher/runtime config files such as
//! `KartRider.pin`.
//!
//! This is the same small wrapper the managed Launcher payload uses: byte tag
//! `0x53`, flags, an Adler checksum over the clear payload, optional KartCrypto,
//! and optional zlib framing.

use crate::crypto::{adler32, decrypt_data, simple_adler32};

pub const TAG: u8 = 0x53;
pub const FLAG_ZLIB: u8 = 0x01;
pub const FLAG_KART_CRYPTO: u8 = 0x02;

/// Decode the `i32 length` + `KREncodedBlock` container used by `KartRider.pin`
/// and similar runtime config blocks. `Data/aaa.pk` uses [`crate::kr_data`]
/// instead.
pub fn decode_len_prefixed(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = Reader::new(input);
    let len = r.i32()?;
    if len < 0 {
        return Err("negative KREncodedBlock length".into());
    }
    let block = r.bytes(len as usize)?;
    if !r.remaining().is_empty() {
        return Err(format!(
            "{} trailing byte(s) after KREncodedBlock",
            r.remaining().len()
        ));
    }
    decode(block)
}

/// Decode a `KREncodedBlock`. If the buffer does not start with the block tag,
/// Launcher returns the original bytes; this mirrors that behavior.
pub fn decode(input: &[u8]) -> Result<Vec<u8>, String> {
    if input.first().copied() != Some(TAG) {
        return Ok(input.to_vec());
    }
    let mut r = Reader::new(input);
    let tag = r.u8()?;
    if tag != TAG {
        return Err("invalid KREncodedBlock tag".into());
    }
    let flags = r.u8()?;
    if flags & !(FLAG_ZLIB | FLAG_KART_CRYPTO) != 0 {
        return Err(format!("unsupported KREncodedBlock flags 0x{flags:02x}"));
    }
    let checksum = r.u32()?;
    let crypto_key = if flags & FLAG_KART_CRYPTO != 0 {
        Some(r.u32()?)
    } else {
        None
    };
    let clear_len = if flags & FLAG_ZLIB != 0 {
        Some(r.u32()? as usize)
    } else {
        None
    };
    let mut payload = r.remaining().to_vec();

    if let Some(key) = crypto_key {
        decrypt_data(key, &mut payload);
    }

    if let Some(expected_len) = clear_len {
        if payload.len() < 6 {
            return Err("zlib payload too short".into());
        }
        if payload[0] != 0x78 || payload[1] != 0xda {
            return Err(format!(
                "invalid zlib magic 0x{:02x}{:02x}",
                payload[0], payload[1]
            ));
        }
        let expected_adler = u32::from_be_bytes(
            payload[payload.len() - 4..]
                .try_into()
                .map_err(|_| "missing zlib adler")?,
        );
        let clear = miniz_oxide::inflate::decompress_to_vec_zlib(&payload)
            .map_err(|e| format!("zlib: {e:?}"))?;
        if clear.len() != expected_len {
            return Err(format!(
                "decoded length {} != expected {}",
                clear.len(),
                expected_len
            ));
        }
        if simple_adler32(&clear) != expected_adler {
            return Err("zlib adler mismatch".into());
        }
        payload = clear;
    }

    let actual = adler32(&payload);
    if actual != checksum {
        return Err(format!(
            "KREncodedBlock checksum mismatch: got 0x{actual:08x}, expected 0x{checksum:08x}"
        ));
    }
    Ok(payload)
}

/// Encode helper for tests and future round-trip work. This does not imply any
/// caller should rewrite runtime config files without an explicit higher-level
/// round-trip contract.
pub fn encode(input: &[u8], flags: u8, crypto_key: Option<u32>) -> Result<Vec<u8>, String> {
    if flags & !(FLAG_ZLIB | FLAG_KART_CRYPTO) != 0 {
        return Err(format!("unsupported KREncodedBlock flags 0x{flags:02x}"));
    }
    let mut payload = if flags & FLAG_ZLIB != 0 {
        miniz_oxide::deflate::compress_to_vec_zlib(input, 9)
    } else {
        input.to_vec()
    };

    let key = if flags & FLAG_KART_CRYPTO != 0 {
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
        out.extend_from_slice(&(input.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(&payload);
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
        let end = self.p.checked_add(4).ok_or("offset overflow")?;
        let bytes = self.d.get(self.p..end).ok_or("short read u32")?;
        self.p = end;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
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
    fn untagged_bytes_pass_through() {
        assert_eq!(decode(b"plain").unwrap(), b"plain");
    }

    #[test]
    fn roundtrip_plain_block() {
        let clear = b"pin payload";
        let block = encode(clear, 0, None).unwrap();
        assert_eq!(decode(&block).unwrap(), clear);
    }

    #[test]
    fn roundtrip_zlib_crypto_block() {
        let clear = b"KartRider PIN payload with enough bytes to compress.";
        let block = encode(clear, FLAG_ZLIB | FLAG_KART_CRYPTO, Some(862_557_747)).unwrap();
        assert_eq!(block[0], TAG);
        assert_eq!(block[1], FLAG_ZLIB | FLAG_KART_CRYPTO);
        assert_eq!(decode(&block).unwrap(), clear);
    }

    #[test]
    fn roundtrip_len_prefixed_block() {
        let clear = b"length-prefixed block";
        let block = encode(clear, FLAG_ZLIB | FLAG_KART_CRYPTO, Some(912_888_630)).unwrap();
        let mut wrapped = Vec::new();
        wrapped.extend_from_slice(&(block.len() as i32).to_le_bytes());
        wrapped.extend_from_slice(&block);
        assert_eq!(decode_len_prefixed(&wrapped).unwrap(), clear);
    }
}
