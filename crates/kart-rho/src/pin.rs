//! Read-only parser for `KartRider.pin`.
//!
//! The local Launcher payload decodes PIN as a length-prefixed
//! [`crate::encoded_block`] and then reads a runtime config object containing
//! locale/version fields, auth methods, login endpoints, and trailing BML-like
//! objects.

use crate::encoded_block;
use std::path::Path;

const RTTI_PINOBJECT: u32 = 284_099_454;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinFile {
    pub header: PinHeader,
    pub auth_methods: Vec<AuthMethod>,
    pub nulls_before_bml_objects: usize,
    pub bml_objects: Vec<PinBmlObject>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinHeader {
    pub unk1: u8,
    pub locale_id: u16,
    pub unk2: u16,
    pub locale_type: u8,
    pub unk3: u8,
    pub minor_version: u16,
    pub unk4: u8,
    pub unk5: u8,
    pub login_type: u8,
    pub aes_key: String,
    pub url: String,
    pub patch_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthMethod {
    pub index: u8,
    pub name: String,
    pub account_config: Option<PinBmlObject>,
    pub login_servers: Vec<PinEndpoint>,
    pub extra_config: Option<PinBmlObject>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinEndpoint {
    pub ip: String,
    pub port: u16,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinBmlObject {
    pub name: String,
    pub value: String,
    pub values: Vec<(String, String)>,
    pub children: Vec<PinBmlObject>,
}

pub fn read_pin_file(path: &Path) -> Result<PinFile, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    PinFile::from_encrypted_bytes(&bytes)
}

impl PinFile {
    pub fn from_encrypted_bytes(bytes: &[u8]) -> Result<Self, String> {
        let clear = encoded_block::decode_len_prefixed(bytes)?;
        Self::from_clear_bytes(&clear)
    }

    pub fn from_clear_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(bytes);
        let magic = r.u32()?;
        if magic != RTTI_PINOBJECT {
            return Err(format!("invalid PIN magic 0x{magic:08x}"));
        }

        let header = PinHeader {
            unk1: r.u8()?,
            locale_id: r.u16()?,
            unk2: r.u16()?,
            locale_type: r.u8()?,
            unk3: r.u8()?,
            minor_version: r.u16()?,
            unk4: r.u8()?,
            unk5: r.u8()?,
            login_type: r.u8()?,
            aes_key: r.string()?,
            url: r.string()?,
            patch_url: r.string()?,
        };

        let auth_count = r.count("auth method count")?;
        let mut auth_methods = Vec::with_capacity(auth_count);
        for _ in 0..auth_count {
            let index = r.u8()?;
            let name = r.string()?;
            let account_config = r.optional_bml()?;
            let server_count = r.count("login server count")?;
            let mut login_servers = Vec::with_capacity(server_count);
            for _ in 0..server_count {
                let a = r.u8()?;
                let b = r.u8()?;
                let c = r.u8()?;
                let d = r.u8()?;
                let port = r.u16()?;
                login_servers.push(PinEndpoint {
                    ip: format!("{a}.{b}.{c}.{d}"),
                    port,
                });
            }
            let extra_config = r.optional_bml()?;
            auth_methods.push(AuthMethod {
                index,
                name,
                account_config,
                login_servers,
                extra_config,
            });
        }

        let mut nulls_before_bml_objects = 0usize;
        while r.remaining() > 0 && r.peek_u8() == Some(0) {
            r.u8()?;
            nulls_before_bml_objects += 1;
        }

        let mut bml_objects = Vec::new();
        while r.remaining() > 0 {
            if let Some(bml) = r.optional_bml()? {
                bml_objects.push(bml);
            }
        }

        Ok(Self {
            header,
            auth_methods,
            nulls_before_bml_objects,
            bml_objects,
        })
    }
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }

    fn remaining(&self) -> usize {
        self.d.len().saturating_sub(self.p)
    }

    fn peek_u8(&self) -> Option<u8> {
        self.d.get(self.p).copied()
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("offset overflow")?;
        let bytes = self.d.get(self.p..end).ok_or("short read bytes")?;
        self.p = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        let b = *self.d.get(self.p).ok_or("short read u8")?;
        self.p += 1;
        Ok(b)
    }

    fn bool(&mut self) -> Result<bool, String> {
        Ok(self.u8()? == 1)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let bytes = self.bytes(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }

    fn count(&mut self, label: &str) -> Result<usize, String> {
        let n = self.i32()?;
        if n < 0 {
            return Err(format!("negative {label}"));
        }
        Ok(n as usize)
    }

    fn string(&mut self) -> Result<String, String> {
        let chars = self.count("string length")?;
        let bytes_len = chars.checked_mul(2).ok_or("string length overflow")?;
        let bytes = self.bytes(bytes_len)?;
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(String::from_utf16_lossy(&units))
    }

    fn optional_bml(&mut self) -> Result<Option<PinBmlObject>, String> {
        if self.bool()? {
            Ok(Some(self.bml()?))
        } else {
            Ok(None)
        }
    }

    fn bml(&mut self) -> Result<PinBmlObject, String> {
        let name = self.string()?;
        let value = self.string()?;
        let value_count = self.count("BML value count")?;
        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push((self.string()?, self.string()?));
        }
        let child_count = self.count("BML child count")?;
        let mut children = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            children.push(self.bml()?);
        }
        Ok(PinBmlObject {
            name,
            value,
            values,
            children,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoded_block::{encode, FLAG_KART_CRYPTO, FLAG_ZLIB};

    #[test]
    fn parses_minimal_pin() {
        let mut clear = Vec::new();
        clear.extend_from_slice(&RTTI_PINOBJECT.to_le_bytes());
        clear.push(0);
        clear.extend_from_slice(&1u16.to_le_bytes());
        clear.extend_from_slice(&2u16.to_le_bytes());
        clear.push(3);
        clear.push(4);
        clear.extend_from_slice(&123u16.to_le_bytes());
        clear.push(5);
        clear.push(6);
        clear.push(7);
        write_string(&mut clear, "aes");
        write_string(&mut clear, "url");
        write_string(&mut clear, "patch");
        clear.extend_from_slice(&0i32.to_le_bytes());

        let block = encode(&clear, FLAG_ZLIB | FLAG_KART_CRYPTO, Some(862_557_747)).unwrap();
        let mut encrypted = Vec::new();
        encrypted.extend_from_slice(&(block.len() as i32).to_le_bytes());
        encrypted.extend_from_slice(&block);

        let pin = PinFile::from_encrypted_bytes(&encrypted).unwrap();
        assert_eq!(pin.header.locale_id, 1);
        assert_eq!(pin.header.minor_version, 123);
        assert_eq!(pin.header.patch_url, "patch");
        assert!(pin.auth_methods.is_empty());
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        let units: Vec<u16> = s.encode_utf16().collect();
        out.extend_from_slice(&(units.len() as i32).to_le_bytes());
        for u in units {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
}
