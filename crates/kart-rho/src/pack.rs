//! Read-only parser for `Data/aaa.pk`.
//!
//! Launcher decodes this as KRData, then reads a UTF-16 BinaryXml root named
//! `PackFolder`. Its `RhoFolder` nodes are the runtime `.rho` mount list.

use crate::kr_data;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackIndex {
    pub root: BinaryTag,
    pub mounts: Vec<RhoMount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RhoMount {
    pub mount_path: String,
    pub name: String,
    pub file_name: String,
    pub key: Option<u32>,
    pub data_hash: Option<u32>,
    pub media_size: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BinaryTag {
    pub name: String,
    pub text: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<BinaryTag>,
}

impl BinaryTag {
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

pub fn read_pack_index(path: &Path) -> Result<PackIndex, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    PackIndex::from_encrypted_bytes(&bytes)
}

impl PackIndex {
    pub fn from_encrypted_bytes(bytes: &[u8]) -> Result<Self, String> {
        let clear = kr_data::decode_len_prefixed(bytes)?;
        Self::from_clear_bytes(&clear)
    }

    pub fn from_clear_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(bytes);
        let root = r.tag()?;
        if r.remaining() != 0 {
            return Err(format!(
                "{} trailing byte(s) after PackFolder root",
                r.remaining()
            ));
        }
        if root.name != "PackFolder" {
            return Err(format!("invalid aaa.pk root tag {:?}", root.name));
        }
        let mut mounts = Vec::new();
        collect_mounts(&root, "", true, &mut mounts);
        Ok(Self { root, mounts })
    }
}

fn collect_mounts(tag: &BinaryTag, parent: &str, parent_is_root: bool, mounts: &mut Vec<RhoMount>) {
    for child in &tag.children {
        match child.name.as_str() {
            "PackFolder" => {
                let name = child.attr("name").unwrap_or("");
                let folder_name = if parent_is_root {
                    format!("{name}_")
                } else {
                    name.to_string()
                };
                let path = join_path(parent, &folder_name);
                collect_mounts(child, &path, false, mounts);
            }
            "RhoFolder" => {
                let name = child.attr("name").unwrap_or("").to_string();
                let file_name = child.attr("fileName").unwrap_or("").to_string();
                if file_name.is_empty() {
                    continue;
                }
                let mount_path = if name.is_empty() {
                    parent.to_string()
                } else {
                    join_path(parent, &name)
                };
                mounts.push(RhoMount {
                    mount_path,
                    name,
                    file_name,
                    key: parse_u32_attr(child, "key"),
                    data_hash: parse_u32_attr(child, "dataHash"),
                    media_size: parse_u64_attr(child, "mediaSize"),
                });
            }
            _ => {}
        }
    }
}

fn join_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else if child.is_empty() {
        parent.to_string()
    } else {
        format!("{parent}/{child}")
    }
}

fn parse_u32_attr(tag: &BinaryTag, key: &str) -> Option<u32> {
    tag.attr(key)?.parse().ok()
}

fn parse_u64_attr(tag: &BinaryTag, key: &str) -> Option<u64> {
    tag.attr(key)?.parse().ok()
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

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("offset overflow")?;
        let bytes = self.d.get(self.p..end).ok_or("short read bytes")?;
        self.p = end;
        Ok(bytes)
    }

    fn i32(&mut self) -> Result<i32, String> {
        let bytes = self.bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let chars = self.i32()?;
        if chars < 0 {
            return Err("negative BinaryXml string length".into());
        }
        let bytes_len = (chars as usize)
            .checked_mul(2)
            .ok_or("BinaryXml string length overflow")?;
        let bytes = self.bytes(bytes_len)?;
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(String::from_utf16_lossy(&units))
    }

    fn count(&mut self, label: &str) -> Result<usize, String> {
        let n = self.i32()?;
        if n < 0 {
            return Err(format!("negative {label}"));
        }
        Ok(n as usize)
    }

    fn tag(&mut self) -> Result<BinaryTag, String> {
        let name = self.string()?;
        let text = self.string()?;
        let attr_count = self.count("BinaryXml attr count")?;
        let mut attributes = Vec::with_capacity(attr_count);
        for _ in 0..attr_count {
            attributes.push((self.string()?, self.string()?));
        }
        let child_count = self.count("BinaryXml child count")?;
        let mut children = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            children.push(self.tag()?);
        }
        Ok(BinaryTag {
            name,
            text,
            attributes,
            children,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kr_data::{encode_len_prefixed, FLAG_ZLIB};

    #[test]
    fn parses_packfolder_mounts() {
        let root = BinaryTag {
            name: "PackFolder".into(),
            children: vec![BinaryTag {
                name: "PackFolder".into(),
                attributes: vec![("name".into(), "track".into())],
                children: vec![BinaryTag {
                    name: "RhoFolder".into(),
                    attributes: vec![
                        ("name".into(), "common".into()),
                        ("fileName".into(), "track_common.rho".into()),
                        ("key".into(), "123".into()),
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let clear = to_bytes(&root);
        let wrapped = encode_len_prefixed(&clear, FLAG_ZLIB, None).unwrap();

        let index = PackIndex::from_encrypted_bytes(&wrapped).unwrap();
        assert_eq!(index.mounts.len(), 1);
        assert_eq!(index.mounts[0].mount_path, "track_/common");
        assert_eq!(index.mounts[0].file_name, "track_common.rho");
        assert_eq!(index.mounts[0].key, Some(123));
    }

    fn to_bytes(tag: &BinaryTag) -> Vec<u8> {
        let mut out = Vec::new();
        write_tag(&mut out, tag);
        out
    }

    fn write_tag(out: &mut Vec<u8>, tag: &BinaryTag) {
        write_string(out, &tag.name);
        write_string(out, &tag.text);
        out.extend_from_slice(&(tag.attributes.len() as i32).to_le_bytes());
        for (k, v) in &tag.attributes {
            write_string(out, k);
            write_string(out, v);
        }
        out.extend_from_slice(&(tag.children.len() as i32).to_le_bytes());
        for child in &tag.children {
            write_tag(out, child);
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        let units: Vec<u16> = s.encode_utf16().collect();
        out.extend_from_slice(&(units.len() as i32).to_le_bytes());
        for u in units {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }
}
