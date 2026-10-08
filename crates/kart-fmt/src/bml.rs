//! BML — KartRider's **BinaryXml**: a length-prefixed UTF-16LE tag tree
//! (`name`, `text`, attributes, children) used for track / pet / font / scene
//! metadata and config (`*.bml` in `track_common.rho`, `gui_font.rho`,
//! `stage_scene.rho`, per-model `param.bml`, …).
//!
//! Clean-room from the documented wire format, validated against the C#
//! `BinaryXmlDocument` / `BinaryXmlTag` reference:
//!   - string = `i32 charCount` then `charCount*2` bytes UTF-16LE
//!   - tag    = `name, text, i32 attrCount, attrCount×(key,value), i32 childCount, childCount×tag`
//!
//! A `.bml` file is exactly one root tag with no trailing bytes (the C# viewer
//! reads the whole buffer as a single tag). The same wire tags also appear
//! embedded inside KartObject scenes — see `kart_object::read_binary_xml_tag`.
//!
//! Unlike the raw-geometry parsers (which panic on a desynced stream, per the
//! crate-root contract), this one is **bounds-checked and fallible**: `parse`
//! returns `None` on a truncated or ill-framed buffer, since `.bml` is config
//! data we often probe loosely.

/// A parsed BinaryXml tag. Attributes keep file order (so re-serialization is
/// byte-faithful — the C# side uses a dictionary, but order is what round-trips).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tag {
    pub name: String,
    pub text: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Tag>,
}

impl Tag {
    /// First attribute value for `key`, if present.
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Direct children with the given tag name.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Tag> {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// Re-serialize to the BML wire format (inverse of [`parse`]). Used to prove
    /// round-trip byte-identity against the original file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_tag(&mut out, self);
        out
    }
}

/// Parse a whole `.bml` buffer as a single root tag. Returns `None` if the bytes
/// are truncated, ill-framed, or have trailing data after the root tag.
pub fn parse(data: &[u8]) -> Option<Tag> {
    let (tag, consumed) = parse_prefix(data)?;
    (consumed == data.len()).then_some(tag)
}

/// Parse one BinaryXml tag from the start of `data`, returning its byte length.
///
/// KartObject fields embed a complete BinaryXml tree inside a larger `.1s`
/// stream, so requiring EOF (as [`parse`] correctly does for standalone BML)
/// would discard authoritative track metadata such as `property/course`.
pub fn parse_prefix(data: &[u8]) -> Option<(Tag, usize)> {
    let mut r = Reader { d: data, p: 0 };
    let tag = r.tag()?;
    Some((tag, r.p))
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn i32(&mut self) -> Option<i32> {
        let end = self.p.checked_add(4)?;
        let v = i32::from_le_bytes(self.d.get(self.p..end)?.try_into().ok()?);
        self.p = end;
        Some(v)
    }

    fn text(&mut self) -> Option<String> {
        let count = self.i32()?;
        if count < 0 {
            return None;
        }
        let bytes = (count as usize).checked_mul(2)?;
        let end = self.p.checked_add(bytes)?;
        let slice = self.d.get(self.p..end)?;
        self.p = end;
        let units: Vec<u16> = slice
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Some(String::from_utf16_lossy(&units))
    }

    fn tag(&mut self) -> Option<Tag> {
        let name = self.text()?;
        let text = self.text()?;
        // Counts come straight from the file; don't pre-allocate from them (a
        // garbage count would OOM) — push as we go and let a short read fail.
        let att_count = self.i32()?;
        if att_count < 0 {
            return None;
        }
        let mut attributes = Vec::new();
        for _ in 0..att_count {
            let k = self.text()?;
            let v = self.text()?;
            attributes.push((k, v));
        }
        let sub_count = self.i32()?;
        if sub_count < 0 {
            return None;
        }
        let mut children = Vec::new();
        for _ in 0..sub_count {
            children.push(self.tag()?);
        }
        Some(Tag {
            name,
            text,
            attributes,
            children,
        })
    }
}

fn write_text(out: &mut Vec<u8>, s: &str) {
    let units: Vec<u16> = s.encode_utf16().collect();
    out.extend_from_slice(&(units.len() as i32).to_le_bytes());
    for u in units {
        out.extend_from_slice(&u.to_le_bytes());
    }
}

fn write_tag(out: &mut Vec<u8>, tag: &Tag) {
    write_text(out, &tag.name);
    write_text(out, &tag.text);
    out.extend_from_slice(&(tag.attributes.len() as i32).to_le_bytes());
    for (k, v) in &tag.attributes {
        write_text(out, k);
        write_text(out, v);
    }
    out.extend_from_slice(&(tag.children.len() as i32).to_le_bytes());
    for c in &tag.children {
        write_tag(out, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a BML buffer from a Tag (mirrors the C# writer) for a golden test.
    fn b(tag: &Tag) -> Vec<u8> {
        tag.to_bytes()
    }

    #[test]
    fn roundtrip_nested() {
        let root = Tag {
            name: "PetParam".into(),
            text: String::new(),
            attributes: vec![
                ("addModel".into(), "1".into()),
                ("scale".into(), "0.5".into()),
            ],
            children: vec![Tag {
                name: "Bone".into(),
                text: "spine".into(),
                attributes: vec![("id".into(), "3".into())],
                children: vec![],
            }],
        };
        let bytes = b(&root);
        let parsed = parse(&bytes).expect("parse");
        assert_eq!(parsed, root); // structural round-trip
        assert_eq!(parsed.to_bytes(), bytes); // byte round-trip
        assert_eq!(parsed.attr("addModel"), Some("1"));
        assert_eq!(parsed.children_named("Bone").count(), 1);
    }

    #[test]
    fn rejects_truncated_and_trailing() {
        let root = Tag {
            name: "A".into(),
            ..Default::default()
        };
        let good = b(&root);
        assert!(parse(&good).is_some());
        assert!(parse(&good[..good.len() - 1]).is_none()); // truncated
        let mut trailing = good.clone();
        trailing.push(0xff); // trailing garbage
        assert!(parse(&trailing).is_none());
        let (prefix, consumed) = parse_prefix(&trailing).expect("embedded prefix");
        assert_eq!(prefix, root);
        assert_eq!(consumed, good.len());
        assert!(parse(&[]).is_none());
    }
}
