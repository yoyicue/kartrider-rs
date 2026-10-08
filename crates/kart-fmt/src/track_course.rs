//! Embedded TrackContainer `property/course` metadata.
//!
//! Legacy track `.1s` files append serialized event KartObjects after the scene
//! graph. One event owns a BinaryXml field whose root is `property`; its
//! `course` / `courseReverse` children name the authored road meshes and their
//! entry/exit markers. The scene parser historically retained only
//! `event_count`, which made path-query graphs look process-memory-only.

use crate::bml::{self, Tag};

const NEW_FIELD_MARKER: [u8; 2] = [0xAA, 0x27];

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedTrackProperty {
    pub offset: usize,
    pub encoded_len: usize,
    pub root: Tag,
}

impl EmbeddedTrackProperty {
    pub fn course(&self, reverse: bool) -> Option<&Tag> {
        let name = if reverse { "courseReverse" } else { "course" };
        self.root.children_named(name).next()
    }
}

/// Find the event BinaryXml field containing `property/course`.
///
/// A new KartObject field is framed as `AA 27 <i16 field-id>` followed directly
/// by the BinaryXml tag. The full candidate tree and its semantic root are
/// validated before acceptance; UTF-16 string hits are not treated as assets.
pub fn find_embedded_property(data: &[u8]) -> Option<EmbeddedTrackProperty> {
    data.windows(NEW_FIELD_MARKER.len())
        .enumerate()
        .filter(|(_, marker)| **marker == NEW_FIELD_MARKER)
        .filter_map(|(marker_offset, _)| {
            let offset = marker_offset.checked_add(4)?;
            let (root, encoded_len) = bml::parse_prefix(data.get(offset..)?)?;
            (root.name == "property" && root.children.iter().any(|child| child.name == "course"))
                .then_some(EmbeddedTrackProperty {
                    offset,
                    encoded_len,
                    root,
                })
        })
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn property() -> Tag {
        Tag {
            name: "property".into(),
            children: vec![Tag {
                name: "course".into(),
                children: vec![Tag {
                    name: "road".into(),
                    attributes: vec![
                        ("name".into(), "RoadObj_rail1+".into()),
                        ("start".into(), "start".into()),
                        ("end".into(), "end".into()),
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn finds_framed_property_without_requiring_asset_eof() {
        let expected = property();
        let mut bytes = vec![0xCC, 0x01, 0xAA, 0x27, 0x34, 0x12];
        bytes.extend(expected.to_bytes());
        bytes.extend([0xAA, 0x47, 0x00]);
        let found = find_embedded_property(&bytes).expect("property");
        assert_eq!(found.offset, 6);
        assert_eq!(found.encoded_len, expected.to_bytes().len());
        assert_eq!(found.root, expected);
        assert!(found.course(false).is_some());
        assert!(found.course(true).is_none());
    }

    #[test]
    fn rejects_unframed_or_non_course_bml() {
        assert!(find_embedded_property(&property().to_bytes()).is_none());
        let root = Tag {
            name: "property".into(),
            ..Default::default()
        };
        let mut bytes = vec![0xAA, 0x27, 0, 0];
        bytes.extend(root.to_bytes());
        assert!(find_embedded_property(&bytes).is_none());
    }
}
