//! Typed view over `track_common.rho/randomTrack@<cc>.bml` — the **random-match
//! track rotations** (which tracks the "random track" pick draws from per game
//! type / level), parsed from [`crate::bml`].
//!
//! `<RandomTrackSetList cc>` holds `<RandomTrackSet gameType randomType level>`
//! entries whose `<track id>` children are the eligible tracks. Some sets sit
//! directly under the root; others are grouped under `<RandomTrackList
//! randomType="new"|"reverse">`. The parser flattens both, recording the parent
//! group on each set so nothing about the structure is lost.
//!
//! The track ids reference the [`crate::track_list`] registry, so referential
//! integrity is checkable (every rotation track should be a defined track).

use crate::bml::{self, Tag};

/// One `<RandomTrackSet>`: the eligible tracks for a (gameType, randomType, level)
/// random pick.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RandomTrackSet {
    pub game_type: String,
    pub random_type: String,
    pub level: Option<i32>,
    /// `randomType` of the enclosing `<RandomTrackList>` group, if any.
    pub group: Option<String>,
    /// Track ids (`<track id>` children).
    pub tracks: Vec<String>,
    pub attrs: Vec<(String, String)>,
}

/// The parsed `<RandomTrackSetList>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RandomTrackSetList {
    pub cc: String,
    pub sets: Vec<RandomTrackSet>,
}

impl RandomTrackSetList {
    /// Every track id referenced by any set (with duplicates across sets).
    pub fn track_refs(&self) -> impl Iterator<Item = &str> {
        self.sets
            .iter()
            .flat_map(|s| s.tracks.iter().map(String::as_str))
    }
}

fn attr<'a>(t: &'a Tag, k: &str) -> Option<&'a str> {
    t.attributes
        .iter()
        .find(|(ak, _)| ak == k)
        .map(|(_, v)| v.as_str())
}

/// Parse a `randomTrack@<cc>.bml` (or any `<RandomTrackSetList>`) buffer.
pub fn parse(bml_bytes: &[u8]) -> Option<RandomTrackSetList> {
    let root = bml::parse(bml_bytes)?;
    if root.name != "RandomTrackSetList" {
        return None;
    }
    let mut out = RandomTrackSetList {
        cc: attr(&root, "cc").unwrap_or_default().to_string(),
        sets: Vec::new(),
    };
    collect(&root, None, &mut out.sets);
    Some(out)
}

/// Recursively gather `<RandomTrackSet>`s, descending through `<RandomTrackList>`
/// groups and tagging each set with its enclosing group's `randomType`.
fn collect(tag: &Tag, group: Option<&str>, out: &mut Vec<RandomTrackSet>) {
    for c in &tag.children {
        match c.name.as_str() {
            "RandomTrackSet" => {
                out.push(RandomTrackSet {
                    game_type: attr(c, "gameType").unwrap_or_default().to_string(),
                    random_type: attr(c, "randomType").unwrap_or_default().to_string(),
                    level: attr(c, "level").and_then(|v| v.parse().ok()),
                    group: group.map(str::to_string),
                    tracks: c
                        .children_named("track")
                        .filter_map(|t| attr(t, "id").map(str::to_string))
                        .collect(),
                    attrs: c.attributes.clone(),
                });
            }
            "RandomTrackList" => collect(c, attr(c, "randomType").or(group), out),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bml::Tag;

    fn t(name: &str, attrs: &[(&str, &str)], children: Vec<Tag>) -> Tag {
        Tag {
            name: name.into(),
            attributes: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            children,
            ..Default::default()
        }
    }
    fn track(id: &str) -> Tag {
        t("track", &[("id", id)], vec![])
    }

    #[test]
    fn flattens_groups_and_collects_refs() {
        let root = t(
            "RandomTrackSetList",
            &[("cc", "cn")],
            vec![
                t(
                    "RandomTrackSet",
                    &[("gameType", "item"), ("randomType", "hot1"), ("level", "1")],
                    vec![track("desert_I03"), track("forest_I07")],
                ),
                t(
                    "RandomTrackList",
                    &[("randomType", "reverse")],
                    vec![t(
                        "RandomTrackSet",
                        &[("gameType", "speed"), ("randomType", "rev1")],
                        vec![track("ice_R08")],
                    )],
                ),
            ],
        );
        let rl = parse(&root.to_bytes()).expect("parse");
        assert_eq!(rl.cc, "cn");
        assert_eq!(rl.sets.len(), 2);
        assert_eq!(rl.sets[0].tracks, vec!["desert_I03", "forest_I07"]);
        assert_eq!(rl.sets[0].group, None); // top-level set
        assert_eq!(rl.sets[1].group.as_deref(), Some("reverse")); // nested under group
        assert_eq!(rl.sets[1].tracks, vec!["ice_R08"]);
        assert_eq!(
            rl.track_refs().collect::<Vec<_>>(),
            vec!["desert_I03", "forest_I07", "ice_R08"]
        );
    }

    #[test]
    fn rejects_non_randomtrack() {
        assert!(parse(
            &Tag {
                name: "trackList".into(),
                ..Default::default()
            }
            .to_bytes()
        )
        .is_none());
    }
}
