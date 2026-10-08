//! Typed view over `track_common.rho/challenge.bml` — the **license / challenge
//! mode** progression, parsed from [`crate::bml`].
//!
//! `<challengeList>` is a flat list of `<challenge>` entries. Each binds a track
//! and rules (`time`, item slots, AI setup, UI flags, …) and — except the very
//! first — names a `require`: the id of the challenge that must be cleared first.
//! Those `require` edges form a progression DAG.
//!
//! Structurally important attributes are promoted to fields; the full attribute
//! set is kept verbatim in `attrs` so nothing is lost. Unknown child tags land in
//! [`ChallengeList::other`] rather than being dropped.

use crate::bml::{self, Tag};

/// A `<challenge>` entry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Challenge {
    /// `id`, a hex string like `"0x00"` (kept verbatim — it is also what `require`
    /// references).
    pub id: String,
    /// `type` (1–4; the field is renamed since `type` is a Rust keyword).
    pub kind: i32,
    /// `track` this challenge runs on (a license/course track id).
    pub track: Option<String>,
    /// `require` — the `id` of the prerequisite challenge (`None` for the root).
    pub require: Option<String>,
    /// `time` limit in ms.
    pub time_ms: Option<i32>,
    /// Every attribute verbatim, in file order (including the promoted ones).
    pub attrs: Vec<(String, String)>,
}

/// The parsed `<challengeList>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChallengeList {
    pub challenges: Vec<Challenge>,
    /// Any child that is not a `<challenge>` — surfaced rather than dropped.
    pub other: Vec<Tag>,
}

impl Challenge {
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

impl ChallengeList {
    /// Challenges with no `require` — the entry point(s) of the progression.
    pub fn roots(&self) -> impl Iterator<Item = &Challenge> {
        self.challenges.iter().filter(|c| c.require.is_none())
    }

    /// `require` edges that point to an id no challenge defines (a broken
    /// prerequisite). Empty ⇒ the progression DAG is self-consistent.
    pub fn dangling_requires(&self) -> Vec<(&str, &str)> {
        let ids: std::collections::HashSet<&str> =
            self.challenges.iter().map(|c| c.id.as_str()).collect();
        self.challenges
            .iter()
            .filter_map(|c| {
                c.require
                    .as_deref()
                    .filter(|r| !ids.contains(r))
                    .map(|r| (c.id.as_str(), r))
            })
            .collect()
    }
}

/// Parse a `challenge.bml` (or any `<challengeList>` BML) buffer.
pub fn parse(bml_bytes: &[u8]) -> Option<ChallengeList> {
    let root = bml::parse(bml_bytes)?;
    (root.name == "challengeList").then(|| from_tag(&root))
}

/// Build the list from an already-parsed `<challengeList>` tag.
pub fn from_tag(root: &Tag) -> ChallengeList {
    let mut out = ChallengeList::default();
    for c in &root.children {
        if c.name != "challenge" {
            out.other.push(c.clone());
            continue;
        }
        let get = |k: &str| {
            c.attributes
                .iter()
                .find(|(ak, _)| ak == k)
                .map(|(_, v)| v.as_str())
        };
        let Some(id) = get("id") else {
            out.other.push(c.clone()); // a <challenge> with no id is malformed → surface it
            continue;
        };
        out.challenges.push(Challenge {
            id: id.to_string(),
            kind: get("type").and_then(|v| v.parse().ok()).unwrap_or(0),
            track: get("track").map(str::to_string),
            require: get("require").map(str::to_string),
            time_ms: get("time").and_then(|v| v.parse().ok()),
            attrs: c.attributes.clone(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bml::Tag;

    fn ch(attrs: &[(&str, &str)]) -> Tag {
        Tag {
            name: "challenge".into(),
            attributes: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn parses_and_validates_dag() {
        let root = Tag {
            name: "challengeList".into(),
            children: vec![
                ch(&[("id", "0x00"), ("type", "2"), ("track", "village_C005")]), // root, no require
                ch(&[
                    ("id", "0x01"),
                    ("type", "1"),
                    ("require", "0x00"),
                    ("track", "village_L01"),
                    ("time", "10000"),
                ]),
                ch(&[("id", "0x02"), ("type", "1"), ("require", "0xFF")]), // dangling require
                Tag {
                    name: "note".into(),
                    ..Default::default()
                }, // unknown child → other
            ],
            ..Default::default()
        };
        let cl = parse(&root.to_bytes()).expect("parse");
        assert_eq!(cl.challenges.len(), 3);
        assert_eq!(cl.other.len(), 1);
        assert_eq!(cl.roots().count(), 1); // only 0x00
        assert_eq!(cl.challenges[1].require.as_deref(), Some("0x00"));
        assert_eq!(cl.challenges[1].time_ms, Some(10000));
        assert_eq!(cl.challenges[1].kind, 1);
        assert_eq!(cl.dangling_requires(), vec![("0x02", "0xFF")]); // 0xFF undefined
    }

    #[test]
    fn rejects_non_challengelist() {
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
