//! Typed view over `track_common.rho/track@zz.bml` — the client's authoritative
//! **track registry** (`<trackList>`), parsed from [`crate::bml`].
//!
//! `<trackList>` holds three observed child kinds, and the parser keeps every one
//! (nothing is silently dropped):
//!
//! - `<track id="forest_R03" gameType="speed" … texTheme="beach" folder="…"/>` — a base track.
//! - `<track_rvs refId="…" …/>` — a *reverse* variant of a base track.
//! - `<track_crz refId="…" …/>` — a *cruise* variant of a base track.
//!
//! Any other `track_*` child carrying a `refId` is captured as a [`Variant`] too
//! (tagged by its raw tag name), and anything unexpected lands in
//! [`TrackList::other`] rather than being discarded.
//!
//! This turns the previously filename-guessed resource graph into BML-confirmed
//! data: `texTheme` / `bgmTheme` are the real texture/music theme references
//! (which can differ from the `<theme>_…` id prefix), and `folder` is geometry
//! reuse. Structurally important attributes are promoted to fields; the full
//! attribute set is kept verbatim in `attrs` so nothing is lost.

use crate::bml::{self, Tag};

/// A `<track>` entry (a base track).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Track {
    pub id: String,
    pub game_type: String,
    /// `texTheme` — texture theme override; may be `"a|b"` alternates. `None` ⇒
    /// use [`Track::theme`].
    pub tex_theme: Option<String>,
    /// `bgmTheme` — music theme override.
    pub bgm_theme: Option<String>,
    /// `folder` — id of another track whose geometry folder this one reuses.
    pub folder: Option<String>,
    /// Every attribute verbatim, in file order (including the promoted ones).
    pub attrs: Vec<(String, String)>,
}

/// A `track_*` variant (`<track_rvs>` reverse, `<track_crz>` cruise, …) that
/// references a base track by `refId` and overrides a few attributes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Variant {
    /// The raw tag name, e.g. `"track_rvs"` or `"track_crz"`.
    pub kind: String,
    /// `refId` — the base [`Track::id`] this variant derives from.
    pub ref_id: String,
    pub attrs: Vec<(String, String)>,
}

/// The parsed `<trackList>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackList {
    pub tracks: Vec<Track>,
    /// Reverse / cruise / any other `refId`-bearing variant, tagged by `kind`.
    pub variants: Vec<Variant>,
    /// Any child that is neither a `<track>` nor a `refId`-bearing `track_*` —
    /// surfaced rather than dropped, so an unrecognized kind is visible.
    pub other: Vec<Tag>,
}

fn attr<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

impl Track {
    /// Default theme = the id prefix before the first `_` (e.g. `forest_R03` → `forest`).
    pub fn theme(&self) -> &str {
        self.id.split('_').next().unwrap_or(&self.id)
    }

    /// Texture theme(s) that supply this track's materials: the `texTheme`
    /// override split on `|`, else the default [`theme`](Track::theme).
    pub fn texture_themes(&self) -> Vec<&str> {
        match &self.tex_theme {
            Some(t) => t.split('|').filter(|s| !s.is_empty()).collect(),
            None => vec![self.theme()],
        }
    }

    /// Geometry folder this track loads from (its own id unless `folder` overrides).
    pub fn geometry_folder(&self) -> &str {
        self.folder.as_deref().unwrap_or(&self.id)
    }

    /// Raw attribute lookup for anything not promoted to a field.
    pub fn attr(&self, key: &str) -> Option<&str> {
        attr(&self.attrs, key)
    }
}

impl Variant {
    pub fn is_reverse(&self) -> bool {
        self.kind == "track_rvs"
    }
    pub fn is_cruise(&self) -> bool {
        self.kind == "track_crz"
    }
    /// The id other configs reference this variant by: `<refId>_<suffix>`, where
    /// the suffix is the tag name without the `track_` prefix (e.g. a
    /// `track_rvs` of `abyss_R02` is referenced as `abyss_R02_rvs`).
    pub fn derived_id(&self) -> String {
        format!(
            "{}_{}",
            self.ref_id,
            self.kind.strip_prefix("track_").unwrap_or(&self.kind)
        )
    }
    /// Base track's default theme (id prefix of `ref_id`).
    pub fn theme(&self) -> &str {
        self.ref_id.split('_').next().unwrap_or(&self.ref_id)
    }
    pub fn attr(&self, key: &str) -> Option<&str> {
        attr(&self.attrs, key)
    }
}

impl TrackList {
    /// The full id space other configs may reference: every base [`Track::id`]
    /// plus each variant's [`Variant::derived_id`] (e.g. `abyss_R02_rvs`).
    pub fn defined_ids(&self) -> std::collections::HashSet<String> {
        self.tracks
            .iter()
            .map(|t| t.id.clone())
            .chain(self.variants.iter().map(|v| v.derived_id()))
            .collect()
    }
}

/// Parse a `track@zz.bml` (or any `<trackList>` BML) buffer.
pub fn parse(bml_bytes: &[u8]) -> Option<TrackList> {
    let root = bml::parse(bml_bytes)?;
    (root.name == "trackList").then(|| from_tag(&root))
}

/// Build the registry from an already-parsed `<trackList>` tag.
pub fn from_tag(root: &Tag) -> TrackList {
    let mut out = TrackList::default();
    for c in &root.children {
        if c.name == "track" {
            let Some(id) = attr(&c.attributes, "id") else {
                out.other.push(c.clone()); // a <track> with no id is malformed → surface it
                continue;
            };
            out.tracks.push(Track {
                id: id.to_string(),
                game_type: attr(&c.attributes, "gameType")
                    .unwrap_or_default()
                    .to_string(),
                tex_theme: attr(&c.attributes, "texTheme").map(str::to_string),
                bgm_theme: attr(&c.attributes, "bgmTheme").map(str::to_string),
                folder: attr(&c.attributes, "folder").map(str::to_string),
                attrs: c.attributes.clone(),
            });
        } else if let Some(ref_id) = attr(&c.attributes, "refId") {
            // any track_* (track_rvs / track_crz / future) that references a base
            out.variants.push(Variant {
                kind: c.name.clone(),
                ref_id: ref_id.to_string(),
                attrs: c.attributes.clone(),
            });
        } else {
            out.other.push(c.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bml::Tag;

    fn tag(name: &str, attrs: &[(&str, &str)]) -> Tag {
        Tag {
            name: name.into(),
            attributes: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn parses_all_three_kinds_via_bml() {
        let root = Tag {
            name: "trackList".into(),
            children: vec![
                tag(
                    "track",
                    &[
                        ("id", "forest_R03"),
                        ("gameType", "speed"),
                        ("texTheme", "beach"),
                    ],
                ),
                tag(
                    "track",
                    &[
                        ("id", "korea_I02"),
                        ("texTheme", "olympos|world"),
                        ("bgmTheme", "lotte"),
                    ],
                ),
                tag("track", &[("id", "desert_R05"), ("folder", "desert_I02")]),
                tag(
                    "track_rvs",
                    &[("refId", "forest_B02"), ("gameType", "bigTrack")],
                ),
                tag("track_crz", &[("refId", "forest_I03"), ("level", "2")]),
                tag("mystery", &[("foo", "bar")]), // unknown child → other, not dropped
            ],
            ..Default::default()
        };
        let tl = parse(&root.to_bytes()).expect("parse");
        assert_eq!(tl.tracks.len(), 3);
        assert_eq!(tl.variants.len(), 2);
        assert_eq!(tl.other.len(), 1); // the <mystery> child surfaced

        let forest = &tl.tracks[0];
        assert_eq!(forest.theme(), "forest");
        assert_eq!(forest.texture_themes(), vec!["beach"]); // override ≠ id prefix
        assert_eq!(tl.tracks[1].texture_themes(), vec!["olympos", "world"]);
        assert_eq!(tl.tracks[2].geometry_folder(), "desert_I02");

        let rvs = tl.variants.iter().find(|v| v.is_reverse()).unwrap();
        assert_eq!(rvs.ref_id, "forest_B02");
        assert_eq!(rvs.theme(), "forest");
        assert_eq!(rvs.derived_id(), "forest_B02_rvs"); // id other configs reference it by
        assert!(tl
            .variants
            .iter()
            .any(|v| v.is_cruise() && v.ref_id == "forest_I03"));

        // full id space includes base ids + variant-derived ids
        let ids = tl.defined_ids();
        assert!(ids.contains("forest_R03")); // base
        assert!(ids.contains("forest_B02_rvs")); // reverse variant
        assert!(ids.contains("forest_I03_crz")); // cruise variant
    }

    #[test]
    fn rejects_non_tracklist() {
        assert!(parse(
            &Tag {
                name: "challengeList".into(),
                ..Default::default()
            }
            .to_bytes()
        )
        .is_none());
    }
}
