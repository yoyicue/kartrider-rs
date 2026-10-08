//! Typed view over `track_common.rho/rail.bml` — **guide-rail physics presets**,
//! parsed from [`crate::bml`].
//!
//! `<railList>` is a flat list of `<rail>` presets (e.g. `normal`, `highSpeed`).
//! When a kart rides a guide rail (the `ToGuideRail` track trigger names one of
//! these by id), the preset's velocity bounds and accel / resist / gravity
//! factors drive its motion. Together with the vehicle physics in
//! [`crate::param`], this is the data a kart-game physics step consumes.

use crate::bml::{self, Tag};

/// A `<rail>` physics preset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rail {
    pub id: String,
    pub min_velocity: f32,
    pub max_velocity: f32,
    pub accel_factor: f32,
    pub resist_factor: f32,
    pub gravity_factor: f32,
    /// Every attribute verbatim (including any not promoted above).
    pub attrs: Vec<(String, String)>,
}

/// The parsed `<railList>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RailList {
    pub rails: Vec<Rail>,
    /// Any child that is not a `<rail>` — surfaced rather than dropped.
    pub other: Vec<Tag>,
}

impl RailList {
    pub fn by_id(&self, id: &str) -> Option<&Rail> {
        self.rails.iter().find(|r| r.id == id)
    }
}

/// Parse a `rail.bml` (or any `<railList>` BML) buffer.
pub fn parse(bml_bytes: &[u8]) -> Option<RailList> {
    let root = bml::parse(bml_bytes)?;
    if root.name != "railList" {
        return None;
    }
    let mut out = RailList::default();
    for c in &root.children {
        if c.name != "rail" {
            out.other.push(c.clone());
            continue;
        }
        let f = |k: &str| {
            c.attributes
                .iter()
                .find(|(ak, _)| ak == k)
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0.0)
        };
        let Some(id) = c
            .attributes
            .iter()
            .find(|(k, _)| k == "id")
            .map(|(_, v)| v.clone())
        else {
            out.other.push(c.clone());
            continue;
        };
        out.rails.push(Rail {
            id,
            min_velocity: f("minVelocity"),
            max_velocity: f("maxVelocity"),
            accel_factor: f("accelFactor"),
            resist_factor: f("resistFactor"),
            gravity_factor: f("gravityFactor"),
            attrs: c.attributes.clone(),
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bml::Tag;

    #[test]
    fn parses_rail_presets() {
        let root = Tag {
            name: "railList".into(),
            children: vec![Tag {
                name: "rail".into(),
                attributes: [
                    ("id", "highSpeed"),
                    ("minVelocity", "300"),
                    ("maxVelocity", "350"),
                    ("accelFactor", "2"),
                    ("resistFactor", "1"),
                    ("gravityFactor", "0.5"),
                ]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let rl = parse(&root.to_bytes()).expect("parse");
        assert_eq!(rl.rails.len(), 1);
        let r = rl.by_id("highSpeed").unwrap();
        assert_eq!((r.min_velocity, r.max_velocity), (300.0, 350.0));
        assert_eq!(r.accel_factor, 2.0);
        assert_eq!(r.gravity_factor, 0.5);
    }

    #[test]
    fn rejects_non_raillist() {
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
