//! Port of `scripts/evaluate_jm_dj_trigger.py` — exact JM additive / DJ
//! replacement C0 writers and state14/15 transitions.

/// Python result key `policy` on `evaluate_jm_dj_trigger`.
pub const POLICY: &str = "The special-family gate suppresses only FX/state14-or-15; the C0 replacement and 0.3-second cooldown reset already occurred.";

/// Python `parse_jm_dj_surface_token(...)` result dict. The regex grammar is
/// `[+-]?\d+(?:\.\d+)?` numbers after an uppercased/space-stripped `JM` or
/// `DJ` prefix, full-match; the hand-rolled parser below mirrors it
/// exactly for ASCII tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum JmDjSurfaceToken {
    /// JM variant: two authored numbers scaled by 20.
    Jm {
        jm_horizontal_param: f64,
        jm_basis_param: f64,
        token: String,
    },
    /// DJ variant: three authored numbers, unscaled.
    Dj { dj_vector: [f64; 3], token: String },
}

fn is_pattern_space(byte: u8) -> bool {
    // Python re `\s` over an ASCII token: space, \t, \n, \r, \x0b, \x0c.
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Parse one `[+-]?\d+(?:\.\d+)?` number starting at `*cursor`; on success
/// advance the cursor past it and return the value.
fn parse_number(text: &[u8], cursor: &mut usize) -> Option<f64> {
    let start = *cursor;
    let mut index = start;
    if index < text.len() && (text[index] == b'+' || text[index] == b'-') {
        index += 1;
    }
    let digits_start = index;
    while index < text.len() && text[index].is_ascii_digit() {
        index += 1;
    }
    if index == digits_start {
        return None;
    }
    if index < text.len() && text[index] == b'.' {
        let fraction_start = index + 1;
        let mut fraction_end = fraction_start;
        while fraction_end < text.len() && text[fraction_end].is_ascii_digit() {
            fraction_end += 1;
        }
        if fraction_end > fraction_start {
            index = fraction_end;
        }
    }
    let slice = std::str::from_utf8(&text[start..index]).ok()?;
    let value: f64 = slice.parse().ok()?;
    *cursor = index;
    Some(value)
}

fn skip_spaces(text: &[u8], cursor: &mut usize) {
    while *cursor < text.len() && is_pattern_space(text[*cursor]) {
        *cursor += 1;
    }
}

fn expect_slash(text: &[u8], cursor: &mut usize) -> Option<()> {
    skip_spaces(text, cursor);
    if *cursor < text.len() && text[*cursor] == b'/' {
        *cursor += 1;
        Some(())
    } else {
        None
    }
}

fn parse_jm(normalized: &[u8]) -> Option<(f64, f64)> {
    let mut cursor = 2usize; // after "JM"
    skip_spaces(normalized, &mut cursor);
    let first = parse_number(normalized, &mut cursor)?;
    expect_slash(normalized, &mut cursor)?;
    skip_spaces(normalized, &mut cursor);
    let second = parse_number(normalized, &mut cursor)?;
    skip_spaces(normalized, &mut cursor);
    if cursor != normalized.len() {
        return None;
    }
    Some((first, second))
}

fn parse_dj(normalized: &[u8]) -> Option<[f64; 3]> {
    let mut cursor = 2usize; // after "DJ"
    let mut values = [0.0; 3];
    skip_spaces(normalized, &mut cursor);
    values[0] = parse_number(normalized, &mut cursor)?;
    for slot in values.iter_mut().skip(1) {
        expect_slash(normalized, &mut cursor)?;
        skip_spaces(normalized, &mut cursor);
        *slot = parse_number(normalized, &mut cursor)?;
    }
    skip_spaces(normalized, &mut cursor);
    if cursor != normalized.len() {
        return None;
    }
    Some(values)
}

/// `parse_jm_dj_surface_token(token)` — decode the authored JM/DJ payload
/// used by the contact dispatcher.
pub fn parse_jm_dj_surface_token(token: &str) -> Result<JmDjSurfaceToken, String> {
    let normalized = token.trim().to_uppercase();
    let bytes = normalized.as_bytes();
    if bytes.starts_with(b"JM") {
        if let Some((first, second)) = parse_jm(bytes) {
            return Ok(JmDjSurfaceToken::Jm {
                jm_horizontal_param: first * 20.0,
                jm_basis_param: second * 20.0,
                token: normalized,
            });
        }
    }
    if bytes.starts_with(b"DJ") {
        if let Some(values) = parse_dj(bytes) {
            return Ok(JmDjSurfaceToken::Dj { dj_vector: values, token: normalized });
        }
    }
    Err(format!("unsupported JM/DJ authored token: {token}"))
}

/// Python `state_transition` sub-dict: `{"state": 14|15, "remaining_ms": 1000}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JmDjStateTransition {
    pub state: i64,
    pub remaining_ms: i64,
}

/// Python `evaluate_jm_dj_trigger(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct JmDjTrigger {
    pub variant: String,
    pub branch: &'static str,
    pub velocity_c0_after: [f64; 3],
    pub accepted: bool,
    /// Present only when accepted (Python `None` otherwise).
    pub state_transition: Option<JmDjStateTransition>,
    pub cooldown_648_after: f64,
    pub velocity_write_occurs_before_state_family_gate: bool,
    pub velocity_operation: &'static str,
    pub policy: &'static str,
}

/// `evaluate_jm_dj_trigger(...)`. Component-count validation in Python is
/// unreachable here (fixed `[f64; 3]` inputs); presence validation of the
/// per-variant optionals is kept with the Python messages.
pub fn evaluate_jm_dj_trigger(
    variant: &str,
    special_state_family_already_active: bool,
    velocity_c0_before: Option<[f64; 3]>,
    jm_basis_190: Option<[f64; 3]>,
    jm_horizontal_param: Option<f64>,
    jm_basis_param: Option<f64>,
    dj_vector: Option<[f64; 3]>,
    boost_multiplier_64c: bool,
) -> Result<JmDjTrigger, String> {
    let (velocity, state, branch): ([f64; 3], i64, &'static str) = if variant == "JM" {
        let (Some(velocity_c0_before), Some(jm_basis_190), Some(jm_horizontal_param), Some(jm_basis_param)) =
            (velocity_c0_before, jm_basis_190, jm_horizontal_param, jm_basis_param)
        else {
            return Err(
                "JM requires velocity_c0_before, jm_basis_190, jm_horizontal_param, and jm_basis_param"
                    .to_string(),
            );
        };
        let velocity = [
            velocity_c0_before[0] + jm_basis_190[0] * jm_basis_param,
            velocity_c0_before[1] + jm_basis_190[1] * jm_basis_param,
            velocity_c0_before[2] + jm_basis_190[2] * jm_basis_param + jm_horizontal_param,
        ];
        (velocity, 14, "jm_add_basis_and_world_z")
    } else if variant == "DJ" {
        let Some(dj_vector) = dj_vector else {
            return Err("DJ requires a three-component dj_vector".to_string());
        };
        let multiplier = if boost_multiplier_64c { 1.2 } else { 1.0 };
        let velocity = [
            dj_vector[0] * multiplier,
            dj_vector[1] * multiplier,
            dj_vector[2] * multiplier,
        ];
        (velocity, 15, "dj_direct_vector_optional_1p2")
    } else {
        return Err("variant must be JM or DJ".to_string());
    };
    let accepted = !special_state_family_already_active;
    Ok(JmDjTrigger {
        variant: variant.to_string(),
        branch,
        velocity_c0_after: velocity,
        accepted,
        state_transition: if accepted {
            Some(JmDjStateTransition { state, remaining_ms: 1000 })
        } else {
            None
        },
        cooldown_648_after: 0.3,
        velocity_write_occurs_before_state_family_gate: true,
        velocity_operation: if variant == "JM" { "additive" } else { "replacement" },
        policy: POLICY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_jm_and_dj_tokens() {
        // Constants computed with the Python authority:
        //   parse_jm_dj_surface_token("jm 3/4") ->
        //     {"variant": "JM", "jm_horizontal_param": 60.0,
        //      "jm_basis_param": 80.0, "token": "JM 3/4"}
        match parse_jm_dj_surface_token("jm 3/4").unwrap() {
            JmDjSurfaceToken::Jm { jm_horizontal_param, jm_basis_param, token } => {
                assert_eq!(jm_horizontal_param, 60.0);
                assert_eq!(jm_basis_param, 80.0);
                assert_eq!(token, "JM 3/4");
            }
            other => panic!("expected JM, got {other:?}"),
        }
        //   parse_jm_dj_surface_token(" DJ-1.5/+2 / -3.25 ") ->
        //     {"variant": "DJ", "dj_vector": [-1.5, 2.0, -3.25], "token": "DJ-1.5/+2 / -3.25"}
        match parse_jm_dj_surface_token(" DJ-1.5/+2 / -3.25 ").unwrap() {
            JmDjSurfaceToken::Dj { dj_vector, token } => {
                assert_eq!(dj_vector, [-1.5, 2.0, -3.25]);
                assert_eq!(token, "DJ-1.5/+2 / -3.25");
            }
            other => panic!("expected DJ, got {other:?}"),
        }
        assert_eq!(
            parse_jm_dj_surface_token("XX 1/2").unwrap_err(),
            "unsupported JM/DJ authored token: XX 1/2"
        );
        // Trailing garbage breaks the full match, like re.fullmatch.
        assert!(parse_jm_dj_surface_token("JM 1/2/3").is_err());
        assert!(parse_jm_dj_surface_token("DJ 1/2").is_err());
        assert!(parse_jm_dj_surface_token("JM 1./2").is_err());
        assert!(parse_jm_dj_surface_token("JM .5/2").is_err());
    }

    #[test]
    fn jm_adds_basis_and_world_z() {
        // Constants computed with the Python authority:
        //   evaluate_jm_dj_trigger(variant="JM",
        //     special_state_family_already_active=False,
        //     velocity_c0_before=[1,2,3], jm_basis_190=[0.5,-0.5,0.25],
        //     jm_horizontal_param=60.0, jm_basis_param=80.0)
        //   -> velocity [41, -38, 83], state14/1000ms.
        let r = evaluate_jm_dj_trigger(
            "JM",
            false,
            Some([1.0, 2.0, 3.0]),
            Some([0.5, -0.5, 0.25]),
            Some(60.0),
            Some(80.0),
            None,
            false,
        )
        .unwrap();
        assert_eq!(r.velocity_c0_after, [41.0, -38.0, 83.0]);
        assert_eq!(r.branch, "jm_add_basis_and_world_z");
        assert_eq!(r.velocity_operation, "additive");
        assert_eq!(r.state_transition, Some(JmDjStateTransition { state: 14, remaining_ms: 1000 }));
        assert_eq!(r.cooldown_648_after, 0.3);
        assert_eq!(
            evaluate_jm_dj_trigger("JM", false, None, None, None, None, None, false)
                .unwrap_err(),
            "JM requires velocity_c0_before, jm_basis_190, jm_horizontal_param, and jm_basis_param"
        );
    }

    #[test]
    fn dj_replaces_with_optional_1p2() {
        // Constants computed with the Python authority:
        //   evaluate_jm_dj_trigger(variant="DJ",
        //     special_state_family_already_active=True,
        //     dj_vector=[1,2,3], boost_multiplier_64c=True)
        //   -> velocity [1.2, 2.4, 3.5999999999999996], gate suppresses only the transition.
        let r = evaluate_jm_dj_trigger(
            "DJ",
            true,
            None,
            None,
            None,
            None,
            Some([1.0, 2.0, 3.0]),
            true,
        )
        .unwrap();
        assert_eq!(r.velocity_c0_after, [1.2, 2.4, 3.5999999999999996]);
        assert_eq!(r.branch, "dj_direct_vector_optional_1p2");
        assert_eq!(r.velocity_operation, "replacement");
        assert!(!r.accepted);
        assert_eq!(r.state_transition, None);
        assert_eq!(
            evaluate_jm_dj_trigger("DJ", false, None, None, None, None, None, false).unwrap_err(),
            "DJ requires a three-component dj_vector"
        );
        assert_eq!(
            evaluate_jm_dj_trigger("XX", false, None, None, None, None, None, false).unwrap_err(),
            "variant must be JM or DJ"
        );
    }
}
