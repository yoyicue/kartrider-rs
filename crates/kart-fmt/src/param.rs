//! Vehicle physics parser for `kart/<vehicle>/param.xml` — the oracle-local
//! `karttools` capability the C# glTF converter lacks. The file is a UTF-16
//! `<BodyParam .../>` element whose attributes are stringly-typed; the runtime
//! treats them as mixed bool/int/float/string. We coerce them (preserving the
//! original CamelCase names so the data round-trips) exactly like the oracle's
//! `kart_params.py`: explicit key registries first, then a numeric/bool auto
//! heuristic, else the raw string.

/// A coerced parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    /// JSON-encode this value (so a tool can emit a typed `param.json`).
    pub fn to_json(&self) -> String {
        match self {
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => {
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
            Value::Str(s) => format!("{:?}", s), // quoted/escaped
        }
    }
}

// Oracle key registries (kart_params.py).
const BOOL_KEYS: &[&str] = &[
    "Reverse",
    "WheelFixed",
    "BoosterBlur",
    "UseRearWheelFx",
    "DriftGaugeReset",
    "UseFrontWheelFx",
    "UseTransformBooster",
];
const INT_KEYS: &[&str] = &[
    "BoosterType",
    "ExtWheelType",
    "LampFlareType",
    "TachometerType",
    "BoosterWaveType",
];
const FLOAT_KEYS: &[&str] = &[
    "TailLampSize",
    "ExtWheelSpeed",
    "TransformTime",
    "WheelPosition",
];

/// Parse a `param.xml` byte buffer into ordered (name, value) pairs.
pub fn parse(bytes: &[u8]) -> Vec<(String, Value)> {
    let text = decode(bytes);
    let attrs = extract_attributes(&text);
    attrs
        .into_iter()
        .map(|(k, v)| {
            let val = coerce(&k, &v);
            (k, val)
        })
        .collect()
}

/// Decode UTF-16 (LE/BE per BOM) or fall back to UTF-8/Latin-1.
fn decode(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    } else if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    } else if bytes.iter().take(64).filter(|&&b| b == 0).count() > 8 {
        // no BOM but lots of NULs → assume UTF-16LE
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Pull `Name='value'` / `Name="value"` attribute pairs out of the BodyParam
/// element (ignoring the `<?xml ...?>` prolog).
fn extract_attributes(text: &str) -> Vec<(String, String)> {
    let body = match text.find("<BodyParam") {
        Some(i) => &text[i + "<BodyParam".len()..],
        None => return Vec::new(),
    };
    let body = &body[..body.find('>').unwrap_or(body.len())];
    let chars: Vec<char> = body.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            // skip whitespace then '='
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            if i < chars.len() && chars[i] == '=' {
                i += 1;
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                if i < chars.len() && (chars[i] == '\'' || chars[i] == '"') {
                    let quote = chars[i];
                    i += 1;
                    let vstart = i;
                    while i < chars.len() && chars[i] != quote {
                        i += 1;
                    }
                    let value: String = chars[vstart..i].iter().collect();
                    i += 1; // closing quote
                    out.push((name, value));
                    continue;
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

fn coerce(key: &str, value: &str) -> Value {
    let v = value.trim();
    if BOOL_KEYS.contains(&key) {
        return Value::Bool(v.eq_ignore_ascii_case("true"));
    }
    if INT_KEYS.contains(&key) {
        if let Ok(i) = v.parse::<i64>() {
            return Value::Int(i);
        }
        return Value::Str(v.to_string());
    }
    if FLOAT_KEYS.contains(&key) {
        if let Ok(f) = v.parse::<f64>() {
            return Value::Float(f);
        }
        return Value::Str(v.to_string());
    }
    // auto: bool → int → float → string
    if v.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if v.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    if let Ok(i) = v.parse::<i64>() {
        return Value::Int(i);
    }
    if let Ok(f) = v.parse::<f64>() {
        if v.contains(['.', 'e', 'E']) || !v.contains(' ') {
            // a lone float like "-0.0768"; multi-token color strings stay strings
            if !v.contains(' ') {
                return Value::Float(f);
            }
        }
    }
    Value::Str(v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a UTF-16LE BodyParam document (BOM + body) from attribute pairs.
    fn doc(attrs: &[(&str, &str)]) -> Vec<u8> {
        let mut s = String::from("<?xml version='1.0' encoding='UTF-16'?>\n<BodyParam\n");
        for (k, v) in attrs {
            s.push_str(&format!("\t{k}='{v}'\n"));
        }
        s.push_str("/>\n");
        let mut out = vec![0xFF, 0xFE]; // LE BOM
        for u in s.encode_utf16() {
            out.extend_from_slice(&u.to_le_bytes());
        }
        out
    }

    #[test]
    fn coerces_types_and_registries() {
        let d = doc(&[
            ("ForwardAccelForce", "147"),       // bare int → Int
            ("DragFactor", "-0.0768"),          // float → Float
            ("BoosterBlur", "true"),            // BOOL_KEY → Bool
            ("TransformTime", "700"),           // FLOAT_KEY → forced Float
            ("BoosterType", "12soon"),          // INT_KEY but non-numeric → Str fallback
            ("Port0", "mod2b1"),                // plain string
            ("TailLampColor", "0 255 100 100"), // multi-token → Str
        ]);
        let m: std::collections::HashMap<_, _> = parse(&d).into_iter().collect();
        assert_eq!(m["ForwardAccelForce"], Value::Int(147));
        assert_eq!(m["DragFactor"], Value::Float(-0.0768));
        assert_eq!(m["BoosterBlur"], Value::Bool(true));
        assert_eq!(m["TransformTime"], Value::Float(700.0));
        assert_eq!(m["BoosterType"], Value::Str("12soon".into()));
        assert_eq!(m["Port0"], Value::Str("mod2b1".into()));
        assert_eq!(m["TailLampColor"], Value::Str("0 255 100 100".into()));
    }

    #[test]
    fn preserves_order_and_names() {
        let d = doc(&[("B", "1"), ("A", "2"), ("C", "3")]);
        let ps = parse(&d);
        assert_eq!(
            ps.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["B", "A", "C"]
        );
    }

    #[test]
    fn empty_or_non_bodyparam_is_empty() {
        assert!(parse(b"not xml at all").is_empty());
        assert!(parse(&doc(&[])).is_empty());
    }
}
