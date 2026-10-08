//! Decode a `.bml` (BinaryXml) file to indented XML and report round-trip
//! byte-identity (parse → re-serialize == original) — the self-validation that
//! the wire format matches the C# BinaryXmlDocument without needing its GUI.
//!
//! Usage: dump_bml <file.bml>

use kart_fmt::bml::{self, Tag};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn print_tag(t: &Tag, depth: usize) {
    let pad = "  ".repeat(depth);
    let attrs: String = t
        .attributes
        .iter()
        .map(|(k, v)| format!(" {}=\"{}\"", k, esc(v)))
        .collect();
    let has_text = !t.text.is_empty();
    if t.children.is_empty() && !has_text {
        println!("{pad}<{}{attrs}/>", t.name);
    } else if t.children.is_empty() {
        println!("{pad}<{}{attrs}>{}</{}>", t.name, esc(&t.text), t.name);
    } else {
        println!("{pad}<{}{attrs}>{}", t.name, esc(&t.text));
        for c in &t.children {
            print_tag(c, depth + 1);
        }
        println!("{pad}</{}>", t.name);
    }
}

fn count(t: &Tag) -> usize {
    1 + t.children.iter().map(count).sum::<usize>()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_bml <file.bml>");
    let data = std::fs::read(&path).expect("read");
    let tag = match bml::parse(&data) {
        Some(t) => t,
        None => {
            eprintln!("not a well-formed .bml (truncated / trailing / ill-framed)");
            std::process::exit(1);
        }
    };
    print_tag(&tag, 0);
    let roundtrip = tag.to_bytes() == data;
    eprintln!(
        "tags={} roundtrip={}",
        count(&tag),
        if roundtrip {
            "byte-identical"
        } else {
            "DIFFERS"
        }
    );
    if !roundtrip {
        std::process::exit(2);
    }
}
