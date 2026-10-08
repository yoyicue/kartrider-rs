//! rho_find — locate archive entries by name substring across .rho packs.
//!
//! Usage: rho_find <pattern...> -- <pack.rho...>
//! Prints one `<pack>\t<entry>` line per entry whose lowercase name contains
//! any pattern. Diagnostic companion to kart-unpack (read-only; extracts
//! nothing).

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(split) = args.iter().position(|a| a == "--") else {
        eprintln!("usage: rho_find <pattern...> -- <pack.rho...>");
        return ExitCode::FAILURE;
    };
    let patterns: Vec<String> = args[..split].iter().map(|p| p.to_lowercase()).collect();
    let packs = &args[split + 1..];
    if patterns.is_empty() || packs.is_empty() {
        eprintln!("usage: rho_find <pattern...> -- <pack.rho...>");
        return ExitCode::FAILURE;
    }
    let mut rc = ExitCode::SUCCESS;
    for pack in packs {
        let path = Path::new(pack);
        match kart_rho::archive::unpack(path) {
            Ok(entries) => {
                for entry in entries {
                    let lower = entry.path.to_lowercase();
                    if patterns.iter().any(|p| lower.contains(p)) {
                        println!("{pack}\t{}", entry.path);
                    }
                }
            }
            Err(err) => {
                eprintln!("error: {pack}: {err}");
                rc = ExitCode::FAILURE;
            }
        }
    }
    rc
}
