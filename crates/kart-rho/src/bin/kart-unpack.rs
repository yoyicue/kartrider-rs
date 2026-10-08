//! kart-unpack — extract a .rho archive (clean-room, single binary, all platforms).
//!
//! Usage: kart-unpack <input.rho> <output_dir> [cn|kr|tw|salt]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        eprintln!(
            "usage: {} <input.rho> <output_dir> [cn|kr|tw|salt]",
            args[0]
        );
        return ExitCode::FAILURE;
    }
    let input = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);

    let is_rho5 = input
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("rho5"))
        .unwrap_or(false);
    let result = if is_rho5 {
        let salt = match args.get(3).map(String::as_str).unwrap_or("cn") {
            "cn" => kart_rho::rho5_archive::SALT_CN,
            "kr" => kart_rho::rho5_archive::SALT_KR,
            "tw" => kart_rho::rho5_archive::SALT_TW,
            salt => salt,
        };
        kart_rho::rho5_archive::unpack_rho5(input, salt)
    } else {
        kart_rho::archive::unpack(input)
    };
    let entries = match result {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut written = 0usize;
    for e in &entries {
        // archive entries are trusted but still guard against path escape
        let rel = e.path.replace('\\', "/");
        if rel.split('/').any(|c| c == ".." || c.is_empty()) {
            eprintln!("skip suspicious path: {rel}");
            continue;
        }
        let dest: PathBuf = out_dir.join(&rel);
        if let Some(parent) = dest.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                eprintln!("mkdir {}: {err}", parent.display());
                return ExitCode::FAILURE;
            }
        }
        if let Err(err) = std::fs::write(&dest, &e.data) {
            eprintln!("write {}: {err}", dest.display());
            return ExitCode::FAILURE;
        }
        written += 1;
    }
    println!("unpacked {written} files -> {}", out_dir.display());
    ExitCode::SUCCESS
}
