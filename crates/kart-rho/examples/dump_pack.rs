use std::collections::BTreeSet;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if !(2..=4).contains(&args.len()) {
        eprintln!(
            "usage: {} <Data/aaa.pk> [client-root] [mount-filter]",
            args[0]
        );
        return ExitCode::FAILURE;
    }

    let aaa = Path::new(&args[1]);
    let index = match kart_rho::pack::read_pack_index(aaa) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!("root={}", index.root.name);
    println!("mounts={}", index.mounts.len());
    let unique: BTreeSet<_> = index.mounts.iter().map(|m| m.file_name.as_str()).collect();
    println!("unique_rho={}", unique.len());

    if let Some(root) = args.get(2).map(Path::new) {
        let missing = index
            .mounts
            .iter()
            .filter(|m| !root.join("Data").join(&m.file_name).is_file())
            .count();
        println!("missing={missing}");
    }

    let filter = args.get(3).map(|value| value.to_ascii_lowercase());
    let matching: Vec<_> = index
        .mounts
        .iter()
        .filter(|mount| {
            filter.as_ref().is_none_or(|needle| {
                mount.mount_path.to_ascii_lowercase().contains(needle)
                    || mount.file_name.to_ascii_lowercase().contains(needle)
            })
        })
        .collect();
    let display_limit = if filter.is_some() {
        matching.len()
    } else {
        matching.len().min(40)
    };
    for mount in matching.iter().take(display_limit) {
        println!(
            "{}\t{}\tkey={}\tsize={}",
            mount.mount_path,
            mount.file_name,
            fmt_opt(mount.key),
            fmt_opt(mount.media_size)
        );
    }
    if matching.len() > display_limit {
        println!("... {} more", matching.len() - display_limit);
    }
    ExitCode::SUCCESS
}

fn fmt_opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
}
