use kart_rho::rho5_archive::{SALT_CN, SALT_KR, SALT_TW};
use kart_rho::rho5_keyprovider::Rho5KeyProvider;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "usage: {} <input.rho5> [cn|kr|tw|salt] [scan-limit]",
            args[0]
        );
        return ExitCode::FAILURE;
    }
    let path = Path::new(&args[1]);
    let data = match std::fs::read(path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("read {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let name = match path.file_name().and_then(|name| name.to_str()) {
        Some(name) => name,
        None => {
            eprintln!("invalid input file name");
            return ExitCode::FAILURE;
        }
    };
    let salt = match args.get(2).map(String::as_str).unwrap_or("cn") {
        "cn" => SALT_CN,
        "kr" => SALT_KR,
        "tw" => SALT_TW,
        salt => salt,
    };
    let scan_limit = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(data.len())
        .min(data.len());

    let mut provider = Rho5KeyProvider::new();
    provider.init_header_key(name, salt);
    let sub = [
        provider.next_sub_num(),
        provider.next_sub_num(),
        provider.next_sub_num(),
    ];
    let mut matches = 0usize;
    for offset in 0..scan_limit.saturating_sub(12) {
        let words = [
            decrypt_word(&data, offset, sub[0]),
            decrypt_word(&data, offset + 4, sub[1]),
            decrypt_word(&data, offset + 8, sub[2]),
        ];
        let crc = words[0] as i32;
        let second = words[1].to_le_bytes();
        let third = words[2].to_le_bytes();
        let version = second[0];
        let file_count = i32::from_le_bytes([second[1], second[2], second[3], third[0]]);
        if crc == version as i32 + file_count
            && version <= 32
            && (1..=10_000_000).contains(&file_count)
        {
            println!("offset={offset}\tcrc={crc}\tversion={version}\tfile_count={file_count}");
            matches += 1;
        }
    }
    println!("matches={matches}\tscanned={scan_limit}");
    ExitCode::SUCCESS
}

fn decrypt_word(data: &[u8], offset: usize, sub: u32) -> u32 {
    let raw = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    raw.wrapping_sub(sub)
}
