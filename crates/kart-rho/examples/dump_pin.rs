use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <KartRider.pin>", args[0]);
        return ExitCode::FAILURE;
    }

    let pin = match kart_rho::pin::read_pin_file(Path::new(&args[1])) {
        Ok(pin) => pin,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!("locale_id={}", pin.header.locale_id);
    println!("client_loc={}", pin.header.unk2);
    println!("minor_version={}", pin.header.minor_version);
    println!("login_type={}", pin.header.login_type);
    println!("url={}", empty_dash(&pin.header.url));
    println!("patch_url={}", empty_dash(&pin.header.patch_url));
    println!("aes_key_len={}", pin.header.aes_key.len());
    println!("auth_methods={}", pin.auth_methods.len());
    for method in &pin.auth_methods {
        let servers = method
            .login_servers
            .iter()
            .map(|s| format!("{}:{}", s.ip, s.port))
            .collect::<Vec<_>>()
            .join(",");
        println!("  [{}] {} -> {}", method.index, method.name, servers);
    }
    println!("top_bml_objects={}", pin.bml_objects.len());
    for object in &pin.bml_objects {
        println!(
            "  {} values={} children={}",
            object.name,
            object.values.len(),
            object.children.len()
        );
    }
    ExitCode::SUCCESS
}

fn empty_dash(s: &str) -> &str {
    if s.is_empty() {
        "-"
    } else {
        s
    }
}
