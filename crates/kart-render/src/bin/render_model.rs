use std::path::Path;
fn main() {
    let mut a = std::env::args().skip(1);
    let input = a
        .next()
        .expect("usage: render_model <model.1s> <out.png> [w h]");
    let output = a
        .next()
        .expect("usage: render_model <model.1s> <out.png> [w h]");
    let w: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(640);
    let h: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(640);
    match kart_render::render_model_png(Path::new(&input), w, h) {
        Ok(png) => {
            std::fs::write(&output, png).unwrap();
            println!("rendered -> {output}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
