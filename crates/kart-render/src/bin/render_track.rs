use std::path::Path;
fn main() {
    let mut a = std::env::args().skip(1);
    let usage = "usage: render_track <track.1s> <tex_dir> <out.png> [w h]";
    let track = a.next().expect(usage);
    let tex_dir = a.next().expect(usage);
    let output = a.next().expect(usage);
    let w: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(960);
    let h: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(720);
    match kart_render::render_track_png(Path::new(&track), Path::new(&tex_dir), w, h) {
        Ok(png) => {
            std::fs::write(&output, png).unwrap();
            println!("rendered track -> {output}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
