use std::path::Path;
fn main() {
    let mut a = std::env::args().skip(1);
    let usage = "usage: render_pose <model.1s> <anim.1s> <time_ms> <out.png> [w h]";
    let model = a.next().expect(usage);
    let anim = a.next().expect(usage);
    let time_ms: i32 = a
        .next()
        .expect(usage)
        .parse()
        .expect("time_ms must be an integer");
    let output = a.next().expect(usage);
    let w: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(640);
    let h: u32 = a.next().map(|s| s.parse().unwrap()).unwrap_or(640);
    match kart_render::render_posed_character_png(
        Path::new(&model),
        Path::new(&anim),
        time_ms,
        w,
        h,
    ) {
        Ok(png) => {
            std::fs::write(&output, png).unwrap();
            println!("rendered pose @ {time_ms}ms -> {output}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
