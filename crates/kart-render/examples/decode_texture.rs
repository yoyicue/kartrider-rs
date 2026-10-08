fn main() {
    let mut a = std::env::args().skip(1);
    let input = a.next().expect("usage: decode_texture <in> <out.png>");
    let output = a.next().expect("usage: decode_texture <in> <out.png>");
    let img = kart_render::texture::load(std::path::Path::new(&input)).expect("load failed");
    let mut buf = Vec::new();
    {
        let mut e = png::Encoder::new(&mut buf, img.width, img.height);
        e.set_color(png::ColorType::Rgba);
        e.set_depth(png::BitDepth::Eight);
        let mut w = e.write_header().unwrap();
        w.write_image_data(&img.rgba).unwrap();
    }
    std::fs::write(&output, buf).unwrap();
    println!("{}x{} -> {}", img.width, img.height, output);
}
