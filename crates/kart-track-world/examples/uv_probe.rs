fn main() {
    let path = std::env::args().nth(1).expect("usage: uv_probe <track.1s>");
    let data = std::fs::read(&path).unwrap();
    let track = kart_fmt::kart_object::parse(&data).unwrap();
    for node in &track.nodes {
        if node.uv_curves.is_empty() {
            continue;
        }
        print!("{}:", node.name);
        for (nm, curve) in &node.uv_curves {
            if curve.kf_type == 0 {
                print!(" {nm}(ty=0, keys={:?})", curve.keys);
            } else {
                print!(" {nm}(ty={}, {} keys)", curve.kf_type, curve.keys.len());
            }
        }
        println!();
    }
}
