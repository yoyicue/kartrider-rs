//! dds_alpha — print alpha-channel min/max of a .dds (diagnostic for alpha
//! semantics of exported MASK materials). Usage: dds_alpha <file.dds...>

fn main() {
    let mut rc = 0;
    for path in std::env::args().skip(1) {
        match std::fs::read(&path)
            .ok()
            .and_then(|bytes| kart_fmt::dds::decode(&bytes))
        {
            Some(img) => {
                let (mut lo, mut hi) = (u8::MAX, u8::MIN);
                for px in img.pixels.chunks_exact(4) {
                    lo = lo.min(px[3]);
                    hi = hi.max(px[3]);
                }
                println!("{path}\t{}x{}\talpha min={lo} max={hi}", img.width, img.height);
            }
            None => {
                println!("{path}\tdecode failed");
                rc = 1;
            }
        }
    }
    std::process::exit(rc);
}
