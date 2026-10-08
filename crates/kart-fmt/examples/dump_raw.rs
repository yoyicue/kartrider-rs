use kart_fmt::raw_model;
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_raw <model.1s>");
    let d = std::fs::read(&path).unwrap();
    if !raw_model::is_raw_model(&d) {
        eprintln!("not a raw model");
        std::process::exit(1);
    }
    let m = raw_model::parse(&d);
    let mut verts = 0usize;
    let mut uvs = 0usize;
    let mut faces = 0usize;
    let mut meshnodes = 0usize;
    let mut h: u64 = 0xcbf29ce484222325;
    let fnv = |x: u64, h: &mut u64| {
        *h = (*h ^ x).wrapping_mul(0x100000001b3);
    };
    for n in m.all_nodes() {
        if !n.is_mesh {
            continue;
        }
        meshnodes += 1;
        verts += n.vertices.len();
        uvs += n.uvs.len();
        faces += n.vertex_indices.len() / 3;
        for v in &n.vertices {
            for f in v {
                fnv(f.to_bits() as u64, &mut h);
            }
        }
        for u in &n.uvs {
            for f in u {
                fnv(f.to_bits() as u64, &mut h);
            }
        }
        for &i in &n.vertex_indices {
            fnv(i as u32 as u64, &mut h);
        }
        for &i in &n.uv_indices {
            fnv(i as u32 as u64, &mut h);
        }
    }
    println!(
        "asset={} meshnodes={} verts={} uvs={} faces={} hash={:016x}",
        m.asset_type, meshnodes, verts, uvs, faces, h
    );
}
