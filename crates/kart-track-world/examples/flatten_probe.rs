//! flatten_probe — print flatten() world matrix for a node + the root's
//! local transform. Usage: flatten_probe <track.1s> <node-index>
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: flatten_probe <track.1s> <node-index>");
    let index: usize = args.next().unwrap().parse().unwrap();
    let data = std::fs::read(&path).unwrap();
    let track = kart_fmt::kart_object::parse(&data).unwrap();
    if let Some(root) = track.root {
        let r = &track.nodes[root];
        println!(
            "root({root}) name={:?} transform={:?} pos={:?} scale={:?}",
            r.name, r.transform, r.position, r.scale
        );
        for &c in &r.children {
            let n = &track.nodes[c];
            println!("  child({c}) name={:?} transform={:?}", n.name, n.transform);
        }
    }
    for (i, world) in track.flatten() {
        if i == index {
            println!("node {index} world (column-major 16):");
            for c in 0..4 {
                println!(
                    "  [{:>10.6} {:>10.6} {:>10.6} {:>10.6}]",
                    world[c * 4],
                    world[c * 4 + 1],
                    world[c * 4 + 2],
                    world[c * 4 + 3]
                );
            }
        }
    }
    let n = &track.nodes[index];
    println!(
        "node {index} name={:?} transform={:?} pos={:?} scale={:?}",
        n.name, n.transform, n.position, n.scale
    );
}
