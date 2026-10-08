//! FNV-1a checksum of a parsed KartObject track: node count + per-node name,
//! position, scale, transform, and geometry. Matches the C# reference dumper for
//! byte-for-byte parser validation.

use kart_fmt::kart_object;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}
fn ff(h: &mut u64, v: f32) {
    fnv(h, &v.to_le_bytes());
}

fn main() {
    // Quiet the global panic hook so kart-fmt's internally-recovered
    // trailing-MorphTontroller probe doesn't print on a successful hash of a
    // morph-bearing track (the library no longer owns the hook). A genuinely
    // unparseable file still desyncs to a non-zero exit.
    std::panic::set_hook(Box::new(|_| {}));
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_track <track.1s>");
    let data = std::fs::read(&path).expect("read");
    let track = match kart_object::parse(&data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    };
    // Pre-order DFS from the root so the hash order matches the C# WalkNode walk
    // (a shared/back-referenced node is visited once per parent, like C#).
    let mut order = Vec::new();
    fn dfs(track: &kart_object::Track, i: usize, out: &mut Vec<usize>) {
        out.push(i);
        for &c in &track.nodes[i].children {
            dfs(track, c, out);
        }
    }
    if let Some(r) = track.root {
        dfs(&track, r, &mut order);
    }

    let mut h = 0xcbf29ce484222325u64;
    let (mut hn, mut hx, mut hg, mut ht) = (h, h, h, h);
    let mut meshes = 0usize;
    let mut tris = 0usize;
    let mut tex_nodes = 0usize;
    for &ni in &order {
        let n = &track.nodes[ni];
        fnv(&mut h, n.name.as_bytes());
        fnv(&mut hn, n.name.as_bytes());
        for v in n
            .position
            .iter()
            .chain(n.scale.iter())
            .chain(n.transform.iter())
        {
            ff(&mut h, *v);
            ff(&mut hx, *v);
        }
        // (geom already carries expanded ReToonRigid faces too)
        if let Some(g) = &n.geom {
            meshes += 1;
            tris += if g.strip {
                g.indices.len().saturating_sub(2)
            } else {
                g.indices.len() / 3
            };
            for v in g.vertices.iter().chain(g.normals.iter()) {
                v.iter().for_each(|x| {
                    ff(&mut h, *x);
                    ff(&mut hg, *x);
                });
            }
            for uv in &g.uvs {
                uv.iter().for_each(|x| {
                    ff(&mut h, *x);
                    ff(&mut hg, *x);
                });
            }
            for i in &g.indices {
                fnv(&mut h, &i.to_le_bytes());
                fnv(&mut hg, &i.to_le_bytes());
            }
        }
        if let Some(t) = &n.tex {
            tex_nodes += 1;
            fnv(&mut h, t.as_bytes());
            fnv(&mut ht, t.as_bytes());
        }
    }
    println!(
        "nodes={} meshes={} tris={} tex={} hash={:016x} hn={:016x} hx={:016x} hg={:016x} ht={:016x}",
        order.len(), meshes, tris, tex_nodes, h, hn, hx, hg, ht
    );
}
