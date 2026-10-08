//! Parse a kart `param.xml` and print the coerced parameters as JSON.

use kart_fmt::param;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_param <param.xml>");
    let data = std::fs::read(&path).expect("read");
    let params = param::parse(&data);
    println!("{{");
    for (i, (k, v)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        println!("  {:?}: {}{}", k, v.to_json(), comma);
    }
    println!("}}");
    eprintln!("{} params", params.len());
}
