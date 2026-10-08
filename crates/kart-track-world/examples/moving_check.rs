fn main() {
    let track = "../analysis/rail/factory_I02.track.1s";
    let index =
        kart_track_world::query::load_track_query_index(std::path::Path::new(track)).unwrap();
    println!(
        "has_moving_road_candidates: {}",
        index.has_moving_road_candidates
    );
    println!(
        "has_dynamic_geom_candidates: {}",
        index.has_dynamic_geom_candidates
    );
}
