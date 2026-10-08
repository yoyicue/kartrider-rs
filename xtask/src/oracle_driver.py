"""Oracle (karttools) stat emitter for the xtask cross-validation harness.

Invoked as:  python oracle_driver.py <kind> <path>
where <kind> is one of: model | param | track | char_anim | kart_anim.
Emits a single line of `key=value` pairs to stdout describing structural counts
that the clean-room Rust parsers also produce, so the harness can diff them
without re-deriving the closed-source format. Any failure prints `ERROR <msg>`.

karttools is imported from the gitignored oracle-local/ tree; this file carries
no oracle code, only calls its public parse_* entry points.
"""
import contextlib
import io
import os
import sys


@contextlib.contextmanager
def quiet():
    """Swallow the oracle parsers' own progress prints to stdout/stderr so only
    our stat line reaches the harness."""
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


def dfs_nodes(nodes):
    """Yield every node in a children-linked tree (pre-order)."""
    for n in nodes:
        yield n
        yield from dfs_nodes(getattr(n, "children", []) or [])


def model_stats(path):
    from karttools.parsers.model import parse_model
    sc = parse_model(path, scale=1.0)
    mesh = verts = uvs = faces = 0
    for n in dfs_nodes(getattr(sc, "nodes", []) or []):
        if getattr(n, "is_mesh", False):
            mesh += 1
            verts += len(getattr(n, "vertices", []) or [])
            uvs += len(getattr(n, "uvs", []) or [])
            # oracle stores vertex_indices as a list of [a,b,c] triangles, so its
            # length is already the triangle count.
            faces += len(getattr(n, "vertex_indices", []) or [])
    return dict(asset=sc.asset_type, meshnodes=mesh, verts=verts, uvs=uvs, faces=faces)


def norm_value(v):
    """Type-stable string form shared with the Rust side (param::Value::norm),
    so floats checksum identically regardless of each language's default repr."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def param_stats(path):
    from karttools.parsers.kart_params import parse_kart_params
    p = parse_kart_params(path) or {}
    # checksum the (key,value) set order-independently so the comparison is
    # robust to dict ordering on either side.
    items = sorted((str(k), norm_value(v)) for k, v in p.items())
    h = 0xCBF29CE484222325  # FNV-1a 64-bit offset basis (matches the Rust side)
    for k, v in items:
        for b in f"{k}={v}".encode():
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return dict(count=len(p), checksum=f"{h:016x}")


def track_stats(path):
    # parse_track takes the track *directory* and returns
    # (root_RelementNode, meta_dict, source_list); the root tree bundles both
    # track.1s and skydome.1s, so the Rust side parses both to match.
    from karttools.parsers.track import parse_track
    root, meta, _ = parse_track(path, scale=1.0)
    total = mesh = 0
    for n in dfs_nodes([root]):
        total += 1
        if getattr(n, "is_mesh", False):
            mesh += 1
    return dict(nodes=total, meshes=mesh, events=meta.get("event_count", -1))


def char_anim_stats(path):
    from karttools.parsers.animation import parse_animation
    c = parse_animation(path)
    if c is None:
        raise RuntimeError("parse_animation returned None")
    kf = sum(len(t.pos_keys) + len(t.rot_keys) + len(t.scale_keys) for t in c.tracks)
    return dict(bones=len(c.tracks), kf=kf, dur=c.duration_ms)


def kart_anim_stats(path):
    from karttools.parsers.kart_animation import parse_kart_animation
    k = parse_kart_animation(path)
    if k is None:
        raise RuntimeError("parse_kart_animation returned None")
    kf = sum(len(ch.keyframes) for ch in k.channels)
    animated = sum(1 for ch in k.channels if len(ch.keyframes) > 1)
    return dict(channels=len(k.channels), animated=animated, frames=k.num_frames, kf=kf, dur=k.duration_ms)


DISPATCH = {
    "model": model_stats,
    "param": param_stats,
    "track": track_stats,
    "char_anim": char_anim_stats,
    "kart_anim": kart_anim_stats,
}


def main():
    kind, path = sys.argv[1], sys.argv[2]
    try:
        with quiet():
            stats = DISPATCH[kind](path)
    except Exception as e:  # noqa: BLE001 - surface any oracle failure to the harness
        print(f"ERROR {type(e).__name__}: {e}")
        return
    print(" ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
