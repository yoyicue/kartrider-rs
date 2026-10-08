//! DDS texture decoding (BC1/BC2/BC3 + uncompressed) → RGBA8. Ported from the
//! C# DdsPng; shared by the renderer and the glTF exporter (which re-encodes to
//! PNG, the only image format core glTF accepts).

/// Decoded image: row-major RGBA8.
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Decode a `.dds` byte buffer to RGBA8, or None if unsupported/invalid.
pub fn decode(d: &[u8]) -> Option<Rgba> {
    let img = decode_dds(d)?;
    Some(img)
}

fn decode_dds(d: &[u8]) -> Option<Rgba> {
    if d.len() < 128 || &d[0..4] != b"DDS " {
        return None;
    }
    let h = u32::from_le_bytes(d[12..16].try_into().ok()?);
    let w = u32::from_le_bytes(d[16..20].try_into().ok()?);
    if w == 0 || h == 0 || (w as u64) * (h as u64) > 64 * 1024 * 1024 {
        return None;
    }
    let pf_flags = u32::from_le_bytes(d[80..84].try_into().ok()?);
    let rgb_bits = u32::from_le_bytes(d[88..92].try_into().ok()?);
    let masks = [
        u32::from_le_bytes(d[92..96].try_into().ok()?),
        u32::from_le_bytes(d[96..100].try_into().ok()?),
        u32::from_le_bytes(d[100..104].try_into().ok()?),
        u32::from_le_bytes(d[104..108].try_into().ok()?),
    ];
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    const DDPF_FOURCC: u32 = 0x4;
    let body = &d[128..];
    if pf_flags & DDPF_FOURCC != 0 {
        match &d[84..88] {
            b"DXT1" => decode_bc1(body, w, h, &mut rgba),
            b"DXT3" => decode_bc2(body, w, h, &mut rgba),
            b"DXT5" => decode_bc3(body, w, h, &mut rgba),
            _ => return None,
        }
    } else {
        decode_uncompressed(body, w, h, rgb_bits, masks, &mut rgba)?;
    }
    Some(Rgba {
        width: w,
        height: h,
        pixels: rgba,
    })
}

fn rgb565(c: u16) -> (u8, u8, u8) {
    let r = ((c >> 11) & 0x1f) as u32;
    let g = ((c >> 5) & 0x3f) as u32;
    let b = (c & 0x1f) as u32;
    (
        ((r << 3) | (r >> 2)) as u8,
        ((g << 2) | (g >> 4)) as u8,
        ((b << 3) | (b >> 2)) as u8,
    )
}

fn bc1_palette(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    let (r0, g0, b0) = rgb565(c0);
    let (r1, g1, b1) = rgb565(c1);
    let mut p = [[0u8; 4]; 4];
    p[0] = [r0, g0, b0, 255];
    p[1] = [r1, g1, b1, 255];
    if c0 > c1 {
        p[2] = [
            ((2 * r0 as u32 + r1 as u32) / 3) as u8,
            ((2 * g0 as u32 + g1 as u32) / 3) as u8,
            ((2 * b0 as u32 + b1 as u32) / 3) as u8,
            255,
        ];
        p[3] = [
            ((r0 as u32 + 2 * r1 as u32) / 3) as u8,
            ((g0 as u32 + 2 * g1 as u32) / 3) as u8,
            ((b0 as u32 + 2 * b1 as u32) / 3) as u8,
            255,
        ];
    } else {
        p[2] = [
            ((r0 as u32 + r1 as u32) / 2) as u8,
            ((g0 as u32 + g1 as u32) / 2) as u8,
            ((b0 as u32 + b1 as u32) / 2) as u8,
            255,
        ];
        p[3] = [0, 0, 0, 0];
    }
    p
}

fn put(rgba: &mut [u8], w: u32, x: u32, y: u32, c: [u8; 4]) {
    let o = ((y * w + x) * 4) as usize;
    rgba[o..o + 4].copy_from_slice(&c);
}

fn decode_bc1(d: &[u8], w: u32, h: u32, rgba: &mut [u8]) {
    let mut p = 0;
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            let c0 = u16::from_le_bytes([d[p], d[p + 1]]);
            let c1 = u16::from_le_bytes([d[p + 2], d[p + 3]]);
            let idx = u32::from_le_bytes([d[p + 4], d[p + 5], d[p + 6], d[p + 7]]);
            p += 8;
            let pal = bc1_palette(c0, c1);
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx + px, by + py);
                    if x < w && y < h {
                        let ci = ((idx >> (2 * (py * 4 + px))) & 3) as usize;
                        put(rgba, w, x, y, pal[ci]);
                    }
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

fn decode_bc2(d: &[u8], w: u32, h: u32, rgba: &mut [u8]) {
    let mut p = 0;
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            let alpha = u64::from_le_bytes(d[p..p + 8].try_into().unwrap());
            let c0 = u16::from_le_bytes([d[p + 8], d[p + 9]]);
            let c1 = u16::from_le_bytes([d[p + 10], d[p + 11]]);
            let idx = u32::from_le_bytes([d[p + 12], d[p + 13], d[p + 14], d[p + 15]]);
            p += 16;
            let pal = bc1_palette(c0, c1);
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx + px, by + py);
                    if x < w && y < h {
                        let bit = py * 4 + px;
                        let a4 = ((alpha >> (4 * bit)) & 0xF) as u8;
                        let ci = ((idx >> (2 * bit)) & 3) as usize;
                        let mut c = pal[ci];
                        c[3] = (a4 << 4) | a4;
                        put(rgba, w, x, y, c);
                    }
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

fn decode_bc3(d: &[u8], w: u32, h: u32, rgba: &mut [u8]) {
    let mut p = 0;
    let mut by = 0;
    while by < h {
        let mut bx = 0;
        while bx < w {
            let a0 = d[p];
            let a1 = d[p + 1];
            let mut a = [0u8; 8];
            a[0] = a0;
            a[1] = a1;
            if a0 > a1 {
                for i in 1..7 {
                    a[i + 1] = (((7 - i) as u32 * a0 as u32 + i as u32 * a1 as u32) / 7) as u8;
                }
            } else {
                for i in 1..5 {
                    a[i + 1] = (((5 - i) as u32 * a0 as u32 + i as u32 * a1 as u32) / 5) as u8;
                }
                a[6] = 0;
                a[7] = 255;
            }
            let mut aidx = 0u64;
            for i in 0..6 {
                aidx |= (d[p + 2 + i] as u64) << (8 * i);
            }
            let c0 = u16::from_le_bytes([d[p + 8], d[p + 9]]);
            let c1 = u16::from_le_bytes([d[p + 10], d[p + 11]]);
            let idx = u32::from_le_bytes([d[p + 12], d[p + 13], d[p + 14], d[p + 15]]);
            p += 16;
            let pal = bc1_palette(c0, c1);
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx + px, by + py);
                    if x < w && y < h {
                        let bit = py * 4 + px;
                        let ai = ((aidx >> (3 * bit)) & 7) as usize;
                        let ci = ((idx >> (2 * bit)) & 3) as usize;
                        let mut c = pal[ci];
                        c[3] = a[ai];
                        put(rgba, w, x, y, c);
                    }
                }
            }
            bx += 4;
        }
        by += 4;
    }
}

fn decode_uncompressed(
    d: &[u8],
    w: u32,
    h: u32,
    bits: u32,
    masks: [u32; 4],
    rgba: &mut [u8],
) -> Option<()> {
    let bpp = (bits / 8) as usize;
    if bpp != 3 && bpp != 4 {
        return None;
    }
    let shift = |m: u32| if m == 0 { 0 } else { m.trailing_zeros() };
    let chan = |px: u32, m: u32| -> u8 {
        if m == 0 {
            return 0;
        }
        let s = shift(m);
        let v = (px & m) >> s;
        let range = m >> s;
        match (v * 255).checked_div(range) {
            Some(value) => value as u8,
            None => 0,
        }
    };
    for y in 0..h {
        for x in 0..w {
            let sp = ((y * w + x) as usize) * bpp;
            let px = if bpp == 4 {
                u32::from_le_bytes(d[sp..sp + 4].try_into().ok()?)
            } else {
                d[sp] as u32 | (d[sp + 1] as u32) << 8 | (d[sp + 2] as u32) << 16
            };
            put(
                rgba,
                w,
                x,
                y,
                [
                    chan(px, masks[0]),
                    chan(px, masks[1]),
                    chan(px, masks[2]),
                    if masks[3] != 0 {
                        chan(px, masks[3])
                    } else {
                        255
                    },
                ],
            );
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4×4 DXT1 image of one block: 128-byte header + 8-byte block.
    fn dxt1(c0: u16, c1: u16, idx: u32) -> Vec<u8> {
        let mut d = vec![0u8; 128];
        d[0..4].copy_from_slice(b"DDS ");
        d[12..16].copy_from_slice(&4u32.to_le_bytes()); // height
        d[16..20].copy_from_slice(&4u32.to_le_bytes()); // width
        d[80..84].copy_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
        d[84..88].copy_from_slice(b"DXT1");
        d.extend_from_slice(&c0.to_le_bytes());
        d.extend_from_slice(&c1.to_le_bytes());
        d.extend_from_slice(&idx.to_le_bytes());
        d
    }

    #[test]
    fn bc1_solid_white() {
        // c0 = 0xFFFF (565 white); every 2-bit index → palette[0]
        let img = decode(&dxt1(0xFFFF, 0x0000, 0)).expect("decode");
        assert_eq!((img.width, img.height), (4, 4));
        assert_eq!(img.pixels.len(), 4 * 4 * 4);
        assert!(img
            .pixels
            .chunks_exact(4)
            .all(|p| p == [255, 255, 255, 255]));
    }

    #[test]
    fn bc1_two_colors() {
        // c0=white, c1=black; texel (0,0) index 1 → palette[1] = black
        let img = decode(&dxt1(0xFFFF, 0x0000, 0b01)).expect("decode");
        assert_eq!(&img.pixels[0..4], &[0, 0, 0, 255]); // (0,0) black
        assert_eq!(&img.pixels[4..8], &[255, 255, 255, 255]); // (1,0) white
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(b"not a dds file").is_none());
        assert!(decode(&[0u8; 200]).is_none()); // no "DDS " magic
    }
}
