//! Texture loading: PNG (character/kart) and DDS (BC1/2/3 + uncompressed, for
//! track/stage geometry). DDS decode ported from the C# DdsPng. Returns RGBA8.

use std::path::Path;

#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Load a texture from a file, dispatching on extension; None if missing/unsupported.
pub fn load(path: &Path) -> Option<Image> {
    let bytes = std::fs::read(path).ok()?;
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => load_png(&bytes),
        Some("dds") => decode_dds(&bytes),
        _ => None,
    }
}

fn load_png(bytes: &[u8]) -> Option<Image> {
    let dec = png::Decoder::new(bytes);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..(w * h * 4) as usize].to_vec(),
        png::ColorType::Rgb => {
            let mut o = Vec::with_capacity((w * h * 4) as usize);
            for c in buf[..(w * h * 3) as usize].chunks_exact(3) {
                o.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            o
        }
        png::ColorType::Grayscale => {
            let mut o = Vec::with_capacity((w * h * 4) as usize);
            for &g in &buf[..(w * h) as usize] {
                o.extend_from_slice(&[g, g, g, 255]);
            }
            o
        }
        png::ColorType::GrayscaleAlpha => {
            let mut o = Vec::with_capacity((w * h * 4) as usize);
            for c in buf[..(w * h * 2) as usize].chunks_exact(2) {
                o.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
            o
        }
        _ => return None,
    };
    Some(Image {
        width: w,
        height: h,
        rgba,
    })
}

// ---- DDS (ported from C# DdsPng) ------------------------------------------

fn decode_dds(d: &[u8]) -> Option<Image> {
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
    Some(Image {
        width: w,
        height: h,
        rgba,
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
        if range == 0 {
            0
        } else {
            (v * 255 / range) as u8
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
