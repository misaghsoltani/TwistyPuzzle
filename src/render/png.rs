//! A minimal PNG encoder, so a rendered frame can be written to a file
//! without pulling in an image library.

use std::io::Write;

use flate2::write::ZlibEncoder;
use flate2::Compression;

fn crc32(bytes: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, e) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(&kind);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode tightly packed RGBA8 pixels as a PNG.
pub fn encode_rgba(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 1024);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, truecolor + alpha
    chunk(&mut out, *b"IHDR", &ihdr);

    // Filter type 1 (Sub) predicts each pixel from its left neighbor, which
    // compresses flat-shaded polygons well.
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for y in 0..height as usize {
        raw.push(1);
        let row = &data[y * stride..(y + 1) * stride];
        for x in 0..stride {
            let left = if x >= 4 { row[x - 4] } else { 0 };
            raw.push(row[x].wrapping_sub(left));
        }
    }

    let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(&raw).expect("in-memory write cannot fail");
    let compressed = enc.finish().expect("in-memory write cannot fail");
    chunk(&mut out, *b"IDAT", &compressed);
    chunk(&mut out, *b"IEND", &[]);
    out
}
