//! Render a puzzle to a PNG, and report timings and a color histogram.
//!
//! Usage: `cargo run --release --example render '?shell=C$1&cut=C$1/3' out.png [size]`
//!
//! Layers can be isolated with environment variables, which is how the render was checked layer by layer:
//!
//! | variable | default | |
//! | --- | --- | --- |
//! | `BG` | `255,255,255,255` | background as `r,g,b,a` |
//! | `SS` | `2` | supersampling factor |
//! | `EDGES` | on | `0` to omit the edge lines |
//! | `ARROWS` | on | `0` to omit the grip arrows |
//! | `PIECES` | on | `0` to omit the pieces themselves |

use twistypuzzle::builder::build;
use twistypuzzle::math::Quat;
use twistypuzzle::movement::{find_cuts, Cut};
use twistypuzzle::parse::parse_query;
use twistypuzzle::render::camera::Camera;
use twistypuzzle::render::scene::{arrow_instances, render_frame, PuzzleMeshes, SceneOptions};

fn main() {
    /// Frames timed for the steady-state figure.
    const N: u32 = 20;

    let mut args = std::env::args().skip(1);
    let recipe_str = args.next().unwrap_or_else(|| "?shell=C$1&cut=C$1/3".to_string());
    let out = args.next().unwrap_or_else(|| "out.png".to_string());
    let size: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);

    let t0 = std::time::Instant::now();
    let recipe = parse_query(&recipe_str).expect("parse_query");
    let built = build(&recipe).expect("build");
    let mut puzzle = built.puzzle;
    eprintln!("built {} pieces in {:?}", puzzle.pieces.len(), t0.elapsed());

    let t1 = std::time::Instant::now();
    let meshes = PuzzleMeshes::build(&puzzle).expect("meshes");
    let tri_count: usize = meshes.pieces.iter().map(|m| m.triangles.len()).sum();
    let edge_count: usize = meshes.pieces.iter().map(|m| m.edges.len()).sum();
    eprintln!("{tri_count} triangles, {edge_count} edges in {:?}", t1.elapsed());

    // Grips, as `draw_arrows` computes them.
    let mut grips: Vec<Cut> = Vec::new();
    for cut in find_cuts(&mut puzzle, None).expect("find_cuts") {
        let s = cut.plane.constant.sign().expect("sign");
        if s <= 0 {
            grips.push(cut.clone());
        }
        if s >= 0 {
            grips.push(cut.neg());
        }
    }
    twistypuzzle::sort::sort_by(&mut grips, |a, b| b.plane.constant.compare(&a.plane.constant)).expect("sort");
    let arrows = arrow_instances(&grips, puzzle.global_rot).expect("arrows");
    eprintln!("{} grips, {} arrows", grips.len(), arrows.len());

    let quats: Vec<Quat> = puzzle
        .pieces
        .iter()
        .map(|p| puzzle.global_rot.mul(&p.rot.to_f64().expect("rot")))
        .collect();

    let camera = Camera::default();
    let bg = std::env::var("BG").unwrap_or_else(|_| "255,255,255,255".into());
    let bg: Vec<u8> = bg.split(',').map(|v| v.parse().unwrap_or(255)).collect();
    let opts = SceneOptions {
        background: [bg[0], bg[1], bg[2], bg[3]],
        supersample: std::env::var("SS").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
        draw_edges: std::env::var("EDGES").map_or(true, |v| v != "0"),
        draw_arrows: std::env::var("ARROWS").map_or(true, |v| v != "0"),
        draw_pieces: std::env::var("PIECES").map_or(true, |v| v != "0"),
        ..Default::default()
    };

    let t2 = std::time::Instant::now();
    let fb = render_frame(&meshes, &quats, built.scale, &arrows, &camera, size, size, &opts);
    eprintln!("rendered {size}x{size} in {:?}", t2.elapsed());

    // Time a steady-state frame too.
    let t3 = std::time::Instant::now();
    for _ in 0..N {
        let _ = render_frame(&meshes, &quats, built.scale, &arrows, &camera, size, size, &opts);
    }
    let per = t3.elapsed() / N;
    eprintln!("steady state: {per:?}/frame ({:.0} fps)", 1.0 / per.as_secs_f64());

    // Color histogram, to check shading without relying on the eye.
    let mut hist: std::collections::HashMap<[u8; 4], usize> = std::collections::HashMap::new();
    for px in fb.as_bytes().chunks_exact(4) {
        *hist.entry([px[0], px[1], px[2], px[3]]).or_insert(0) += 1;
    }
    let mut v: Vec<_> = hist.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("top colors:");
    for (c, n) in v.iter().take(8) {
        eprintln!("  #{:02x}{:02x}{:02x}{:02x}  {n}", c[0], c[1], c[2], c[3]);
    }

    std::fs::write(&out, fb.to_png()).expect("write png");
    eprintln!("wrote {out}");
}
