//! Benchmark: one interactive frame, the way the desktop interface draws it.
//!
//! The batch draws thousands of small frames from a lookup table (`bench_batch`).
//! This is the other end: one puzzle, one big frame, redrawn every time the pointer
//! moves or a turn animates, where what costs is the rasterizer and the buffers it needs.
//! A frame at 1280x1280 with 2x supersampling rasterizes 2560x2560 pixels and needs 26 MB
//! of color and 26 MB of depth to do it, so whether those buffers are allocated per frame
//! or kept is most of the difference.
//!
//! Run with `cargo run --release --example bench_frame`. `RECIPE`, `SIZE`,
//! `SUPERSAMPLE` and `FRAMES` override the defaults.

use std::time::Instant;

use twistypuzzle::render::framebuffer::Framebuffer;
use twistypuzzle::render::scene::FrameScratch;
use twistypuzzle::simulator::Simulator;

fn env(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn main() {
    let recipe = std::env::var("RECIPE").unwrap_or_else(|_| "?shell=C$1&cut=C$1/3".into());
    let size = env("SIZE", 1280) as u32;
    let ss = env("SUPERSAMPLE", 2) as u32;
    let frames = env("FRAMES", 60);
    println!("{recipe}, {size}x{size}, {ss}x supersample, {frames} frames");

    let mut sim = Simulator::from_query(&recipe).expect("build");
    sim.options_mut().supersample = ss;
    sim.scramble(20);
    sim.settle().expect("settle");

    // Fresh buffers every frame, which is what a caller with no scratch pays.
    let owned = Framebuffer::new(size, size);
    let mut out = owned.into_bytes();
    let t = Instant::now();
    for _ in 0..frames {
        let fb = sim.render(size, size);
        out.copy_from_slice(fb.as_bytes());
        std::hint::black_box(&out);
    }
    let fresh = t.elapsed().as_secs_f64() / frames as f64;

    // The same frame, into a buffer the caller owns, off a kept scratch.
    let mut scratch = FrameScratch::new();
    sim.render_into(size, size, &mut scratch, &mut out).expect("render");
    let t = Instant::now();
    for _ in 0..frames {
        sim.render_into(size, size, &mut scratch, &mut out).expect("render");
        std::hint::black_box(&out);
    }
    let kept = t.elapsed().as_secs_f64() / frames as f64;

    let mb = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    println!(
        "  fresh buffers  {:8.2} ms/frame  ({:5.1} fps)  {:6.1} MB allocated per frame",
        fresh * 1e3,
        1.0 / fresh,
        mb(scratch.capacity() + out.len())
    );
    println!(
        "  kept scratch   {:8.2} ms/frame  ({:5.1} fps)  {:6.1} MB held, none per frame",
        kept * 1e3,
        1.0 / kept,
        mb(scratch.capacity())
    );
    println!("  speedup        {:8.2}x", fresh / kept);
}
