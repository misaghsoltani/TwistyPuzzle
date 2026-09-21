//! Benchmark: what a batch buys over stepping and drawing puzzles one at a time.
//!
//! Two claims are worth measuring, because both are the reason the batch exists at all:
//!
//! * a move on a puzzle whose turns are fixed permutations is a gather, not a
//!   re-derivation of the cut structure.
//! * a frame of a state is a table lookup, not a rasterization.
//!
//! Run with `cargo run --release --example bench_batch`. `ROWS`, `STEPS` and
//! `SIZE` override the defaults.

use std::time::Instant;

use twistypuzzle::batch::PuzzleBatch;
use twistypuzzle::simulator::Simulator;

fn env(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn rate(n: usize, secs: f64) -> String {
    let per = n as f64 / secs;
    if per >= 1.0e6 {
        format!("{:.1}M/s", per / 1.0e6)
    } else if per >= 1.0e3 {
        format!("{:.1}k/s", per / 1.0e3)
    } else {
        format!("{per:.0}/s")
    }
}

fn main() {
    let recipe = std::env::var("RECIPE").unwrap_or_else(|_| "?shell=C$1&cut=C$1/3".into());
    let rows = env("ROWS", 256);
    let steps = env("STEPS", 100);
    let size = env("SIZE", 64) as u32;
    println!("{recipe}, {rows} puzzles, {steps} steps, {size}x{size} frames");

    // --- building ---------------------------------------------------------
    let t = Instant::now();
    let mut batch = PuzzleBatch::build(&recipe, rows).expect("batch");
    let build_secs = t.elapsed().as_secs_f64();
    println!(
        "  build batch of {rows:<5}  {build_secs:8.3} s   tabular: {}",
        batch.is_tabular()
    );

    let t = Instant::now();
    let mut sims: Vec<Simulator> = (0..rows.min(32))
        .map(|_| Simulator::from_query(&recipe).expect("build"))
        .collect();
    let one_by_one = t.elapsed().as_secs_f64() / rows.min(32) as f64;
    println!(
        "  one puzzle at a time     {:8.3} s   ({:.0}x for the batch)",
        one_by_one * rows as f64,
        one_by_one * rows as f64 / build_secs.max(1e-9)
    );

    // --- stepping ---------------------------------------------------------
    let count = batch.action_count() as u32;
    let actions: Vec<u32> = (0..rows).map(|i| (i as u32 * 7 + 3) % count).collect();
    batch.step(&actions).expect("warm up");

    let t = Instant::now();
    for _ in 0..steps {
        batch.step(&actions).expect("step");
    }
    let secs = t.elapsed().as_secs_f64();
    let moves = rows * steps;
    println!(
        "  batch step               {secs:8.3} s   {:>9} moves   {:.2} us/move",
        rate(moves, secs),
        secs / moves as f64 * 1.0e6
    );

    let t = Instant::now();
    let probe = steps.min(4);
    for _ in 0..probe {
        for (s, &a) in sims.iter_mut().zip(&actions) {
            s.apply_action(a as usize).expect("apply");
        }
    }
    let secs_geom = t.elapsed().as_secs_f64() / (probe * sims.len()) as f64;
    println!(
        "  turning the geometry     {:8.3} s   {:>9} moves   {:.2} us/move   ({:.0}x for the batch)",
        secs_geom * moves as f64,
        rate(1, secs_geom),
        secs_geom * 1.0e6,
        secs_geom * moves as f64 / secs,
    );

    // --- scrambling -------------------------------------------------------
    let t = Instant::now();
    batch.reset(30).expect("reset");
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  reset + scramble 30      {secs:8.3} s   {:>9} moves",
        rate(rows * 30, secs)
    );

    // --- drawing ----------------------------------------------------------
    batch.render(size, size).expect("warm up the table");
    let t = Instant::now();
    let frames = 4;
    for _ in 0..frames {
        std::hint::black_box(batch.render(size, size).expect("render"));
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  batch render             {secs:8.3} s   {:>9} frames  {:.1} us/frame",
        rate(rows * frames, secs),
        secs / (rows * frames) as f64 * 1.0e6
    );

    sims[0].options_mut().draw_arrows = false;
    let t = Instant::now();
    let probe = 8i32;
    for _ in 0..probe {
        std::hint::black_box(sims[0].render(size, size));
    }
    let secs_one = t.elapsed().as_secs_f64() / f64::from(probe);
    println!(
        "  rasterizing each frame   {:8.3} s   {:>9} frames  {:.1} us/frame   ({:.0}x for the table)",
        secs_one * (rows * frames) as f64,
        rate(1, secs_one),
        secs_one * 1.0e6,
        secs_one * (rows * frames) as f64 / secs,
    );
}
