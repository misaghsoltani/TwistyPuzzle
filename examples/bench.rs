//! Benchmark: build every cataloged puzzle and test it the way
//! every cataloged puzzle, so a change in the exact core shows up as a
//! change in the number.

use twistypuzzle::builder::build;
use twistypuzzle::math::ExactQuaternion;
use twistypuzzle::movement::{find_cuts, find_stops, make_move, Cut, Puzzle};
use twistypuzzle::parse::parse_query;
use twistypuzzle::render::scene::PuzzleMeshes;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        f64::from(self.0) / 4294967296.0
    }
}

fn grips_of(puzzle: &mut Puzzle) -> Vec<Cut> {
    let mut grips: Vec<Cut> = Vec::new();
    for cut in find_cuts(puzzle, None).expect("find_cuts") {
        let s = cut.plane.constant.sign().expect("sign");
        if s <= 0 {
            grips.push(cut.clone());
        }
        if s >= 0 {
            grips.push(cut.neg());
        }
    }
    twistypuzzle::sort::sort_by(&mut grips, |a, b| {
        b.plane.constant.compare(&a.plane.constant)
    })
    .expect("sort");
    grips
}

fn main() {
    let nmoves: usize = std::env::var("NMOVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let render = std::env::var("RENDER").is_ok();
    let mut total_pieces = 0usize;
    let mut slowest: Vec<(u128, usize, String)> = Vec::new();
    let t_all = std::time::Instant::now();

    for &(name, _, _, recipe) in twistypuzzle::catalog::CATALOG {
        let t0 = std::time::Instant::now();
        let parsed = parse_query(recipe).expect("parse_query");
        let built = build(&parsed).expect("build");
        let mut puzzle = built.puzzle;
        total_pieces += puzzle.pieces.len();

        let cuts = find_cuts(&mut puzzle, None).expect("find_cuts");
        for c in &cuts {
            let _ = find_stops(&mut puzzle, c).expect("find_stops");
        }

        let mut rng = Lcg(0x5eed_1234);
        let mut grips = grips_of(&mut puzzle);
        for _ in 0..nmoves {
            if grips.is_empty() {
                break;
            }
            let ci = (rng.next() * grips.len() as f64).floor() as usize;
            let dir = (rng.next() * 2.0).floor() as i32 * 2 - 1;
            let cut = grips[ci].clone();
            let rots = find_stops(&mut puzzle, &cut).expect("find_stops");
            let rot = if rots.is_empty() {
                ExactQuaternion::identity()
            } else {
                let mut zi = if dir < 0 { rots.len() - 1 } else { 0 };
                if dir < 0 {
                    while rots[zi].pseudo_angle().expect("angle").is_zero() && zi > 0 {
                        zi -= 1;
                    }
                } else {
                    while rots[zi].pseudo_angle().expect("angle").is_zero() && zi < rots.len() - 1 {
                        zi += 1;
                    }
                }
                rots[zi].clone()
            };
            make_move(&mut puzzle, &cut, &rot).expect("make_move");
            grips = grips_of(&mut puzzle);
        }

        if render {
            let meshes = PuzzleMeshes::build(&puzzle).expect("meshes");
            std::hint::black_box(&meshes);
        }
        let ms = t0.elapsed().as_millis();
        slowest.push((ms, puzzle.pieces.len(), name.to_string()));
    }

    let total = t_all.elapsed();
    slowest.sort_by_key(|x| std::cmp::Reverse(x.0));
    println!(
        "{} puzzles, {total_pieces} pieces, {nmoves} moves each: {:.2} s",
        twistypuzzle::catalog::len(),
        total.as_secs_f64()
    );
    println!("slowest:");
    for (ms, n, name) in slowest.iter().take(6) {
        println!("  {ms:>6} ms  {n:>4}p  {name}");
    }
}
