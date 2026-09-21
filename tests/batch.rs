//! Many puzzles stepped together, checked against one puzzle stepped alone.
//!
//! A batch takes two quite different routes to the same answers. For a puzzle
//! whose moves are fixed permutations it keeps arrays and gathers. For one
//! whose layers can lock or whose stickers leave the lattice it turns each
//! copy's geometry. Both must agree with what a single [`Simulator`] does, and
//! the drawing table must agree with the rasterizer to the byte, or the fast
//! path is not the same path.

use twistypuzzle::batch::PuzzleBatch;
use twistypuzzle::simulator::Simulator;

/// A small, seedable PRNG, so a failing case can be reproduced from its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const SAMPLE: &[&str] = &[
    "?shell=C$1&cut=C$1/3",       // 3x3x3
    "?shell=C$1&cut=C$1/2",       // 2x2x2
    "?shell=D$1&cut=D$sqrt(5)/5", // Megaminx-ish
    "?shell=T$1&cut=T$1/3",       // tetrahedral
    "?shell=O$1&cut=O$1/3",       // octahedral
];

/// A puzzle that jumbles (the Little Chop), so a batch of it has to keep
/// geometry: its turns leave stickers where no sticker sits when it is
/// solved, so there are no slots to number and no permutations to gather.
const JUMBLING: &str = "?shell=C$3*sqrt(2)/4&cut=jC$0";

/// The 3x3x3, for the tests that want one puzzle instead of a sample.
const CUBE: &str = "?shell=C$1&cut=C$1/3";

fn sim(recipe: &str) -> Simulator {
    Simulator::from_query(recipe).unwrap_or_else(|e| panic!("building {recipe}: {e}"))
}

#[test]
fn a_batch_of_one_agrees_with_a_single_puzzle_move_for_move() {
    for recipe in SAMPLE {
        let mut b = PuzzleBatch::build(recipe, 1).expect("batch");
        let mut s = sim(recipe);
        let count = b.action_count();
        let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
        for step in 0..24 {
            let a = rng.below(count) as u32;
            let outcome = b.step(&[a]).expect("step");
            let applied = s.apply_action(a as usize).expect("apply");
            assert_eq!(
                outcome[0].applied, applied,
                "{recipe} step {step}: the batch and the puzzle disagree on whether {a} applied"
            );
            assert_eq!(
                b.observations::<u8>().expect("observations"),
                s.stickers()
                    .expect("stickers")
                    .into_iter()
                    .map(|c| c as u8)
                    .collect::<Vec<u8>>(),
                "{recipe} step {step}: the states differ after action {a}"
            );
            assert_eq!(
                outcome[0].solved,
                s.is_solved().expect("is_solved"),
                "{recipe} step {step}: solvedness differs"
            );
        }
    }
}

#[test]
fn every_row_of_a_batch_steps_independently() {
    let recipe = "?shell=C$1&cut=C$1/3";
    let rows = 8;
    let mut b = PuzzleBatch::build(recipe, rows).expect("batch");
    let count = b.action_count();
    let mut singles: Vec<Simulator> = (0..rows).map(|_| sim(recipe)).collect();

    let mut rng = Rng(7);
    for _ in 0..12 {
        let actions: Vec<u32> = (0..rows).map(|_| rng.below(count) as u32).collect();
        b.step(&actions).expect("step");
        for (s, &a) in singles.iter_mut().zip(&actions) {
            s.apply_action(a as usize).expect("apply");
        }
    }
    let k = b.sticker_count();
    let seen = b.observations::<u8>().expect("observations");
    for (i, s) in singles.iter_mut().enumerate() {
        let want: Vec<u8> = s.stickers().expect("stickers").into_iter().map(|c| c as u8).collect();
        assert_eq!(&seen[i * k..(i + 1) * k], &want[..], "row {i} differs");
    }
}

#[test]
fn sticker_ids_and_one_hot_describe_the_same_state() {
    let mut b = PuzzleBatch::build("?shell=C$1&cut=C$1/3", 4).expect("batch");
    b.seed_all(11);
    b.reset(8).expect("reset");
    let k = b.sticker_count();
    let c = b.color_count();
    let colors = b.observations::<u8>().expect("observations");
    let ids = b.sticker_ids::<u32>().expect("ids");
    let hot = b.one_hot().expect("one hot");
    let solved = b.solved_colors().to_vec();
    for row in 0..b.len() {
        for i in 0..k {
            let color = colors[row * k + i];
            assert_eq!(
                u16::from(color),
                solved[ids[row * k + i] as usize],
                "row {row} slot {i}: the color does not follow from the identity"
            );
            for v in 0..c {
                assert_eq!(
                    hot[row * k * c + i * c + v],
                    u8::from(v == color as usize),
                    "row {row} slot {i} color {v}: the indicator is wrong"
                );
            }
        }
    }
}

#[test]
fn a_reset_scrambles_only_the_rows_it_is_given() {
    let mut b = PuzzleBatch::build("?shell=C$1&cut=C$1/3", 6).expect("batch");
    b.seed_all(3);
    b.reset(10).expect("reset");
    let before = b.observations::<u8>().expect("observations");
    let k = b.sticker_count();

    let which: Vec<bool> = (0..6).map(|i| i % 2 == 0).collect();
    b.reset_where(&which, 10).expect("reset where");
    let after = b.observations::<u8>().expect("observations");
    for row in 0..6 {
        let same = before[row * k..(row + 1) * k] == after[row * k..(row + 1) * k];
        assert_eq!(
            same,
            !which[row],
            "row {row} was {} when it should not have been",
            if same { "left alone" } else { "changed" }
        );
    }
}

#[test]
fn the_same_seed_gives_the_same_scramble() {
    let mut a = PuzzleBatch::build("?shell=C$1&cut=C$1/3", 4).expect("batch");
    let mut b = PuzzleBatch::build("?shell=C$1&cut=C$1/3", 4).expect("batch");
    a.seed_all(1234);
    b.seed_all(1234);
    a.reset(20).expect("reset");
    b.reset(20).expect("reset");
    assert_eq!(
        a.observations::<u8>().expect("observations"),
        b.observations::<u8>().expect("observations")
    );
    // And two rows of one batch do not walk together.
    let k = a.sticker_count();
    let states = a.observations::<u8>().expect("observations");
    assert_ne!(states[..k], states[k..2 * k], "two rows walked together");
}

#[test]
fn a_solved_batch_reads_as_solved() {
    for recipe in SAMPLE {
        let mut b = PuzzleBatch::build(recipe, 3).expect("batch");
        assert!(
            b.solved().expect("solved").iter().all(|&s| s),
            "{recipe}: a fresh batch is not solved"
        );
        b.seed_all(5);
        b.reset(6).expect("reset");
        assert!(
            b.solved().expect("solved").iter().all(|&s| !s),
            "{recipe}: a scrambled batch reads as solved"
        );
        b.solve().expect("solve");
        assert!(
            b.solved().expect("solved").iter().all(|&s| s),
            "{recipe}: solving the batch did not solve it"
        );
    }
}

#[test]
fn a_jumbling_puzzle_falls_back_to_geometry_and_still_works() {
    let mut b = PuzzleBatch::build(JUMBLING, 2).expect("batch");
    assert!(!b.is_tabular(), "a jumbling puzzle must not claim a permutation table");
    // Solved, it still reads: every sticker is where it started. Turning it
    // is what the fallback is for, and a turn that jumbles takes a sticker
    // off the lattice, after which reading it says so instead of handing
    // back a wrong array.
    let count = b.action_count();
    assert!(count > 0);
    assert!(b.observations::<u8>().is_ok(), "a solved puzzle always reads");
    b.seed_all(4);
    b.scramble(&[12, 12]).expect("scramble");
    match b.observations::<u8>() {
        Ok(state) => assert_eq!(state.len(), b.len() * b.sticker_count()),
        Err(e) => {
            let said = e.to_string();
            assert!(
                said.contains("lattice"),
                "a jumbled state should say why it cannot be read, not {said:?}"
            );
        },
    }
}

#[test]
fn the_fast_path_is_used_where_it_should_be() {
    for recipe in SAMPLE {
        let b = PuzzleBatch::build(recipe, 2).expect("batch");
        assert!(
            b.is_tabular(),
            "{recipe} does not jumble, so a batch of it must be tabular"
        );
    }
}

#[test]
fn a_painted_frame_is_the_frame_the_rasterizer_draws() {
    // The case for the drawing table: looking each pixel's sticker up gives
    // the picture rasterizing the geometry gives. Exactly so on a solved
    // puzzle. On a scrambled one a piece carries its own triangulation and
    // outline around with it, so a pixel sitting exactly on a seam can fall
    // either way. That is allowed to be a handful of pixels and nothing more.
    const BUDGET: f64 = 0.01;
    for recipe in SAMPLE {
        let (w, h) = (72u32, 72u32);
        let pixels = (w * h) as usize;
        let mut s = sim(recipe);
        s.options_mut().draw_arrows = false;
        let lut = s.sticker_lut(w, h).expect("lut");
        let palette = lut.palette().to_vec();

        let colors = s.stickers().expect("stickers");
        assert_eq!(
            lut.frame(&colors, &palette).as_bytes(),
            s.render(w, h).as_bytes(),
            "{recipe}: the table does not draw the solved puzzle exactly"
        );

        let count = s.symbolic().expect("view").actions.action_count();
        let mut rng = Rng(0x5EED);
        for step in 0..20 {
            let a = rng.below(count);
            assert!(s.apply_action(a).expect("apply"));
            let colors = s.stickers().expect("stickers");
            let painted = lut.frame(&colors, &palette);
            let drawn = s.render(w, h);
            let differing = painted
                .as_bytes()
                .chunks_exact(4)
                .zip(drawn.as_bytes().chunks_exact(4))
                .filter(|(p, q)| p != q)
                .count();
            assert!(
                (differing as f64) <= BUDGET * pixels as f64,
                "{recipe} step {step}: {differing} of {pixels} pixels differ, over the budget"
            );
        }
    }
}

#[test]
fn a_batch_draws_what_its_own_table_paints() {
    let recipe = "?shell=C$1&cut=C$1/3";
    let (w, h) = (48u32, 48u32);
    let frame = w as usize * h as usize * 4;
    let mut b = PuzzleBatch::build(recipe, 4).expect("batch");
    b.seed_all(99);
    b.reset(9).expect("reset");

    let painted = b.render(w, h).expect("render");
    let k = b.sticker_count();
    let states = b.observations::<u8>().expect("observations");

    let mut s = sim(recipe);
    s.options_mut().draw_arrows = false;
    let lut = s.sticker_lut(w, h).expect("lut");
    for row in 0..b.len() {
        let colors: Vec<u16> = states[row * k..(row + 1) * k].iter().map(|&c| u16::from(c)).collect();
        assert_eq!(
            &painted[row * frame..(row + 1) * frame],
            lut.frame(&colors, lut.palette()).as_bytes(),
            "row {row}: the batch's frame is not the one its state paints"
        );
    }

    // Twice over, the same frames: a table is not allowed to drift.
    assert_eq!(painted, b.render(w, h).expect("render"));
}

#[test]
fn undoing_a_walk_solves_it_again() {
    for recipe in SAMPLE {
        let mut b = PuzzleBatch::build(recipe, 3).expect("batch");
        b.seed_all(21);
        b.reset(11).expect("reset");
        assert_eq!(b.move_counts(), vec![11; 3], "{recipe}");
        let everything = vec![true; 3];
        for left in (0..11).rev() {
            let done = b.undo(&everything).expect("undo");
            assert!(done.iter().all(|&d| d), "{recipe}: nothing was undone");
            assert_eq!(b.move_counts(), vec![left; 3], "{recipe}");
        }
        assert!(
            b.solved().expect("solved").iter().all(|&s| s),
            "{recipe}: undoing the walk did not solve it"
        );
        assert!(
            b.undo(&everything).expect("undo").iter().all(|&d| !d),
            "{recipe}: something was undone that had not been done"
        );
    }
}

#[test]
fn a_written_sequence_is_the_moves_it_names() {
    let recipe = "?shell=C$1&cut=C$1/3";
    let mut b = PuzzleBatch::build(recipe, 2).expect("batch");
    let mut s = sim(recipe);
    assert_eq!(b.apply("A B2' C").expect("apply"), 4);
    assert!(s.apply_action(0).expect("apply"));
    for _ in 0..2 {
        s.apply_action(3).expect("apply");
    }
    s.apply_action(4).expect("apply");
    let want: Vec<u8> = s.stickers().expect("stickers").into_iter().map(|c| c as u8).collect();
    let k = b.sticker_count();
    let seen = b.observations::<u8>().expect("observations");
    assert_eq!(&seen[..k], &want[..]);
    assert_eq!(&seen[k..], &want[..]);
    assert!(b.apply("Q").is_err(), "an unknown move should be refused");
}

#[test]
fn a_batch_hands_out_the_table_it_steps_with() {
    let mut b = PuzzleBatch::build("?shell=C$1&cut=C$1/3", 1).expect("batch");
    let k = b.sticker_count();
    let perms = b.permutations().expect("a cube has a table");
    assert_eq!(perms.len(), b.action_count() * k);
    // Stepping by the table and stepping the batch agree.
    let mut state: Vec<u8> = (0..k as u8).collect();
    let before = state.clone();
    let a = 5usize;
    for (i, slot) in state.iter_mut().enumerate() {
        *slot = before[perms[a * k + i] as usize];
    }
    b.step(&[a as u32]).expect("step");
    let ids = b.sticker_ids::<u32>().expect("ids");
    for i in 0..k {
        assert_eq!(u32::from(state[i]), ids[i], "slot {i}");
    }
    assert!(
        PuzzleBatch::build(JUMBLING, 1).expect("batch").permutations().is_none(),
        "a jumbling puzzle must not claim a table"
    );
}

#[test]
fn a_state_from_outside_goes_into_a_batch() {
    // The way in. A batch that could only reach the states it had walked to
    // itself could not be handed one from a solver, a file or a layout.
    let mut a = PuzzleBatch::build(CUBE, 3).expect("batch");
    a.seed_all(11);
    a.scramble(&[9, 9, 9]).expect("scramble");
    let ids = a.sticker_ids::<u8>().expect("ids");
    let colors = a.observations::<u8>().expect("colors");

    let mut b = PuzzleBatch::build(CUBE, 3).expect("batch");
    b.set_states(&ids).expect("set");
    assert_eq!(b.sticker_ids::<u8>().expect("ids"), ids);
    assert_eq!(b.observations::<u8>().expect("colors"), colors);
    assert_eq!(b.move_counts(), vec![0, 0, 0], "the history no longer leads there");

    // From colors, the identities are chosen (colors do not fix them),
    // and what a caller reading colors sees is what it asked for.
    let mut c = PuzzleBatch::build(CUBE, 3).expect("batch");
    c.set_colors(&colors).expect("set");
    assert_eq!(c.observations::<u8>().expect("colors"), colors);

    // And a state put in steps like one walked to.
    let moves = vec![4u32; 3];
    b.step(&moves).expect("step");
    a.step(&moves).expect("step");
    assert_eq!(b.sticker_ids::<u8>().expect("ids"), a.sticker_ids::<u8>().expect("ids"));
}

#[test]
fn a_state_that_is_not_one_is_refused() {
    let mut b = PuzzleBatch::build(CUBE, 2).expect("batch");
    let k = b.sticker_count();
    let said = b.set_states(&vec![0u8; k * 2]).unwrap_err().to_string();
    assert!(said.contains("arrangement"), "{said:?}");
    let said = b.set_colors(&vec![0u8; k * 2]).unwrap_err().to_string();
    assert!(said.contains("colors of this puzzle"), "{said:?}");
    let said = b.set_states(&vec![0u8; k]).unwrap_err().to_string();
    assert!(said.contains("not a state each"), "{said:?}");
    let mut jumbler = PuzzleBatch::build(JUMBLING, 2).expect("batch");
    let wide = jumbler.sticker_count() * 2;
    let said = jumbler.set_states(&vec![0u8; wide]).unwrap_err().to_string();
    assert!(said.contains("geometry"), "{said:?}");
}
