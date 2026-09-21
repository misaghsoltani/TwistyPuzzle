//! The sticker numbering and the move names, checked against every puzzle in
//! the catalog.
//!
//! The load-bearing question these tests answer is whether a state read from
//! the stored geometry is the state a solver would recognize. A move turns one
//! side of a cut and nothing else, so the stored configuration *is* the true
//! one: nothing is held still and no rotation is carried outside the pieces
//! (`SEMANTICS.md` §11). `a_turn_never_rotates_the_whole_puzzle` is what keeps
//! that so, and `an_undone_scramble_is_solved_again` is what would fail on any
//! puzzle where the numbering did not survive a turn.

use twistypuzzle::simulator::Simulator;
use twistypuzzle::symbolic::parse_moves;

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

/// A handful of puzzles that test every shell family cheaply.
const SAMPLE: &[&str] = &[
    "?shell=C$1&cut=C$1/3",           // 3x3x3
    "?shell=C$1&cut=C$1/2",           // 2x2x2
    "?shell=D$1&cut=D$sqrt(5)/5",     // Megaminx-ish
    "?shell=T$1&cut=T$1/3",           // tetrahedral
    "?shell=O$1&cut=O$1/3",           // octahedral
    "?shell=C$1&cut=C$1/2&cut=C$1/4", // 4x4x4-ish
];

fn sim(recipe: &str) -> Simulator {
    Simulator::from_query(recipe).unwrap_or_else(|e| panic!("building {recipe}: {e}"))
}

#[test]
fn a_fresh_puzzle_is_solved_and_its_stickers_are_numbered() {
    for recipe in SAMPLE {
        let mut s = sim(recipe);
        let view = s.symbolic().expect("symbolic view");
        let n = view.stickers.sticker_count();
        assert!(n > 0, "{recipe} has no stickers");
        assert!(view.stickers.color_count() > 1, "{recipe} has only one sticker color");
        assert_eq!(view.stickers.solved_colors().len(), n);

        let colors = s.stickers().expect("stickers");
        assert_eq!(colors.len(), n);
        assert!(s.is_solved().expect("is_solved"), "{recipe} starts unsolved");

        // The permutation of a solved puzzle is the identity.
        let ids = s.sticker_ids().expect("sticker ids");
        assert!(
            ids.iter().enumerate().all(|(i, &v)| v as usize == i),
            "{recipe} does not start in slot order"
        );
    }
}

#[test]
fn the_three_by_three_reads_the_way_a_cube_is_usually_read() {
    let mut s = sim("?shell=C$1&cut=C$1/3");
    let view = s.symbolic().expect("symbolic view");
    assert_eq!(view.stickers.sticker_count(), 54, "a 3x3x3 has 54 stickers");
    assert_eq!(view.stickers.color_count(), 6, "a 3x3x3 has six colors");
    assert_eq!(view.actions.grip_count(), 6, "a 3x3x3 has six face turns");
    assert_eq!(view.actions.action_count(), 12, "and twelve actions");

    // The solved array is `arange(54) // 9`, value for value: slots are
    // numbered in color order precisely so that it is.
    let want: Vec<u16> = (0..54u16).map(|i| i / 9).collect();
    assert_eq!(
        view.stickers.solved_colors(),
        want.as_slice(),
        "the solved array is not `arange(54) // 9`"
    );

    // Every action names a distinct move.
    let names = view.actions.action_names();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "duplicate move names: {names:?}");
}

#[test]
fn one_turn_moves_stickers_and_leaves_the_lattice_intact() {
    for recipe in SAMPLE {
        let mut s = sim(recipe);
        let before = s.stickers().expect("stickers");
        let count = s.symbolic().expect("view").actions.action_count();
        let mut moved = 0;
        for a in 0..count {
            let mut t = sim(recipe);
            if !t.apply_action(a).expect("apply") {
                continue;
            }
            // Reading it at all proves every sticker landed on a solved slot.
            let after = t.stickers().expect("stickers after a turn");
            assert_eq!(after.len(), before.len());
            if after != before {
                moved += 1;
            }
        }
        assert!(moved > 0, "{recipe}: no action changed the state");
    }
}

#[test]
fn an_undone_scramble_is_solved_again() {
    for recipe in SAMPLE {
        for seed in 0..4u64 {
            let mut s = sim(recipe);
            let solved = s.stickers().expect("stickers");
            let count = s.symbolic().expect("view").actions.action_count();

            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut applied: Vec<usize> = Vec::new();
            for _ in 0..12 {
                let a = rng.below(count);
                if s.apply_action(a).expect("apply") {
                    applied.push(a);
                }
            }
            assert!(!applied.is_empty(), "{recipe}: nothing was applied");
            assert_ne!(
                s.stickers().expect("stickers"),
                solved,
                "{recipe} seed {seed}: a twelve-move scramble changed nothing"
            );

            for &a in applied.iter().rev() {
                assert!(
                    s.apply_action(a ^ 1).expect("apply inverse"),
                    "{recipe} seed {seed}: the inverse of action {a} was not available"
                );
            }
            assert_eq!(
                s.stickers().expect("stickers"),
                solved,
                "{recipe} seed {seed}: undoing the scramble did not restore the puzzle"
            );
            assert!(s.is_solved().expect("is_solved"));
        }
    }
}

#[test]
fn a_turn_never_rotates_the_whole_puzzle() {
    // A move turns one side of a cut and leaves the other where it is. The
    // alternative, turning the other side and spinning the whole puzzle back,
    // draws the same picture but moves every slot, so the state would no
    // longer describe the frame it is drawn in (`SEMANTICS.md` §11). Nothing
    // may reintroduce that.
    for recipe in SAMPLE {
        let mut s = sim(recipe);
        let count = s.symbolic().expect("view").actions.action_count();
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        for _ in 0..24 {
            let a = rng.below(count);
            s.apply_action(a).expect("apply");
            let g = s.puzzle().global_rot;
            assert!(
                (g.w.abs() - 1.0).abs() < 1e-12 && g.x.abs() < 1e-12 && g.y.abs() < 1e-12 && g.z.abs() < 1e-12,
                "{recipe}: action {a} spun the whole puzzle to {g:?}"
            );
        }
    }
}

#[test]
fn moves_parse_and_round_trip_through_their_names() {
    let mut s = sim("?shell=C$1&cut=C$1/3");
    let view = s.symbolic().expect("view");
    let names = view.actions.action_names();
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            view.actions.action_index(name),
            Some(i),
            "{name} did not name action {i}"
        );
    }

    // A repeat count binds tighter than the prime, as cube notation reads.
    let parsed = parse_moves(&view.actions, "A A2 A'").expect("parse");
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].repeat, 1);
    assert_eq!(parsed[1].repeat, 2);
    assert_eq!(parsed[0].action, parsed[1].action);
    assert_eq!(parsed[2].action, parsed[0].action ^ 1);

    assert!(parse_moves(&view.actions, "nonsense").is_err());
}

#[test]
fn ground_atoms_describe_the_same_state_as_the_array() {
    let mut s = sim("?shell=C$1&cut=C$1/3");
    s.apply_action(0).expect("apply");
    let colors = s.stickers().expect("stickers");
    let atoms = twistypuzzle::symbolic::ground_atoms(&colors);
    assert_eq!(atoms.len(), colors.len());
    for (i, (pred, slot, color)) in atoms.iter().enumerate() {
        assert_eq!(pred, "color");
        assert_eq!(slot, &format!("s{i}"));
        assert_eq!(color, &format!("c{}", colors[i]));
    }
}

#[test]
fn every_non_jumbling_puzzle_can_be_read_and_restored() {
    for recipe in twistypuzzle::symbolic::NON_JUMBLING {
        let mut s = sim(recipe);
        let view = s
            .symbolic()
            .unwrap_or_else(|e| panic!("{recipe}: building the sticker map: {e}"));
        let count = view.actions.action_count();
        assert!(count > 0, "{recipe} has no actions");
        let solved = s
            .stickers()
            .unwrap_or_else(|e| panic!("{recipe}: reading stickers: {e}"));
        assert_eq!(solved.len(), view.stickers.sticker_count());

        let mut rng = Rng(0x5DEE_CE66_D1CE_4005);
        let mut applied: Vec<usize> = Vec::new();
        for _ in 0..8 {
            let a = rng.below(count);
            match s.apply_action(a) {
                Ok(true) => applied.push(a),
                Ok(false) => {},
                Err(e) => panic!("{recipe}: action {a}: {e}"),
            }
            s.stickers()
                .unwrap_or_else(|e| panic!("{recipe}: after {applied:?} the state no longer reads: {e}"));
        }

        for &a in applied.iter().rev() {
            assert!(
                s.apply_action(a ^ 1).unwrap_or(false),
                "{recipe}: could not undo action {a}"
            );
        }
        assert_eq!(
            s.stickers().expect("stickers"),
            solved,
            "{recipe}: undoing {applied:?} did not restore the puzzle"
        );
    }
}

/// The pinned list is the whole truth about the catalog, so re-derive it.
///
/// Slow by nature because it builds and turns all eighty-five puzzles, but the list
/// is what decides which puzzles get a learning environment, so it is worth the
/// minute it takes.
#[test]
fn every_non_jumbling_entry_is_listed() {
    let listed: std::collections::HashSet<&str> = twistypuzzle::symbolic::NON_JUMBLING.iter().copied().collect();
    assert_eq!(
        listed.len(),
        twistypuzzle::symbolic::NON_JUMBLING.len(),
        "the list repeats a recipe"
    );
    let mut wrong: Vec<String> = Vec::new();
    for entry in twistypuzzle::catalog::entries() {
        let jumbles = twistypuzzle::symbolic::jumbles(entry.recipe)
            .unwrap_or_else(|e| panic!("{} ({}): {e}", entry.name, entry.recipe));
        if jumbles == listed.contains(entry.recipe) {
            wrong.push(format!(
                "{} [{}] jumbles={jumbles} but is {}listed",
                entry.name,
                entry.recipe,
                if jumbles { "" } else { "not " }
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "NON_JUMBLING is out of date:\n  {}",
        wrong.join("\n  ")
    );
}

/// A jumbling puzzle says so, instead of handing back a wrong array.
#[test]
fn a_jumbling_puzzle_refuses_to_pretend() {
    // The Radiolarians are the textbook case: a face turn of an icosahedron
    // leaves pieces where no piece sits when the puzzle is solved.
    let recipe = "?shell=I$1&cut=I$1/3";
    assert!(twistypuzzle::symbolic::jumbles(recipe).expect("probe"));

    let mut s = sim(recipe);
    assert!(s.stickers().is_ok(), "a solved puzzle always reads");
    let count = s.symbolic().expect("view").actions.action_count();
    let mut refused = false;
    for a in 0..count {
        let mut t = sim(recipe);
        if t.apply_action(a).expect("apply") && t.stickers().is_err() {
            let msg = t.stickers().unwrap_err().to_string();
            assert!(
                msg.contains("not at any solved sticker position"),
                "unhelpful message: {msg}"
            );
            refused = true;
        }
    }
    assert!(refused, "{recipe} was expected to jumble");
}

#[test]
fn undo_takes_back_a_turn_however_it_was_made() {
    for recipe in SAMPLE {
        let mut s = sim(recipe);
        let solved = s.stickers().expect("stickers");
        assert!(!s.undo().expect("undo"), "{recipe}: undid a turn never made");

        // By direction.
        s.begin_move(0, 1).expect("turn");
        s.end_move();
        assert_ne!(s.stickers().expect("stickers"), solved, "{recipe}: turn did nothing");
        assert!(s.undo().expect("undo"));
        assert_eq!(s.stickers().expect("stickers"), solved, "{recipe}: undo of a turn");
        assert_eq!(s.history_len(), 0);

        // To a named stop, which may be a half turn the opposite direction
        // would not reach, which is the case an undo that just turns back the other
        // way gets wrong.
        let stops = s.stops(0).expect("stops").len();
        for k in 0..stops {
            s.begin_move_to(0, k).expect("turn to");
            s.end_move();
            assert!(s.undo().expect("undo"), "{recipe}: could not undo stop {k}");
            assert_eq!(
                s.stickers().expect("stickers"),
                solved,
                "{recipe}: undo of stop {k} did not restore the puzzle"
            );
        }

        // A whole scramble, unwound.
        for k in 0..6 {
            s.begin_move(k % s.grip_count(), if k % 2 == 0 { 1 } else { -1 })
                .expect("turn");
            s.end_move();
        }
        assert_eq!(s.history_len(), 6);
        while s.undo().expect("undo") {}
        assert_eq!(s.stickers().expect("stickers"), solved, "{recipe}: unwinding six turns");
    }
}

/// The permutation table has to be right for every puzzle that gets one.
///
/// Deriving it from single moves out of the solved state only shows that each
/// move is a permutation *there*. This turns every non-jumbling puzzle both
/// ways at once, by geometry and by table, and requires them to agree.
#[test]
fn the_permutation_table_agrees_with_the_geometry() {
    use twistypuzzle::symbolic::PermutationTable;

    let mut without = Vec::new();
    for recipe in twistypuzzle::symbolic::NON_JUMBLING {
        let mut s = sim(recipe);
        let Some(table) = PermutationTable::build(&mut s).unwrap_or_else(|e| panic!("{recipe}: {e}")) else {
            without.push(*recipe);
            continue;
        };
        assert_eq!(table.action_count(), s.symbolic().expect("view").actions.action_count());
        assert_eq!(table.sticker_count(), s.stickers().expect("stickers").len());
        assert!(
            s.is_solved().expect("is_solved"),
            "{recipe}: deriving disturbed the puzzle"
        );
        assert!(
            table.verify(&mut s, 24, 0x2545_F491_4F6C_DD1D).expect("verify"),
            "{recipe}: the table and the geometry disagree"
        );
        assert!(
            s.is_solved().expect("is_solved"),
            "{recipe}: verifying disturbed the puzzle"
        );
    }
    assert!(
        without.is_empty(),
        "these do not jumble but have no permutation table: {without:?}"
    );
}

/// A jumbling puzzle has no table, and says so instead of producing a wrong one.
#[test]
fn a_jumbling_puzzle_has_no_permutation_table() {
    let mut s = sim("?shell=I$1&cut=I$1/3");
    let table = twistypuzzle::symbolic::PermutationTable::build(&mut s).expect("build");
    assert!(table.is_none());
}
