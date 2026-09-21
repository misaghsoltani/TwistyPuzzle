//! The face-by-face layout, checked against recorded permutations.
//!
//! A cube's layout is a set of conventions (which face is numbered first,
//! which way a face's rows run, which way a move turns), and conventions
//! cannot be checked against the code that implements them.
//! `tests/data/facelet_cube.txt` holds the twelve move permutations and 128 move
//! sequences with the state each reaches, written down independently of this crate.
//! These tests require the layout derived from the puzzle's geometry to reproduce them exactly.
//!
//! Every other puzzle is numbered in its own order, which is already
//! canonical, so what is checked there is that the layout and the simulator
//! agree about what a move does.

use twistypuzzle::layout::{Layout, FACES};
use twistypuzzle::simulator::Simulator;

/// The recorded layout.
struct Fixture {
    size: usize,
    names: Vec<String>,
    perms: Vec<Vec<u32>>,
    goal_colors: Vec<u8>,
    goal_ids: Vec<u32>,
    cases: Vec<(Vec<u32>, Vec<u8>, Vec<u32>)>,
}

fn fixture() -> Fixture {
    let text = include_str!("data/facelet_cube.txt");
    let mut f = Fixture {
        size: 0,
        names: Vec::new(),
        perms: Vec::new(),
        goal_colors: Vec::new(),
        goal_ids: Vec::new(),
        cases: Vec::new(),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tag, rest) = line.split_once(' ').expect("a tag and its values");
        match tag {
            "size" => f.size = rest.trim().parse().expect("size"),
            "names" => f.names = rest.split_whitespace().map(str::to_string).collect(),
            "perm" => {
                let (_name, values) = rest.split_once(' ').expect("a name and a permutation");
                f.perms.push(numbers(values));
            },
            "goal_colors" => f.goal_colors = numbers(rest).into_iter().map(|v| v as u8).collect(),
            "goal_ids" => f.goal_ids = numbers(rest),
            "case" => {
                let mut parts = rest.split('|');
                let moves = numbers(parts.next().expect("moves"));
                let colors = numbers(parts.next().expect("colors"))
                    .into_iter()
                    .map(|v| v as u8)
                    .collect();
                let ids = numbers(parts.next().expect("ids"));
                f.cases.push((moves, colors, ids));
            },
            other => panic!("unknown tag {other:?} in the fixture"),
        }
    }
    assert_eq!(f.perms.len(), 12, "the fixture should hold twelve moves");
    assert!(!f.cases.is_empty());
    f
}

fn numbers(text: &str) -> Vec<u32> {
    text.split_whitespace().map(|v| v.parse().expect("a number")).collect()
}

fn layout_of(recipe: &str) -> Layout {
    let mut sim = Simulator::from_query(recipe).expect("build");
    Layout::build(&mut sim).expect("layout")
}

const CUBE: &str = "?shell=C$1&cut=C$1/3";
const BIG_CUBE: &str = "?shell=C$1&cut=C$1/2&cut=C$0";
const MEGAMINX: &str = "?shell=D$1&cut=D$2/(sqrt(5)+1)";
const PYRAMINX: &str = "?shell=T$1&cut=T$-1/3&cut=T$-5/3";
const JUMBLER: &str = "?shell=T$sqrt(2)&cut=C$0";

#[test]
fn the_moves_are_the_recorded_permutations() {
    let f = fixture();
    let layout = layout_of(CUBE);
    assert_eq!(layout.size(), Some(f.size));
    assert!(layout.is_cube());
    assert_eq!(layout.facelet_count(), 54);
    assert_eq!(layout.move_names(), f.names.as_slice());
    for (m, want) in f.perms.iter().enumerate() {
        assert_eq!(
            layout.moves()[m],
            *want,
            "move {} ({}) is not the recorded permutation",
            m,
            f.names[m]
        );
    }
    assert_eq!(layout.goal_colors::<u8>(), f.goal_colors);
    assert_eq!(layout.goal_ids::<u32>(), f.goal_ids);
}

#[test]
fn every_recorded_sequence_reaches_the_recorded_state() {
    let f = fixture();
    let layout = layout_of(CUBE);
    for (case, (moves, colors, ids)) in f.cases.iter().enumerate() {
        let mut state = layout.goal_colors::<u8>();
        let mut identity = layout.goal_ids::<u32>();
        for &m in moves {
            state = layout.next_states(&state, &[m]).expect("next");
            identity = layout.next_states(&identity, &[m]).expect("next");
        }
        assert_eq!(&state, colors, "case {case}: the colors differ");
        assert_eq!(&identity, ids, "case {case}: the identities differ");
    }
}

#[test]
fn a_move_is_the_turn_the_simulator_makes() {
    // The layout says which of the puzzle's own actions each move is. Turning
    // the puzzle that way and reading its state back through the layout must
    // give what the permutation says, or the two descriptions of the same
    // puzzle have come apart. This holds across all puzzles.
    for recipe in [CUBE, BIG_CUBE, MEGAMINX, PYRAMINX] {
        let mut sim = Simulator::from_query(recipe).expect("build");
        let layout = Layout::build(&mut sim).expect("layout");
        for (m, action) in layout.actions().iter().enumerate() {
            let action = action.unwrap_or_else(|| {
                panic!(
                    "{recipe}: move {} has no action of the puzzle behind it",
                    layout.move_names()[m]
                )
            });
            let mut s = Simulator::from_query(recipe).expect("build");
            assert!(s.apply_action(action as usize).expect("apply"));
            let colors: Vec<u16> = s.stickers().expect("stickers");
            let seen = layout.to_facelets(&colors).expect("to facelets");
            let want = layout
                .next_states(&layout.goal_colors::<u16>(), &[m as u32])
                .expect("next");
            assert_eq!(
                seen,
                want,
                "{recipe}: move {} and action {action} disagree",
                layout.move_names()[m]
            );
        }
    }
}

#[test]
fn moves_can_be_named_either_way() {
    let layout = layout_of(CUBE);
    for (face, name) in FACES.iter().enumerate() {
        assert_eq!(layout.move_index(&format!("{name}-1")), Some(face * 2));
        assert_eq!(layout.move_index(&format!("{name}'")), Some(face * 2));
        assert_eq!(layout.move_index(&format!("{name}1")), Some(face * 2 + 1));
        assert_eq!(layout.move_index(name), Some(face * 2 + 1));
    }
    assert_eq!(layout.move_index("U2"), None);
    assert_eq!(layout.move_index("Q"), None);
}

#[test]
fn a_state_survives_the_round_trip_through_the_layout() {
    for recipe in [CUBE, MEGAMINX] {
        let mut sim = Simulator::from_query(recipe).expect("build");
        let layout = Layout::build(&mut sim).expect("layout");
        sim.seed(4);
        sim.scramble(25);
        sim.settle().expect("settle");
        let colors: Vec<u16> = sim.stickers().expect("stickers");
        let facelets = layout.to_facelets(&colors).expect("to facelets");
        assert_eq!(
            layout.from_facelets(&facelets).expect("from facelets"),
            colors,
            "{recipe}: a state did not survive the round trip"
        );
        // And every face is still whole.
        for (face, &size) in layout.face_sizes().iter().enumerate() {
            let held = facelets.iter().filter(|&&f| usize::from(f) == face).count();
            assert_eq!(held, size as usize, "{recipe}: face {face} lost a sticker");
        }
    }
}

#[test]
fn the_inverse_of_a_move_undoes_it() {
    for recipe in [CUBE, MEGAMINX, PYRAMINX] {
        let layout = layout_of(recipe);
        let goal = layout.goal_ids::<u32>();
        for m in 0..layout.move_count() as u32 {
            let back = layout.inverse_moves(&[m]).expect("an inverse");
            let once = layout.next_states(&goal, &[m]).expect("next");
            let home = layout.next_states(&once, &back).expect("next");
            assert_eq!(home, goal, "{recipe}: move {m} was not undone by {back:?}");
        }
    }
}

#[test]
fn four_of_a_cubes_quarter_turn_is_the_identity() {
    let layout = layout_of(CUBE);
    let goal = layout.goal_ids::<u32>();
    for m in 0..layout.move_count() as u32 {
        let mut state = goal.clone();
        for _ in 0..4 {
            state = layout.next_states(&state, &[m]).expect("next");
        }
        assert_eq!(state, goal, "four of move {m} is not the identity");
    }
}

#[test]
fn a_walk_of_k_moves_is_k_moves_from_solved() {
    for recipe in [CUBE, MEGAMINX] {
        let layout = layout_of(recipe);
        let depths: Vec<u32> = vec![0, 1, 5, 12];
        let (states, moves) = layout.trajectories::<u8>(&depths, 7).expect("trajectories");
        let k = layout.facelet_count();
        let longest = 12usize;
        let goal = layout.goal_colors::<u8>();
        for (row, &depth) in depths.iter().enumerate() {
            let path = &states[row * (longest + 1) * k..(row + 1) * (longest + 1) * k];
            assert_eq!(&path[..k], goal.as_slice(), "{recipe}: row {row}");
            let taken = &moves[row * longest..row * longest + depth as usize];
            assert_eq!(taken.len(), depth as usize);
            // Walking the recorded moves reproduces the recorded path.
            let mut state = goal.clone();
            for (t, &m) in taken.iter().enumerate() {
                state = layout.next_states(&state, &[m]).expect("next");
                assert_eq!(
                    &path[(t + 1) * k..(t + 2) * k],
                    state.as_slice(),
                    "{recipe}: row {row} step {t}"
                );
            }
            // And walking them back solves it, so the walk is honest about
            // its own length.
            for &m in taken.iter().rev() {
                let back = layout.inverse_moves(&[m]).expect("an inverse");
                state = layout.next_states(&state, &back).expect("next");
            }
            assert_eq!(state, goal, "{recipe}: row {row} did not come home");
            assert_eq!(
                layout.solved(&path[..k]).expect("solved"),
                vec![true],
                "{recipe}: row {row} starts unsolved"
            );
        }
        // The rows that stopped early are marked, not left as a real move.
        for (row, &depth) in depths.iter().enumerate() {
            for t in depth as usize..longest {
                assert_eq!(
                    moves[row * longest + t],
                    layout.move_count() as u32,
                    "{recipe}: row {row} step {t} should be the no-move mark"
                );
            }
        }
    }
}

#[test]
fn a_walk_ends_where_its_trajectory_does() {
    let layout = layout_of(CUBE);
    let depths: Vec<u32> = vec![3, 9, 9, 0];
    let (path, _) = layout.trajectories::<u8>(&depths, 19).expect("trajectories");
    let ends = layout.walk::<u8>(&depths, 19).expect("walk");
    let k = layout.facelet_count();
    let longest = 9usize;
    for row in 0..depths.len() {
        let last = &path[row * (longest + 1) * k + longest * k..][..k];
        assert_eq!(last, &ends[row * k..(row + 1) * k], "row {row}");
    }
}

#[test]
fn a_puzzle_that_is_not_a_cube_is_numbered_in_its_own_order() {
    // A dodecahedron has no cube conventions to follow, so the layout is the
    // puzzle's own numbering, and the translation is the identity.
    let layout = layout_of("?shell=D$1&cut=D$sqrt(5)/5");
    assert!(!layout.is_cube());
    assert_eq!(layout.size(), None);
    assert_eq!(layout.face_count(), 12);
    let k = layout.facelet_count();
    assert_eq!(layout.facelet_of_slot(), (0..k as u32).collect::<Vec<_>>());
    assert_eq!(layout.slot_of_facelet(), (0..k as u32).collect::<Vec<_>>());
    // Slots are numbered in color order, so the goal is still a run of each
    // face in turn.
    let goal = layout.goal_colors::<u8>();
    assert!(goal.windows(2).all(|w| w[0] <= w[1]), "the goal is sorted");
    assert_eq!(goal.iter().fold(0, |n, &f| n + usize::from(f == 0)), k / 12);
}

#[test]
fn a_bigger_cube_keeps_the_conventions_and_adds_its_slices() {
    // The layout is derived, not tabulated, so a 4x4x4 works out to 96
    // facelets, its twelve outer-layer turns under the usual names, and the
    // twelve inner slices behind them under the puzzle's own.
    let layout = layout_of(BIG_CUBE);
    assert_eq!(layout.size(), Some(4));
    assert_eq!(layout.facelet_count(), 96);
    assert_eq!(layout.move_count(), 24);
    assert_eq!(&layout.move_names()[..2], &["U-1".to_string(), "U1".into()]);
    assert!(
        layout.move_names()[12..].iter().all(|n| n.starts_with(':')),
        "a turn the cube conventions do not name should be marked as the puzzle's own"
    );
    assert_eq!(layout.move_index("F1"), Some(11), "the face, not the slice");
    assert_eq!(layout.move_index(":F1"), Some(22));
    let goal = layout.goal_colors::<u8>();
    assert_eq!(goal.iter().fold(0, |n, &f| n + usize::from(f == 0)), 16);
    for m in 0..layout.move_count() as u32 {
        let once = layout.next_states(&goal, &[m]).expect("next");
        assert_ne!(once, goal, "move {m} changed nothing");
        let back = layout.inverse_moves(&[m]).expect("an inverse");
        assert_eq!(
            layout.next_states(&once, &back).expect("next"),
            goal,
            "move {m} was not undone"
        );
    }
}

#[test]
fn one_hot_marks_exactly_one_face_per_facelet() {
    for recipe in [CUBE, MEGAMINX] {
        let layout = layout_of(recipe);
        let (states, _) = layout.trajectories::<u8>(&[6, 6], 11).expect("trajectories");
        let k = layout.facelet_count();
        let faces = layout.face_count();
        let last = &states[6 * k..7 * k];
        let hot = layout.one_hot(last).expect("one hot");
        assert_eq!(hot.len(), k * faces);
        for i in 0..k {
            let row = &hot[i * faces..(i + 1) * faces];
            assert_eq!(row.iter().map(|&v| u32::from(v)).sum::<u32>(), 1);
            assert_eq!(row[last[i] as usize], 1);
        }
    }
}

#[test]
fn a_macro_is_the_moves_it_is_made_of() {
    let layout = layout_of(PYRAMINX);
    let pairs: Vec<Vec<u32>> = (0..4u32).flat_map(|a| (0..4u32).map(move |b| vec![a, b])).collect();
    let macros = layout.with_moves(&pairs).expect("macros");
    assert_eq!(macros.move_count(), 16);
    assert_eq!(macros.facelet_count(), layout.facelet_count());
    let goal = layout.goal_colors::<u8>();
    for (m, parts) in pairs.iter().enumerate() {
        let mut step = goal.clone();
        for &p in parts {
            step = layout.next_states(&step, &[p]).expect("next");
        }
        let once = macros.next_states(&goal, &[m as u32]).expect("next");
        assert_eq!(once, step, "macro {m} is not the moves it is made of");
    }
}

#[test]
fn a_jumbling_puzzle_has_no_layout() {
    // A turn that leaves a sticker where none sits when the puzzle is solved
    // is not a permutation of anything, so there is nothing to number.
    let mut sim = Simulator::from_query(JUMBLER).expect("build");
    let said = match Layout::build(&mut sim) {
        Ok(layout) => panic!("a jumbling puzzle got a layout of {} moves", layout.move_count()),
        Err(e) => e.to_string(),
    };
    assert!(
        said.contains("jumbles") || said.contains("permutations"),
        "the refusal should say why, not {said:?}"
    );
}

#[test]
fn a_state_can_be_read_at_any_width() {
    // The same layout, the same answers, whatever width the caller keeps its
    // states in. A puzzle with more than 256 colors needs the wider ones.
    let layout = layout_of(CUBE);
    let narrow = layout.walk::<u8>(&[7; 4], 5).expect("walk");
    let wide = layout.walk::<u16>(&[7; 4], 5).expect("walk");
    let widest = layout.walk::<u32>(&[7; 4], 5).expect("walk");
    assert_eq!(narrow.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), widest);
    assert_eq!(wide.iter().map(|&v| u32::from(v)).collect::<Vec<_>>(), widest);
}

#[test]
fn every_action_turns_something() {
    // A grip holding one piece (a tip) still turns: the piece spins in
    // its socket, carrying its own stickers and nothing else. Eight of the
    // cataloged Pyraminx's sixteen actions are those four tips, and each has
    // to move exactly the three stickers of the piece it holds. They were
    // once the identity here, because the stops of a grip were looked for
    // only among the planes that *divide* each half and a half of one piece
    // is divided by nothing (`SEMANTICS.md` §14).
    let layout = layout_of(PYRAMINX);
    let goal = layout.goal_ids::<u32>();
    let mut tips = 0;
    for m in 0..layout.move_count() as u32 {
        let after = layout.next_states(&goal, &[m]).expect("next");
        let moved = goal.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert_ne!(moved, 0, "move {m} of the Pyraminx turns nothing");
        if moved == 3 {
            tips += 1;
            // A third of a turn, so three of them come back and two do not.
            let twice = layout.next_states(&after, &[m]).expect("next");
            assert_ne!(twice, goal);
            assert_eq!(layout.next_states(&twice, &[m]).expect("next"), goal);
        }
    }
    assert_eq!(tips, 8, "four tips, each turnable both ways");
}
