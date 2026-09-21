//! Unit tests for the framebuffer, blitter and rasterizer.

use std::collections::HashMap;

use twistypuzzle::math::Vec3;
use twistypuzzle::render::camera::Mat4;
use twistypuzzle::render::font;
use twistypuzzle::render::raster::{draw, Cull, DrawCall, RenderTarget};
use twistypuzzle::render::{BlendMode, Framebuffer, Rect};

/// A vertex rounded to a fixed grid, so coincident corners compare equal.
type Key = (i64, i64, i64);

#[test]
fn fill_sets_every_pixel() {
    let mut fb = Framebuffer::new(7, 5);
    fb.fill([200, 200, 220, 255]);
    for y in 0..5 {
        for x in 0..7 {
            assert_eq!(fb.pixel(x, y), [200, 200, 220, 255], "pixel {x},{y}");
        }
    }
    // The uniform-byte fast path must agree with the general one.
    let mut g = Framebuffer::new(3, 3);
    g.fill([9, 9, 9, 9]);
    assert!(g.as_bytes().iter().all(|&b| b == 9));
}

#[test]
fn blit_clips_on_both_surfaces() {
    let src = Framebuffer::filled(4, 4, [10, 20, 30, 255]);
    let mut dst = Framebuffer::filled(8, 8, [0, 0, 0, 255]);
    // Fully inside.
    dst.blit(&src, Rect { x: 0, y: 0, w: 4, h: 4 }, 2, 2, BlendMode::Copy);
    assert_eq!(dst.pixel(2, 2), [10, 20, 30, 255]);
    assert_eq!(dst.pixel(5, 5), [10, 20, 30, 255]);
    assert_eq!(dst.pixel(1, 1), [0, 0, 0, 255]);
    assert_eq!(dst.pixel(6, 6), [0, 0, 0, 255]);
    // Straddling the top-left corner: only the visible part lands.
    let mut d2 = Framebuffer::filled(8, 8, [0, 0, 0, 255]);
    d2.blit(&src, Rect { x: 0, y: 0, w: 4, h: 4 }, -2, -2, BlendMode::Copy);
    assert_eq!(d2.pixel(0, 0), [10, 20, 30, 255]);
    assert_eq!(d2.pixel(1, 1), [10, 20, 30, 255]);
    assert_eq!(d2.pixel(2, 2), [0, 0, 0, 255]);
    // Entirely outside is a no-op.
    let before = d2.as_bytes().to_vec();
    d2.blit(&src, Rect { x: 0, y: 0, w: 4, h: 4 }, 100, 100, BlendMode::Copy);
    assert_eq!(d2.as_bytes(), &before[..]);
}

#[test]
fn over_blend_matches_the_classic_formula() {
    let src = Framebuffer::filled(2, 2, [255, 0, 0, 128]);
    let mut dst = Framebuffer::filled(2, 2, [0, 0, 255, 255]);
    dst.blit(&src, Rect { x: 0, y: 0, w: 2, h: 2 }, 0, 0, BlendMode::Over);
    let p = dst.pixel(0, 0);
    // 255*128/255 + 0*127/255 = 128 (rounded)
    assert!((i32::from(p[0]) - 128).abs() <= 1, "red {}", p[0]);
    assert!((i32::from(p[2]) - 127).abs() <= 1, "blue {}", p[2]);
    assert_eq!(p[3], 255);
}

#[test]
fn downsample_averages_blocks() {
    let mut fb = Framebuffer::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            // Two black and two white pixels per 2x2 block.
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            fb.set_pixel(x, y, [v, v, v, 255]);
        }
    }
    let out = fb.downsample(2);
    assert_eq!(out.width(), 2);
    assert_eq!(out.height(), 2);
    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(out.pixel(x, y)[0], 128, "block {x},{y}");
        }
    }
}

#[test]
fn rasterizes_a_triangle_with_the_expected_coverage() {
    let mut target = RenderTarget::new(16, 16);
    target.clear([0, 0, 0, 255]);
    // A right triangle covering the top-left half of the surface, in a plane
    // the identity transform maps straight to NDC.
    let tris = [[
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ]];
    let colors = [[255u8, 0, 0, 255]];
    let call = DrawCall {
        triangles: &tris,
        colors: &colors,
        cull: Cull::None,
        ..Default::default()
    };
    // An orthographic-ish transform: w = 1, so clip space is NDC.
    let vp = Mat4::IDENTITY;
    draw(&mut target, &[call], &vp);
    let covered = target
        .color
        .as_bytes()
        .chunks_exact(4)
        .filter(|p| p[0] == 255 && p[1] == 0)
        .count();
    // Half of 256 pixels, within a row of tolerance for the fill rule.
    assert!(
        (100..=156).contains(&covered),
        "expected about half the surface covered, got {covered}"
    );
    // The far corner must be untouched.
    assert_eq!(target.color.pixel(15, 15), [0, 0, 0, 255]);
}

#[test]
fn depth_test_keeps_the_nearer_triangle() {
    let mut target = RenderTarget::new(8, 8);
    target.clear([0, 0, 0, 255]);
    let quad = |z: f64| {
        [
            [
                Vec3::new(-1.0, 1.0, z),
                Vec3::new(-1.0, -1.0, z),
                Vec3::new(1.0, -1.0, z),
            ],
            [Vec3::new(-1.0, 1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z)],
        ]
    };
    let far = quad(0.9);
    let near = quad(-0.9);
    let far_c = [[0u8, 0, 255, 255]; 2];
    let near_c = [[0u8, 255, 0, 255]; 2];
    let calls = [
        DrawCall {
            triangles: &far,
            colors: &far_c,
            cull: Cull::None,
            ..Default::default()
        },
        DrawCall {
            triangles: &near,
            colors: &near_c,
            cull: Cull::None,
            ..Default::default()
        },
    ];
    draw(&mut target, &calls, &Mat4::IDENTITY);
    assert_eq!(target.color.pixel(4, 4), [0, 255, 0, 255], "nearer triangle should win");

    // Drawn in the opposite order, the result must be the same.
    let mut t2 = RenderTarget::new(8, 8);
    t2.clear([0, 0, 0, 255]);
    let calls = [
        DrawCall {
            triangles: &near,
            colors: &near_c,
            cull: Cull::None,
            ..Default::default()
        },
        DrawCall {
            triangles: &far,
            colors: &far_c,
            cull: Cull::None,
            ..Default::default()
        },
    ];
    draw(&mut t2, &calls, &Mat4::IDENTITY);
    assert_eq!(
        t2.color.pixel(4, 4),
        [0, 255, 0, 255],
        "depth test is order-independent"
    );
}

#[test]
fn arrow_mesh_is_a_closed_solid() {
    use twistypuzzle::render::scene::arrow_mesh;
    let m = arrow_mesh();
    assert!(!m.triangles.is_empty(), "arrow mesh is empty");

    // The two caps must have equal, non-zero area: an incomplete
    // triangulation would leave holes and the arrow would blend unevenly.
    let area = |t: &[Vec3; 3]| {
        let u = t[1].sub(&t[0]);
        let v = t[2].sub(&t[0]);
        u.cross(&v).length() * 0.5
    };
    let front: f64 = m
        .triangles
        .iter()
        .filter(|t| t.iter().all(|v| v.z == 0.0))
        .map(area)
        .sum();
    let back: f64 = m
        .triangles
        .iter()
        .filter(|t| t.iter().all(|v| v.z == 0.2))
        .map(area)
        .sum();
    assert!(front > 0.5, "front cap area {front} is implausibly small");
    assert!((front - back).abs() < 1e-9, "caps disagree: front {front}, back {back}");

    // Every edge of a closed solid is shared by exactly two triangles.
    let key = |v: Vec3| -> Key {
        (
            (v.x * 1e9).round() as i64,
            (v.y * 1e9).round() as i64,
            (v.z * 1e9).round() as i64,
        )
    };
    let mut edges: HashMap<(Key, Key), i32> = HashMap::new();
    for t in &m.triangles {
        for i in 0..3 {
            let (a, b) = (key(t[i]), key(t[(i + 1) % 3]));
            let k = if a < b { (a, b) } else { (b, a) };
            *edges.entry(k).or_insert(0) += 1;
        }
    }
    let bad: Vec<_> = edges.iter().filter(|(_, &n)| n != 2).collect();
    assert!(
        bad.is_empty(),
        "{} edges are not shared by exactly two triangles",
        bad.len()
    );
}

/* -------------------------------------------------------------------------- */
/*  Built-in font                                                             */
/* -------------------------------------------------------------------------- */

#[test]
fn every_printable_ascii_has_a_glyph() {
    // A missing glyph would silently become `?`, so check the ink instead of
    // the mapping: only the space is allowed to be blank.
    for code in 0x20u8..=0x7E {
        let ch = code as char;
        let mut fb = Framebuffer::filled(32, 32, [0, 0, 0, 255]);
        fb.draw_text(2, 2, &ch.to_string(), [255, 255, 255, 255], 2, BlendMode::Copy);
        let ink = fb.as_bytes().chunks_exact(4).filter(|p| p[0] == 255).count();
        if ch == ' ' {
            assert_eq!(ink, 0, "space should draw nothing");
        } else {
            assert!(ink > 0, "no ink for {ch:?} (U+{code:04X})");
        }
        assert!(font::text_width(&ch.to_string(), 1) > 0, "zero advance for {ch:?}");
    }
}

#[test]
fn text_width_matches_what_is_drawn() {
    let scale = 3;
    let text = "Rubik's Cube (3x3x3)";
    let w = font::text_width(text, scale);
    let mut fb = Framebuffer::filled(w + 40, 60, [0, 0, 0, 255]);
    fb.draw_text(10, 10, text, [255, 255, 255, 255], scale, BlendMode::Copy);

    let stride = fb.stride();
    let column_has_ink = |x: usize| (0..fb.height() as usize).any(|y| fb.as_bytes()[y * stride + x * 4] == 255);
    // Ink starts at the left edge of the first glyph and the last inked column
    // lies inside the reported width. Tracking is not counted after the final
    // glyph, so the right edge is exact for a string ending in an inked column.
    assert!(column_has_ink(10), "no ink at the starting pen position");
    assert!(
        column_has_ink(10 + w as usize - 1),
        "reported width overshoots the last inked column"
    );
    assert!(!column_has_ink(10 + w as usize), "ink past the reported width");
}

#[test]
fn ellipsize_fits_the_budget() {
    let scale = 2;
    let long = "Pentultimate / Master Pyraminx Crystal";
    for budget in [0, 5, 20, 60, 200, 5000] {
        let cut = font::ellipsize(long, budget, scale);
        assert!(
            font::text_width(&cut, scale) <= budget,
            "{cut:?} is wider than {budget}"
        );
        if budget >= font::text_width(long, scale) {
            assert_eq!(cut, long, "a string that fits should be left alone");
        }
    }
}

#[test]
fn descenders_fall_below_the_baseline() {
    let scale = 2;
    let base = font::baseline(scale) as usize;
    for ch in ['g', 'j', 'p', 'q', 'y'] {
        let mut fb = Framebuffer::filled(32, 40, [0, 0, 0, 255]);
        fb.draw_text(2, 0, &ch.to_string(), [255, 255, 255, 255], scale, BlendMode::Copy);
        let stride = fb.stride();
        let below = (base..fb.height() as usize)
            .any(|y| (0..fb.width() as usize).any(|x| fb.as_bytes()[y * stride + x * 4] == 255));
        assert!(below, "{ch:?} has no descender");
    }
}

#[test]
fn simulator_hover_and_clear_hover() {
    use twistypuzzle::catalog;
    use twistypuzzle::render::trackball::Pointer;
    use twistypuzzle::simulator::Simulator;

    let recipe = catalog::find("Rubik's Cube (3x3x3)").unwrap().recipe;
    let mut sim = Simulator::from_query(recipe).expect("build 3x3x3");
    sim.look_from(-28.0, 20.0, 12.0);
    assert_eq!(sim.hovered_arrow(), None);

    // Pointer over arrow 6
    sim.pointer_move(Pointer { x: 0.050, y: 0.775 });
    assert_eq!(sim.hovered_arrow(), Some(6));

    // Clearing hover resets to None
    sim.clear_hover();
    assert_eq!(sim.hovered_arrow(), None);
}

#[test]
fn simulator_pick_arrow_respects_draw_arrows_option() {
    use twistypuzzle::catalog;
    use twistypuzzle::render::trackball::Pointer;
    use twistypuzzle::simulator::Simulator;

    let recipe = catalog::find("Rubik's Cube (3x3x3)").unwrap().recipe;
    let mut sim = Simulator::from_query(recipe).expect("build 3x3x3");
    sim.look_from(-28.0, 20.0, 12.0);

    let p = Pointer { x: 0.050, y: 0.775 };
    assert_eq!(sim.pick_arrow(p), Some(6));

    // When draw_arrows is turned off, pick_arrow must return None
    sim.options_mut().draw_arrows = false;
    assert_eq!(sim.pick_arrow(p), None);

    sim.pointer_move(p);
    assert_eq!(sim.hovered_arrow(), None);

    // When re-enabled, pick_arrow works again
    sim.options_mut().draw_arrows = true;
    assert_eq!(sim.pick_arrow(p), Some(6));
}

#[test]
fn simulator_hover_visually_highlights_frame() {
    use twistypuzzle::catalog;
    use twistypuzzle::render::trackball::Pointer;
    use twistypuzzle::simulator::Simulator;

    let recipe = catalog::find("Rubik's Cube (3x3x3)").unwrap().recipe;
    let mut sim = Simulator::from_query(recipe).expect("build 3x3x3");
    sim.look_from(-28.0, 20.0, 12.0);
    sim.options_mut().background = [27, 30, 36, 255];

    let base_frame = sim.render(320, 320);

    // Hover over an arrow
    sim.pointer_move(Pointer { x: 0.050, y: 0.775 });
    assert!(sim.hovered_arrow().is_some());
    let hover_frame = sim.render(320, 320);

    // The frames must differ visually (the arrow is lit up)
    assert_ne!(
        base_frame.as_bytes(),
        hover_frame.as_bytes(),
        "hovered arrow must produce visual highlight"
    );

    // Clearing hover returns the frame exactly to base
    sim.clear_hover();
    assert_eq!(sim.hovered_arrow(), None);
    let restored_frame = sim.render(320, 320);
    assert_eq!(
        base_frame.as_bytes(),
        restored_frame.as_bytes(),
        "clearing hover must restore exact frame with zero regression"
    );
}
