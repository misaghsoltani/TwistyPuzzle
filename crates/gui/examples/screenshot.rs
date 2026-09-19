//! Render the interface to a PNG without opening a window.
//!
//! Slint's software renderer will draw into any buffer, so the whole interface
//! can be produced headlessly on a build machine, in CI, or anywhere there is
//! no display server. That makes the layout something a test can inspect rather
//! than something a person has to.
//!
//!     cargo run -p twistypuzzle-gui --example screenshot docs/gui.png

use std::rc::Rc;

use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
use slint::platform::{Platform, WindowAdapter};
use slint::{Image, PhysicalSize, PlatformError, Rgb8Pixel, SharedPixelBuffer, SharedString};
use twistypuzzle::catalog;
use twistypuzzle::simulator::Simulator;

slint::include_modules!();

/// A platform with one window and no windowing system behind it.
struct Headless {
    window: Rc<MinimalSoftwareWindow>,
}

impl Platform for Headless {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, PlatformError> {
        Ok(self.window.clone())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let is_light =
        args.iter().any(|a| a == "--light") || std::env::var("THEME").is_ok_and(|v| v == "light");
    let out = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "gui.png".into());
    let width: u32 = 1040;
    let height: u32 = 700;

    let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless {
        window: window.clone(),
    }))
    .map_err(|e| format!("setting the platform: {e}"))?;

    let ui = MainWindow::new()?;
    if is_light {
        ui.invoke_apply_theme(1);
    }
    window.set_size(PhysicalSize::new(width, height));

    // The same puzzle, camera and palette the application starts with.
    let recipe = catalog::find("Rubik's Cube (3x3x3)")
        .ok_or("the catalog has no 3x3x3")?
        .recipe;
    let mut sim = Simulator::from_query(recipe)?;
    sim.look_from(-28.0, 20.0, 12.0);
    sim.options_mut().background = if is_light {
        [228, 233, 240, 255]
    } else {
        [27, 30, 36, 255]
    };
    sim.seed(7);
    sim.scramble(12);
    sim.settle()?;

    // The stage is everything left of the 292px control panel below the 39px top bar.
    let stage_w = width - 292;
    let stage_h = height - 39;
    let frame = sim.render(stage_w, stage_h);
    let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(stage_w, stage_h);
    buffer.make_mut_bytes().copy_from_slice(frame.as_bytes());
    ui.set_frame(Image::from_rgba8(buffer));

    let names: Vec<SharedString> = catalog::entries()
        .map(|e| SharedString::from(format!("{} - {}", e.name, e.kind)))
        .collect();
    let index = catalog::entries()
        .position(|e| e.name == "Rubik's Cube (3x3x3)")
        .unwrap_or(0);
    ui.set_puzzle_names(slint::ModelRc::from(Rc::new(slint::VecModel::from(names))));
    ui.set_puzzle_index(i32::try_from(index).unwrap_or(0));
    ui.set_piece_count(i32::try_from(sim.piece_count()).unwrap_or(0));
    ui.set_grip_count(i32::try_from(sim.grip_count()).unwrap_or(0));
    ui.set_status("12 moves made, 12 to undo".into());
    ui.set_hint(
        "Drag to turn the view, click an arrow to turn a layer, scroll to zoom.\n\
         Space scrambles, R resets, U undoes."
            .into(),
    );
    ui.set_move_log("built Rubik's Cube (3x3x3)\nscrambled 12".into());
    ui.set_can_undo(true);

    // Two passes: the first lays the interface out, the second draws it with
    // every size settled.
    let mut pixels = vec![Rgb8Pixel::default(); (width * height) as usize];
    for _ in 0..2 {
        window.request_redraw();
        slint::platform::update_timers_and_animations();
        window.draw_if_needed(|renderer| {
            renderer.render(&mut pixels, width as usize);
        });
    }

    let mut rgba = Vec::with_capacity(pixels.len() * 4);
    for p in &pixels {
        rgba.extend_from_slice(&[p.r, p.g, p.b, 255]);
    }
    let shot = twistypuzzle::render::Framebuffer::from_raw(width, height, rgba)
        .ok_or("the screenshot buffer is the wrong size")?;
    std::fs::write(&out, shot.to_png())?;
    println!("wrote {out} ({width}x{height})");
    Ok(())
}
