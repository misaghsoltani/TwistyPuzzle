//! Render the interface to a PNG without opening a window.
//!
//! Slint's software renderer will draw into any buffer, so the whole interface
//! can be produced headlessly on a build machine, in CI, or anywhere there is
//! no display server. That makes the layout something a test can inspect rather
//! than something a person has to.
//!
//!     cargo run -p twistypuzzle-gui example screenshot docs/gui.png

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

#[cfg(target_os = "macos")]
fn platform_system_is_dark() -> Option<bool> {
    let out = std::process::Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
        .ok()?;
    Some(out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "Dark")
}

#[cfg(target_os = "windows")]
fn platform_system_is_dark() -> Option<bool> {
    let out = std::process::Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v",
            "AppsUseLightTheme",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains("AppsUseLightTheme") {
            if line.contains("0x0") {
                return Some(true);
            }
            if line.contains("0x1") {
                return Some(false);
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn platform_system_is_dark() -> Option<bool> {
    if let Ok(out) = std::process::Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "color-scheme"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if s.contains("prefer-dark") {
                return Some(true);
            }
            if s.contains("prefer-light") || s.contains("default") {
                return Some(false);
            }
        }
    }
    if let Ok(out) = std::process::Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "gtk-theme"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if s.contains("dark") {
                return Some(true);
            }
            if s.contains("light") {
                return Some(false);
            }
        }
    }
    None
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn platform_system_is_dark() -> Option<bool> {
    None
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let env_theme = std::env::var("THEME").ok();
    let theme_val = args
        .windows(2)
        .find(|w| w[0] == "--theme")
        .map(|w| w[1].as_str())
        .or(env_theme.as_deref());
    let theme_mode = if args.iter().any(|a| a == "--dark") || theme_val == Some("dark") || theme_val == Some("0") {
        Some(0)
    } else if args.iter().any(|a| a == "--light") || theme_val == Some("light") || theme_val == Some("1") {
        Some(1)
    } else if theme_val == Some("system") || theme_val == Some("2") {
        Some(2)
    } else {
        None
    };

    let mut skip_next = false;
    let out = args
        .iter()
        .skip(1)
        .find(|a| {
            if skip_next {
                skip_next = false;
                false
            } else if *a == "--theme" {
                skip_next = true;
                false
            } else {
                !a.starts_with("--")
            }
        })
        .cloned()
        .unwrap_or_else(|| "gui.png".into());
    let width: u32 = 1040;
    let height: u32 = 700;

    let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless { window: window.clone() }))
        .map_err(|e| format!("setting the platform: {e}"))?;

    let ui = MainWindow::new()?;
    let system_dark = platform_system_is_dark().unwrap_or(false);
    ui.invoke_set_system_theme(system_dark);
    if let Some(mode) = theme_mode {
        ui.invoke_apply_theme(mode);
    }
    window.set_size(PhysicalSize::new(width, height));

    // The same puzzle, camera and palette the application starts with.
    let recipe = catalog::find("Rubik's Cube (3x3x3)")
        .ok_or("the catalog has no 3x3x3")?
        .recipe;
    let mut sim = Simulator::from_query(recipe)?;
    sim.look_from(-28.0, 20.0, 12.0);
    sim.options_mut().background = if ui.get_is_dark() {
        [27, 30, 36, 255]
    } else {
        [228, 233, 240, 255]
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
