//! A desktop interface for the twisty puzzle simulator.
//!
//! The simulator already draws its own frames in software, with no GPU, display
//! server or browser involved. This puts a window around them: the interface is
//! Slint with its software renderer, so the whole application draws the same way
//! on every platform and needs no graphics driver to do it.
//!
//! # How it is put together
//!
//! One thread owns the puzzle. Every Slint callback runs on the UI thread, so
//! the lock around the state is never contended, acting as a lock rather than a
//! `RefCell` only because a puzzle built on a worker has to travel back through
//! the event loop, which requires whatever it carries to be `Send`.
//!
//! Building a puzzle is the one slow thing here (the deeper catalog entries
//! take seconds of exact arithmetic), so it happens on a worker thread and comes
//! back through the event loop. The window stays responsive and says what it is
//! doing rather than freezing, which is the difference between an application
//! that feels broken and one that feels busy.
//!
//! Frames are redrawn when something changes, not on a clock: a still puzzle
//! costs nothing. While a turn animates or the pointer drags, a 60 Hz timer
//! takes over.

use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use slint::winit_030::WinitWindowAccessor;
use slint::{Image, ModelRc, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel};
use twistypuzzle::catalog::{self, CatalogEntry};
use twistypuzzle::render::trackball::Pointer;
use twistypuzzle::simulator::{Simulator, SymbolicView};

slint::include_modules!();

/// How many moves the log keeps before the oldest scroll off.
const LOG_LINES: usize = 8;

/// Frame interval while something is moving.
const FRAME: Duration = Duration::from_millis(16);

/// The state, shared with the worker thread that builds a puzzle.
///
/// Every interface callback runs on the one UI thread, so this is never
/// contended, acting as a `Mutex` rather than a `RefCell` only because the built
/// puzzle has to come back from a worker through Slint's event loop, which
/// requires whatever it carries to be `Send`.
type Shared = Arc<Mutex<App>>;

/// Take the lock, recovering from a poisoned one rather than compounding a
/// panic that already happened.
fn lock(app: &Shared) -> MutexGuard<'_, App> {
    app.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Everything the interface owns, on the one thread that owns it.
struct App {
    /// `None` only while the first puzzle is being built.
    sim: Option<Simulator>,
    entries: Vec<CatalogEntry>,
    /// Physical pixels, so the frame is sharp on a scaled display.
    width: u32,
    height: u32,
    /// Bumped for every build request, so a build that finishes after the user
    /// has moved on is discarded rather than replacing what they chose.
    generation: u64,
    log: Vec<String>,
    /// True between a press and the release that ends it.
    dragging: bool,
    /// Set when the puzzle has changed and the frame has not caught up.
    dirty: bool,
    last_tick: Instant,
    /// The move names, so the log can say `A` rather than `layer 3`. Absent
    /// only if naming them failed.
    names: Option<Arc<SymbolicView>>,
    /// Worked out when the puzzle changes rather than once a frame: reading a
    /// state walks every sticker, and on a puzzle that jumbles it walks them
    /// only to fail.
    solved: Solvedness,
}

/// What is known about whether the puzzle is solved.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Solvedness {
    No,
    Yes,
    /// The puzzle jumbles, so it has no fixed set of sticker slots and there is
    /// nothing to compare against. Asked once and never again.
    Unreadable,
}

impl App {
    fn new(entries: Vec<CatalogEntry>) -> App {
        App {
            sim: None,
            entries,
            width: 640,
            height: 640,
            generation: 0,
            log: Vec::new(),
            dragging: false,
            dirty: true,
            last_tick: Instant::now(),
            names: None,
            solved: Solvedness::No,
        }
    }

    /// Work out whether the puzzle is solved, once, after it has changed.
    fn settle_solved(&mut self) {
        if self.solved == Solvedness::Unreadable {
            return;
        }
        let Some(sim) = self.sim.as_mut() else {
            self.solved = Solvedness::No;
            return;
        };
        self.solved = match sim.is_solved() {
            Ok(true) => Solvedness::Yes,
            Ok(false) => Solvedness::No,
            Err(_) => Solvedness::Unreadable,
        };
    }

    /// The name the solved puzzle gave the grip now at `index`.
    fn grip_name(&self, index: usize) -> String {
        let named = self
            .names
            .as_ref()
            .zip(self.sim.as_ref())
            .and_then(|(v, sim)| {
                let cut = sim.grips().get(index)?;
                v.actions.name_of_plane(&cut.plane).map(str::to_string)
            });
        named.unwrap_or_else(|| format!("layer {index}"))
    }

    fn note(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > LOG_LINES {
            self.log.remove(0);
        }
    }
}

/// What the command line asked for.
enum Request {
    /// Open the window, starting on this puzzle if one was named.
    Open(Option<String>),
    /// Print something and stop.
    Print(String),
    /// Complain and stop.
    Fail(String),
}

/// Read the command line.
///
/// Small enough to do by hand: three options, none of which take more than a
/// word, and an argument parser would be a dependency for no benefit.
fn parse_args(args: &[String]) -> Request {
    let mut wanted: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                return Request::Print(
                    "twistypuzzle-gui - a window for the twisty puzzle simulator\n\
                     \n\
                     Usage: twistypuzzle-gui [--puzzle NAME]\n\
                     \n\
                     Options:\n\
                     \x20 -p, --puzzle NAME   Open this cataloged puzzle\n\
                     \x20 -V, --version       Print the version\n\
                     \x20 -h, --help          Print this\n"
                        .into(),
                );
            },
            "-V" | "--version" => {
                return Request::Print(format!("twistypuzzle-gui {}\n", env!("CARGO_PKG_VERSION")));
            },
            "-p" | "--puzzle" => match rest.next() {
                Some(name) => wanted = Some(name.clone()),
                None => return Request::Fail("--puzzle needs the name of a puzzle".into()),
            },
            other => return Request::Fail(format!("unknown option {other:?}, try --help")),
        }
    }
    Request::Open(wanted)
}

#[cfg(target_os = "macos")]
mod macos_dock {
    #[allow(unsafe_code)]
    pub fn init_dock_icon() {
        extern "C" {
            fn set_macos_dock_icon(bytes: *const u8, len: i32);
        }
        let png = include_bytes!("../ui/app-icon.png");
        if let Ok(len) = i32::try_from(png.len()) {
            unsafe {
                set_macos_dock_icon(png.as_ptr(), len);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    macos_dock::init_dock_icon();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let wanted = match parse_args(&args) {
        Request::Open(w) => w,
        Request::Print(text) => {
            print!("{text}");
            return Ok(());
        },
        Request::Fail(why) => {
            eprintln!("twistypuzzle-gui: {why}");
            std::process::exit(2);
        },
    };

    let entries: Vec<CatalogEntry> = catalog::entries().collect();
    if let Some(name) = &wanted {
        if !entries.iter().any(|e| e.name.eq_ignore_ascii_case(name)) {
            eprintln!("twistypuzzle-gui: no cataloged puzzle is called {name:?}");
            std::process::exit(2);
        }
    }
    let ui = MainWindow::new()?;

    // Ten catalog entries are called "Unknown", so the menu shows what each
    // one is made of rather than ten identical lines.
    let names: Vec<SharedString> = entries
        .iter()
        .map(|e| {
            if e.name == "Unknown" {
                SharedString::from(format!("{} ({})", e.recipe, e.kind))
            } else {
                SharedString::from(format!("{} - {}", e.name, e.kind))
            }
        })
        .collect();
    let model: Rc<VecModel<SharedString>> = Rc::new(VecModel::from(names));
    ui.set_puzzle_names(ModelRc::from(model.clone()));

    let start = wanted
        .as_deref()
        .and_then(|name| {
            entries
                .iter()
                .position(|e| e.name.eq_ignore_ascii_case(name))
        })
        .or_else(|| {
            entries
                .iter()
                .position(|e| e.name == "Rubik's Cube (3x3x3)")
        })
        .unwrap_or(0);
    ui.set_puzzle_index(i32::try_from(start).unwrap_or(0));
    ui.set_hint(
        "Drag to turn the view, click an arrow to turn a layer, scroll to zoom.\n\
         Space scrambles, R resets, U undoes."
            .into(),
    );

    let app: Shared = Arc::new(Mutex::new(App::new(entries)));

    wire_callbacks(&ui, &app);
    begin_build(&ui, &app, start);

    // One timer for the whole application. It does nothing at all unless
    // something is moving, so a still window is idle.
    let tick = Timer::default();
    {
        let weak = ui.as_weak();
        let app = Arc::clone(&app);
        tick.start(TimerMode::Repeated, FRAME, move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut state = lock(&app);
            let elapsed = state.last_tick.elapsed();
            state.last_tick = Instant::now();
            let was_animating = state
                .sim
                .as_ref()
                .is_some_and(twistypuzzle::simulator::Simulator::is_animating);
            if was_animating || state.dragging {
                if let Some(sim) = state.sim.as_mut() {
                    let ms = elapsed.as_secs_f64() * 1000.0;
                    let _ = sim.advance(ms.clamp(1.0, 100.0));
                }
                state.dirty = true;
            }
            let is_animating = state
                .sim
                .as_ref()
                .is_some_and(twistypuzzle::simulator::Simulator::is_animating);
            if was_animating && !is_animating {
                // The move just finished, so the state is worth asking about.
                state.settle_solved();
                state.dirty = true;
            }
            if state.dirty {
                state.dirty = false;
                drop(state);
                refresh(&ui, &app);
                redraw(&ui, &app);
            }
        });
    }

    ui.run()?;
    Ok(())
}

/// Connect every control to the puzzle.
#[allow(clippy::too_many_lines)]
fn wire_callbacks(ui: &MainWindow, app: &Shared) {
    macro_rules! with {
        (|$ui:ident, $state:ident| $body:block) => {{
            let weak = ui.as_weak();
            let app = Arc::clone(app);
            move || {
                let Some($ui) = weak.upgrade() else { return };
                let mut $state = lock(&app);
                $body
                drop($state);
                refresh(&$ui, &app);
            }
        }};
    }

    ui.on_scramble(with!(|ui, state| {
        let depth = ui.get_scramble_depth().round().max(1.0) as u32;
        let animate = ui.get_animate_scramble();
        if let Some(sim) = state.sim.as_mut() {
            sim.clear_hover();
            ui.set_arrow_hovered(false);
            sim.scramble(depth);
            if animate {
                let _ = sim.advance(0.0);
                state.note(format!("scrambled {depth}"));
            } else {
                if let Err(e) = sim.settle() {
                    state.note(format!("scramble failed: {e}"));
                } else {
                    state.note(format!("scrambled {depth}"));
                }
                state.settle_solved();
            }
        }
        state.dirty = true;
    }));

    ui.on_animate_scramble_changed(with!(|ui, state| {
        if !ui.get_animate_scramble() {
            if let Some(sim) = state.sim.as_mut() {
                let _ = sim.settle();
            }
            state.settle_solved();
            state.dirty = true;
        }
    }));

    ui.on_reset(with!(|ui, state| {
        let _ = &ui;
        if let Some(sim) = state.sim.as_mut() {
            sim.cancel_scramble();
            sim.clear_hover();
            ui.set_arrow_hovered(false);
            match sim.restore() {
                Ok(true) => state.note("reset".into()),
                Ok(false) => state.note("reset: a layer is locked".into()),
                Err(e) => state.note(format!("reset failed: {e}")),
            }
        }
        state.settle_solved();
        state.dirty = true;
    }));

    ui.on_undo(with!(|ui, state| {
        let _ = &ui;
        if let Some(sim) = state.sim.as_mut() {
            sim.cancel_scramble();
            sim.clear_hover();
            ui.set_arrow_hovered(false);
            match sim.undo() {
                Ok(true) => state.note("undo".into()),
                Ok(false) => state.note("nothing to undo".into()),
                Err(e) => state.note(format!("undo failed: {e}")),
            }
        }
        state.settle_solved();
        state.dirty = true;
    }));

    ui.on_options_changed(with!(|ui, state| {
        if let Some(sim) = state.sim.as_mut() {
            let show_arrows = ui.get_show_arrows();
            sim.options_mut().draw_arrows = show_arrows;
            sim.options_mut().draw_edges = ui.get_show_edges();
            if !show_arrows {
                sim.clear_hover();
                ui.set_arrow_hovered(false);
            }
        }
        state.dirty = true;
    }));

    ui.on_theme_changed(with!(|ui, state| {
        if let Some(sim) = state.sim.as_mut() {
            sim.options_mut().background = if ui.get_is_dark() {
                [27, 30, 36, 255]
            } else {
                [228, 233, 240, 255]
            };
        }
        state.dirty = true;
    }));

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_chose(move |index| {
            let Some(ui) = weak.upgrade() else { return };
            begin_build(&ui, &app, index.max(0) as usize);
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_viewport_resized(move |w, h| {
            let Some(ui) = weak.upgrade() else { return };
            let is_max = ui.window().is_maximized();
            if ui.get_is_maximized() != is_max {
                ui.set_is_maximized(is_max);
            }
            let scale = f64::from(ui.window().scale_factor());
            let mut state = lock(&app);
            let width = ((f64::from(w) * scale).round() as u32).clamp(64, 4096);
            let height = ((f64::from(h) * scale).round() as u32).clamp(64, 4096);
            if (width, height) == (state.width, state.height) {
                return;
            }
            state.width = width;
            state.height = height;
            state.dirty = true;
        });
    }

    {
        let weak = ui.as_weak();
        ui.on_minimize_window(move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.window().set_minimized(true);
        });
    }

    {
        let weak = ui.as_weak();
        ui.on_toggle_maximize(move || {
            let Some(ui) = weak.upgrade() else { return };
            let is_max = ui.window().is_maximized();
            ui.window().set_maximized(!is_max);
            ui.set_is_maximized(!is_max);
        });
    }

    {
        let weak = ui.as_weak();
        ui.on_close_window(move || {
            let Some(ui) = weak.upgrade() else { return };
            let _ = ui.hide();
            let _ = slint::quit_event_loop();
        });
    }

    {
        let weak = ui.as_weak();
        ui.on_drag_window(move || {
            let Some(ui) = weak.upgrade() else { return };
            ui.window().with_winit_window(|winit_window| {
                let _ = winit_window.drag_window();
            });
        });
    }

    {
        let weak = ui.as_weak();
        ui.window().on_winit_window_event(move |_, event| {
            if let slint::winit_030::winit::event::WindowEvent::ThemeChanged(_) = event {
                if let Some(ui) = weak.upgrade() {
                    ui.invoke_theme_changed();
                }
            }
            slint::winit_030::EventResult::Propagate
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_pointer_pressed(move |x, y| {
            let Some(ui) = weak.upgrade() else { return };
            let mut state = lock(&app);
            if let Some(sim) = state.sim.as_mut() {
                let on_arrow = sim.pointer_down(pointer(x, y));
                state.dragging = !on_arrow;
                ui.set_arrow_hovered(on_arrow);
            }
            state.dirty = true;
            drop(state);
            refresh(&ui, &app);
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_pointer_moved(move |x, y| {
            let Some(ui) = weak.upgrade() else { return };
            let mut state = lock(&app);
            let is_dragging = state.dragging;
            let Some(sim) = state.sim.as_mut() else {
                return;
            };
            if is_dragging {
                sim.pointer_move(pointer(x, y));
                state.dirty = true;
            } else {
                let prev = sim.hovered_arrow();
                sim.pointer_move(pointer(x, y));
                let now = sim.hovered_arrow();
                if prev != now {
                    let is_hovered = now.is_some();
                    ui.set_arrow_hovered(is_hovered);
                    state.dirty = true;
                    drop(state);
                    refresh(&ui, &app);
                    redraw(&ui, &app);
                }
            }
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_pointer_exited(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut state = lock(&app);
            let Some(sim) = state.sim.as_mut() else {
                return;
            };
            if sim.hovered_arrow().is_some() {
                sim.clear_hover();
                ui.set_arrow_hovered(false);
                state.dirty = true;
                drop(state);
                refresh(&ui, &app);
                redraw(&ui, &app);
            }
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_pointer_released(move |x, y| {
            let Some(ui) = weak.upgrade() else { return };
            let mut state = lock(&app);
            state.dragging = false;
            let turned = state
                .sim
                .as_mut()
                .map(|sim| sim.pointer_up(pointer(x, y)))
                .transpose();
            match turned {
                Ok(Some(Some((grip, dir)))) => {
                    let name = state.grip_name(grip);
                    let suffix = if dir > 0 { "" } else { "'" };
                    state.note(format!("{name}{suffix}"));
                    state.solved = Solvedness::No;
                    ui.set_solved(false);
                },
                Ok(_) => {},
                Err(e) => state.note(format!("turn failed: {e}")),
            }
            let is_hovered = state
                .sim
                .as_ref()
                .is_some_and(|s| s.hovered_arrow().is_some());
            ui.set_arrow_hovered(is_hovered);
            state.dirty = true;
            drop(state);
            refresh(&ui, &app);
        });
    }

    {
        let weak = ui.as_weak();
        let app = Arc::clone(app);
        ui.on_zoomed(move |delta| {
            let _ = weak.upgrade();
            let mut state = lock(&app);
            if let Some(sim) = state.sim.as_mut() {
                // A tenth of a turn of the wheel is a tenth closer, and the
                // range is clamped so the camera can neither enter the puzzle
                // nor lose it.
                let now = sim.camera().position.length();
                let want = (now - f64::from(delta) * 0.02).clamp(2.0, 60.0);
                sim.set_distance(want);
            }
            state.dirty = true;
        });
    }
}

/// Normalized device coordinates, as the ray caster wants them.
fn pointer(x: f32, y: f32) -> Pointer {
    Pointer {
        x: f64::from(x),
        y: f64::from(y),
    }
}

/// Start building the puzzle at `index`, off the UI thread.
fn begin_build(ui: &MainWindow, app: &Shared, index: usize) {
    let (recipe, label, generation) = {
        let mut state = lock(app);
        let Some(entry) = state.entries.get(index).copied() else {
            return;
        };
        state.generation += 1;
        state.sim = None;
        state.log.clear();
        (
            entry.recipe.to_string(),
            entry.name.to_string(),
            state.generation,
        )
    };
    ui.set_busy(true);
    ui.set_status(format!("Building {label}…").into());

    let weak = ui.as_weak();
    let app = Arc::clone(app);
    // `upgrade_in_event_loop` is the only safe way back: the puzzle must be
    // installed on the thread that will draw it.
    std::thread::spawn(move || {
        let built = Simulator::from_query(&recipe);
        let _ = weak.upgrade_in_event_loop(move |ui| {
            let mut state = lock(&app);
            if state.generation != generation {
                // The user picked something else while this was building.
                return;
            }
            match built {
                Ok(mut sim) => {
                    // A little of each angle shows the puzzle as a solid rather
                    // than as a flat silhouette.
                    sim.look_from(-28.0, 20.0, 12.0);
                    sim.options_mut().background = if ui.get_is_dark() {
                        [27, 30, 36, 255]
                    } else {
                        [228, 233, 240, 255]
                    };
                    sim.options_mut().draw_arrows = ui.get_show_arrows();
                    sim.options_mut().draw_edges = ui.get_show_edges();
                    // Named while the puzzle is still solved, which is the only
                    // time naming it is cheap.
                    state.names = sim.symbolic().ok();
                    state.solved = Solvedness::No;
                    state.sim = Some(sim);
                    state.settle_solved();
                    state.dirty = true;
                    state.note(format!("built {label}"));
                },
                Err(e) => {
                    state.note(format!("could not build {label}: {e}"));
                },
            }
            ui.set_busy(false);
            drop(state);
            refresh(&ui, &app);
            redraw(&ui, &app);
        });
    });
}

/// Put the current state into the window's text and flags.
fn refresh(ui: &MainWindow, app: &Shared) {
    let state = lock(app);
    let (pieces, grips, moves, history, hovered_grip) = match state.sim.as_ref() {
        Some(sim) => (
            sim.piece_count(),
            sim.grip_count(),
            sim.moves_made(),
            sim.history_len(),
            sim.hovered_arrow()
                .map(|h| (h / 2, if h % 2 == 0 { -1 } else { 1 })),
        ),
        None => (0, 0, 0, 0, None),
    };
    ui.set_piece_count(i32::try_from(pieces).unwrap_or(i32::MAX));
    ui.set_grip_count(i32::try_from(grips).unwrap_or(i32::MAX));
    ui.set_can_undo(history > 0);
    ui.set_move_log(state.log.join("\n").into());
    if !ui.get_busy() {
        if let Some((grip, dir)) = hovered_grip {
            let name = state.grip_name(grip);
            let suffix = if dir > 0 { "" } else { "'" };
            ui.set_status(format!("Turn {name}{suffix}, {moves} moves made").into());
        } else {
            ui.set_status(format!("{moves} moves made, {history} to undo").into());
        }
    }
    drop(state);
}

/// Draw one frame into the window.
fn redraw(ui: &MainWindow, app: &Shared) {
    let mut state = lock(app);
    let (width, height) = (state.width, state.height);
    let Some(sim) = state.sim.as_mut() else {
        ui.set_frame(Image::default());
        return;
    };
    let frame = sim.render(width, height);
    let mut buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    buffer.make_mut_bytes().copy_from_slice(frame.as_bytes());
    let solved = state.solved == Solvedness::Yes;
    drop(state);
    ui.set_frame(Image::from_rgba8(buffer));
    ui.set_solved(solved);
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    use slint::platform::{Platform, WindowAdapter};
    use slint::PlatformError;

    struct Headless {
        window: Rc<MinimalSoftwareWindow>,
    }

    impl Platform for Headless {
        fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, PlatformError> {
            Ok(self.window.clone())
        }
    }

    #[test]
    fn gui_hover_and_pointer_callbacks() {
        let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
        let _ = slint::platform::set_platform(Box::new(Headless {
            window: window.clone(),
        }));

        let entries = catalog::entries().collect::<Vec<_>>();
        let ui = MainWindow::new().expect("create window");
        let app: Shared = Arc::new(Mutex::new(App::new(entries)));

        wire_callbacks(&ui, &app);

        // Load 3x3x3 puzzle
        let recipe = catalog::find("Rubik's Cube (3x3x3)").unwrap().recipe;
        let mut sim = Simulator::from_query(recipe).expect("build 3x3x3");
        sim.look_from(-28.0, 20.0, 12.0);
        {
            let mut state = lock(&app);
            state.sim = Some(sim);
            state.width = 640;
            state.height = 640;
        }

        refresh(&ui, &app);
        assert!(!ui.get_arrow_hovered());
        assert!(ui.get_status().contains("0 moves made"));

        // Move over arrow 6
        ui.invoke_pointer_moved(0.050, 0.775);
        assert!(
            ui.get_arrow_hovered(),
            "hovering over arrow must set arrow_hovered true"
        );
        assert!(
            ui.get_status().starts_with("Turn "),
            "status must show target move when hovered: {}",
            ui.get_status()
        );

        // Move away to empty space
        ui.invoke_pointer_moved(0.0, 0.0);
        assert!(
            !ui.get_arrow_hovered(),
            "moving away must clear arrow_hovered"
        );
        assert!(
            ui.get_status().contains("0 moves made"),
            "status must restore when unhovered: {}",
            ui.get_status()
        );

        // Move over arrow 6 again
        ui.invoke_pointer_moved(0.050, 0.775);
        assert!(ui.get_arrow_hovered());

        // Pointer exits stage
        ui.invoke_pointer_exited();
        assert!(
            !ui.get_arrow_hovered(),
            "pointer exit must clear arrow_hovered"
        );

        // Toggle arrows off
        ui.set_show_arrows(false);
        ui.invoke_options_changed();
        ui.invoke_pointer_moved(0.050, 0.775);
        assert!(
            !ui.get_arrow_hovered(),
            "hidden arrows must not be hoverable"
        );

        // Window controls invocation
        assert!(!ui.get_is_maximized());
        ui.invoke_toggle_maximize();
        assert!(ui.get_is_maximized());
        ui.invoke_toggle_maximize();
        assert!(!ui.get_is_maximized());
        ui.invoke_minimize_window();
        ui.invoke_drag_window();

        // Animate scramble switch tests
        assert!(
            ui.get_animate_scramble(),
            "animate scramble must default to true"
        );
        // Test non-animated scramble (instant settle)
        ui.set_animate_scramble(false);
        ui.invoke_animate_scramble_changed();
        ui.set_scramble_depth(3.0);
        ui.invoke_scramble();
        assert_eq!(lock(&app).sim.as_ref().unwrap().moves_made(), 3);
        assert!(!lock(&app).sim.as_ref().unwrap().is_animating());

        // Test animated scramble
        ui.set_animate_scramble(true);
        ui.invoke_animate_scramble_changed();
        ui.set_scramble_depth(2.0);
        ui.invoke_scramble();
        assert!(lock(&app).sim.as_ref().unwrap().is_animating());

        // Test reset cancels scramble and restores
        ui.invoke_reset();
        assert_eq!(lock(&app).sim.as_ref().unwrap().history_len(), 0);
        assert!(lock(&app).sim.as_mut().unwrap().is_solved().unwrap());
        assert!(!lock(&app).sim.as_ref().unwrap().is_animating());
    }
}
