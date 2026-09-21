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
//! the lock around the state is never contended, acting as a lock instead of a
//! `RefCell` only because a puzzle built on a worker has to travel back through
//! the event loop, which requires whatever it carries to be `Send`.
//!
//! Initial puzzle derivation is computationally intensive (complex catalog entries
//! require exact algebraic plane and polyhedron cuts), so construction executes
//! on a background worker thread and communicates completion via the UI event loop.
//! This ensures the window remains responsive and displays construction progress.
//!
//! Frames are rendered on demand when state changes instead of on an unconstrained
//! continuous loop, avoiding unnecessary computation when the puzzle is stationary.
//! During turn animations or trackball drag operations, a 60 Hz timer drives updates.

use core::fmt::Write as _;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use slint::winit_030::WinitWindowAccessor;
use slint::{Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel};
use twistypuzzle::catalog;
use twistypuzzle::render::scene::FrameScratch;
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
/// contended, acting as a `Mutex` instead of a `RefCell` only because the built
/// puzzle has to come back from a worker through Slint's event loop, which
/// requires whatever it carries to be `Send`.
type Shared = Arc<Mutex<App>>;

/// Take the lock, recovering from a poisoned one instead of compounding a
/// panic that already happened.
fn lock(app: &Shared) -> MutexGuard<'_, App> {
    app.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Everything the interface owns, on the one thread that owns it.
struct App {
    /// `None` only while the first puzzle is being built.
    sim: Option<Simulator>,
    entries: Vec<Entry>,
    /// What the command line asked the first puzzle to start out as, taken
    /// when that puzzle arrives. Later ones are the user's own choice and get
    /// none of it.
    start: Option<Options>,
    /// Physical pixels, so the frame is sharp on a scaled display.
    width: u32,
    height: u32,
    /// Bumped for every build request, so a build that finishes after the user
    /// has moved on is discarded instead of replacing what they chose.
    generation: u64,
    log: Vec<String>,
    /// True between a press and the release that ends it.
    dragging: bool,
    /// Set when the puzzle has changed and the frame has not caught up.
    dirty: bool,
    last_tick: Instant,
    /// The move names, so the log can say `A` instead of `layer 3`. Absent
    /// only if naming them failed.
    names: Option<Arc<SymbolicView>>,
    /// Worked out when the puzzle changes instead of once a frame: reading a
    /// state walks every sticker, and on a puzzle that jumbles it walks them
    /// only to fail.
    solved: Solvedness,
    /// The rasterizer's buffers, kept between frames instead of allocated and
    /// zeroed sixty times a second.
    scratch: FrameScratch,
    /// Two pixel buffers, used in turn. See [`redraw`].
    surfaces: [Option<SharedPixelBuffer<Rgba8Pixel>>; 2],
    /// Which of the two the next frame goes into.
    surface: usize,
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
    fn new(entries: Vec<Entry>, start: Options) -> App {
        App {
            sim: None,
            entries,
            start: Some(start),
            width: 640,
            height: 640,
            generation: 0,
            log: Vec::new(),
            dragging: false,
            dirty: true,
            last_tick: Instant::now(),
            names: None,
            solved: Solvedness::No,
            scratch: FrameScratch::new(),
            surfaces: [None, None],
            surface: 0,
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
        let named = self.names.as_ref().zip(self.sim.as_ref()).and_then(|(v, sim)| {
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

/// A puzzle the menu can offer: one of the cataloged ones, or the one the
/// command line named.
///
/// Owned instead of a `CatalogEntry`, whose fields are `&'static str`,
/// because a recipe given on the command line is not static and leaking it to
/// pretend otherwise would be a poor trade for one string.
#[derive(Clone)]
struct Entry {
    name: String,
    kind: String,
    recipe: String,
}

impl Entry {
    /// The line the menu shows. Ten catalog entries are called "Unknown", so
    /// those say what they are made of instead.
    fn label(&self) -> String {
        if self.name == "Unknown" {
            format!("{} ({})", self.recipe, self.kind)
        } else {
            format!("{} - {}", self.name, self.kind)
        }
    }
}

/// Everything the command line can ask the window to start out as.
#[derive(Clone)]
struct Options {
    /// A puzzle that is not in the catalog, added to the menu and opened.
    custom: Option<Entry>,
    /// A cataloged puzzle to open, by name.
    named: Option<String>,
    /// Random moves to make once it is built.
    scramble: Option<u32>,
    /// Seed for those moves, so a scramble can be repeated.
    seed: Option<u64>,
    /// A written sequence to apply once it is built, such as `"A B' C2"`.
    moves: Option<String>,
    /// Degrees to swing the camera sideways from head-on.
    yaw: f64,
    /// Degrees to raise it.
    pitch: f64,
    /// How far out the camera sits.
    distance: f64,
    arrows: bool,
    edges: bool,
    /// How far the Scramble button scrambles.
    depth: f32,
    animate: bool,
    /// 0 dark, 1 light, 2 follow the system.
    theme: i32,
    /// Window size in logical pixels, or the window's own default.
    size: Option<(u32, u32)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            custom: None,
            named: None,
            scramble: None,
            seed: None,
            moves: None,
            // A little of each angle shows the puzzle as a solid instead of
            // as a flat silhouette.
            yaw: -28.0,
            pitch: 20.0,
            distance: 12.0,
            arrows: true,
            edges: true,
            depth: 20.0,
            animate: true,
            theme: 2,
            size: None,
        }
    }
}

/// What the command line asked for.
enum Request {
    /// Open the window as these options describe.
    Open(Box<Options>),
    /// Print something and stop.
    Print(String),
    /// Complain and stop.
    Fail(String),
}

const USAGE: &str = "\
twistypuzzle-gui - a window for the twisty puzzle simulator

Usage: twistypuzzle-gui [OPTIONS] [PUZZLE]

Arguments:
  [PUZZLE]              A cataloged puzzle's name, or a recipe query string
                        such as '?shell=C$1&cut=C$1/3'. Anything beginning
                        with '?' is read as a recipe and everything else as a
                        name. Without one, the window opens on the 3x3x3.

Options:
  -p, --puzzle NAME     A cataloged puzzle, by name
  -r, --recipe QUERY    A recipe query string, even if it looks like a name
  -s, --scramble N      Make N random moves once the puzzle is built
      --seed N          Seed those moves, so the scramble can be repeated
  -m, --moves SEQUENCE  Apply a written sequence, such as \"A B' C2\"
      --depth N         How far the Scramble button scrambles (default 20)
      --no-animate      Scramble in one step instead of turn by turn
      --yaw DEGREES     Swing the camera sideways from head-on (default -28)
      --pitch DEGREES   Raise the camera above head-on (default 20)
      --distance UNITS  How far the camera sits from the puzzle (default 12)
      --no-arrows       Start with the turn arrows hidden
      --no-edges        Start with the piece outlines hidden
      --theme WHICH     'dark', 'light', or 'system' (the default)
      --size WxH        Window size in logical pixels, such as 1280x800
  -l, --list            Print every cataloged puzzle and stop
      --polyhedra       Print every polyhedron code and stop
  -V, --version         Print the version and stop
  -h, --help            Print this and stop
";

/// Read the command line.
///
/// Done by hand instead of with an argument parser: the whole grammar is the
/// list above, and a dependency that pulls in a derive macro to read it would
/// cost more to build than it saves to write.
#[allow(clippy::too_many_lines)]
fn parse_args(args: &[String]) -> Request {
    let mut opts = Options::default();
    let mut recipe: Option<String> = None;
    let mut positional: Option<String> = None;
    let mut rest = args.iter();

    /// Take an option's value, or say which option was left dangling.
    macro_rules! value {
        ($rest:expr, $flag:expr) => {
            match $rest.next() {
                Some(v) => v.clone(),
                None => return Request::Fail(format!("{} needs a value", $flag)),
            }
        };
    }

    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-h" | "--help" => return Request::Print(USAGE.into()),
            "-V" | "--version" => {
                return Request::Print(format!("twistypuzzle-gui {}\n", env!("CARGO_PKG_VERSION")));
            },
            "-l" | "--list" => return Request::Print(catalog_listing()),
            "--polyhedra" => return Request::Print(polyhedra_listing()),
            "-p" | "--puzzle" => positional = Some(value!(rest, "--puzzle")),
            "-r" | "--recipe" => recipe = Some(value!(rest, "--recipe")),
            "-m" | "--moves" => opts.moves = Some(value!(rest, "--moves")),
            "-s" | "--scramble" => {
                let v = value!(rest, "--scramble");
                match v.parse::<u32>() {
                    Ok(n) => opts.scramble = Some(n),
                    Err(_) => return Request::Fail(format!("--scramble wants a count, not {v:?}")),
                }
            },
            "--seed" => {
                let v = value!(rest, "--seed");
                match v.parse::<u64>() {
                    Ok(n) => opts.seed = Some(n),
                    Err(_) => return Request::Fail(format!("--seed wants a number, not {v:?}")),
                }
            },
            "--depth" => {
                let v = value!(rest, "--depth");
                match v.parse::<f32>() {
                    Ok(n) if n >= 1.0 => opts.depth = n,
                    _ => return Request::Fail(format!("--depth wants a count of 1 or more, not {v:?}")),
                }
            },
            "--no-animate" => opts.animate = false,
            "--no-arrows" => opts.arrows = false,
            "--no-edges" => opts.edges = false,
            "--yaw" | "--pitch" | "--distance" => {
                let flag = arg.clone();
                let v = value!(rest, flag);
                let Ok(n) = v.parse::<f64>() else {
                    return Request::Fail(format!("{flag} wants a number, not {v:?}"));
                };
                match flag.as_str() {
                    "--yaw" => opts.yaw = n,
                    "--pitch" => opts.pitch = n,
                    _ if n > 0.0 => opts.distance = n,
                    _ => return Request::Fail("--distance wants a positive number".into()),
                }
            },
            "--theme" => {
                let v = value!(rest, "--theme");
                opts.theme = match v.as_str() {
                    "dark" => 0,
                    "light" => 1,
                    "system" => 2,
                    _ => return Request::Fail(format!("--theme wants dark, light or system, not {v:?}")),
                };
            },
            "--size" => {
                let v = value!(rest, "--size");
                match parse_size(&v) {
                    Some(wh) => opts.size = Some(wh),
                    None => return Request::Fail(format!("--size wants WIDTHxHEIGHT, not {v:?}")),
                }
            },
            other if other.starts_with('-') && other != "-" => {
                return Request::Fail(format!("unknown option {other:?}, try --help"));
            },
            other => {
                if positional.is_some() {
                    return Request::Fail(format!("only one puzzle can be opened, and {other:?} is a second"));
                }
                positional = Some(other.to_string());
            },
        }
    }

    // A positional argument beginning with `?` is a recipe. Nothing in the
    // catalog starts with one, so this cannot shadow a name.
    if let Some(text) = positional {
        if text.starts_with('?') && recipe.is_none() {
            recipe = Some(text);
        } else {
            opts.named = Some(text);
        }
    }
    if let (Some(_), Some(name)) = (&recipe, &opts.named) {
        return Request::Fail(format!("--recipe and the puzzle {name:?} name two different puzzles"));
    }
    if let Some(query) = recipe {
        opts.custom = Some(Entry {
            name: query.clone(),
            kind: "from the command line".into(),
            recipe: query,
        });
    }
    Request::Open(Box::new(opts))
}

/// Read `WIDTHxHEIGHT`, in either case of `x`.
fn parse_size(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.split_once(['x', 'X'])?;
    let w: u32 = w.trim().parse().ok()?;
    let h: u32 = h.trim().parse().ok()?;
    (w >= 320 && h >= 240 && w <= 16384 && h <= 16384).then_some((w, h))
}

/// Every cataloged puzzle, one per line, with the recipe that identifies it.
fn catalog_listing() -> String {
    let mut out = String::new();
    for e in catalog::entries() {
        let _ = writeln!(out, "{}\t{}\t{}\t{}", e.name, e.family, e.kind, e.recipe);
    }
    out
}

/// Every polyhedron a recipe can name, one per line.
fn polyhedra_listing() -> String {
    let mut out = String::new();
    for (code, name) in twistypuzzle::polyhedra::shapes() {
        let _ = writeln!(out, "{code}\t{name}");
    }
    out
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

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    macos_dock::init_dock_icon();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Request::Open(o) => *o,
        Request::Print(text) => {
            print!("{text}");
            return Ok(());
        },
        Request::Fail(why) => {
            eprintln!("twistypuzzle-gui: {why}");
            eprintln!("try 'twistypuzzle-gui --help'");
            std::process::exit(2);
        },
    };

    // A puzzle named on the command line joins the menu, so it can be chosen
    // again after the user has looked at another one.
    let mut entries: Vec<Entry> = catalog::entries()
        .map(|e| Entry {
            name: e.name.to_string(),
            kind: e.kind.to_string(),
            recipe: e.recipe.to_string(),
        })
        .collect();
    let mut start = entries
        .iter()
        .position(|e| e.name == "Rubik's Cube (3x3x3)")
        .unwrap_or(0);
    if let Some(custom) = opts.custom.clone() {
        entries.push(custom);
        start = entries.len() - 1;
    } else if let Some(name) = &opts.named {
        let Some(i) = entries.iter().position(|e| e.name.eq_ignore_ascii_case(name)) else {
            eprintln!("twistypuzzle-gui: no cataloged puzzle is called {name:?}");
            eprintln!("try 'twistypuzzle-gui --list'");
            std::process::exit(2);
        };
        start = i;
    }

    let ui = MainWindow::new()?;
    if let Some((w, h)) = opts.size {
        ui.window().set_size(slint::LogicalSize::new(w as f32, h as f32));
    }

    let names: Vec<SharedString> = entries.iter().map(|e| SharedString::from(e.label())).collect();
    let model: Rc<VecModel<SharedString>> = Rc::new(VecModel::from(names));
    ui.set_puzzle_names(ModelRc::from(model.clone()));

    ui.set_puzzle_index(i32::try_from(start).unwrap_or(0));
    ui.set_show_arrows(opts.arrows);
    ui.set_show_edges(opts.edges);
    ui.set_scramble_depth(opts.depth);
    ui.set_animate_scramble(opts.animate);
    let system_dark = detect_system_is_dark(&ui);
    ui.invoke_set_system_theme(system_dark);
    if opts.theme != 2 {
        ui.invoke_apply_theme(opts.theme);
    }
    ui.set_hint(
        "Drag to turn the view, click an arrow to turn a layer, scroll to zoom.\n\
         Space scrambles, R resets, U undoes."
            .into(),
    );

    let app: Shared = Arc::new(Mutex::new(App::new(entries, opts)));

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
            if let slint::winit_030::winit::event::WindowEvent::ThemeChanged(theme) = event {
                if let Some(ui) = weak.upgrade() {
                    let is_dark = matches!(theme, slint::winit_030::winit::window::Theme::Dark);
                    ui.invoke_set_system_theme(is_dark);
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
            let turned = state.sim.as_mut().map(|sim| sim.pointer_up(pointer(x, y))).transpose();
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
            let is_hovered = state.sim.as_ref().is_some_and(|s| s.hovered_arrow().is_some());
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

/// Make whatever moves the command line asked for, on a puzzle that has just
/// been built.
///
/// A sequence is applied before a scramble, so moves names a position and
/// scramble walks away from it, which is the order the two read in.
fn opening_moves(state: &mut App, opts: &Options) {
    let Some(sim) = state.sim.as_mut() else {
        return;
    };
    if let Some(text) = &opts.moves {
        match sim.symbolic().and_then(|view| {
            let moves = twistypuzzle::symbolic::parse_moves(&view.actions, text)?;
            let mut made = 0usize;
            for m in moves {
                for _ in 0..m.repeat {
                    if sim.apply_action(m.action)? {
                        made += 1;
                    }
                }
            }
            Ok(made)
        }) {
            Ok(made) => state.note(format!("applied {made} moves")),
            Err(e) => state.note(format!("could not apply the moves: {e}")),
        }
    }
    let Some(sim) = state.sim.as_mut() else {
        return;
    };
    if let Some(depth) = opts.scramble.filter(|&d| d > 0) {
        if let Some(seed) = opts.seed {
            sim.seed(seed);
        }
        sim.scramble(depth);
        match sim.settle() {
            Ok(()) => state.note(format!("scrambled {depth}")),
            Err(e) => state.note(format!("scramble failed: {e}")),
        }
    }
}

/// Start building the puzzle at `index`, off the UI thread.
fn begin_build(ui: &MainWindow, app: &Shared, index: usize) {
    let (recipe, label, generation) = {
        let mut state = lock(app);
        let Some(entry) = state.entries.get(index) else {
            return;
        };
        let (recipe, label) = (entry.recipe.clone(), entry.name.clone());
        state.generation += 1;
        state.sim = None;
        state.log.clear();
        (recipe, label, state.generation)
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
                    // The command line only ever describes the first puzzle,
                    // so this is taken instead of read: whatever the user
                    // picks next is theirs, not the shell's.
                    let start = state.start.take();
                    let view = start.as_ref();
                    // A little of each angle shows the puzzle as a solid instead
                    // of as a flat silhouette.
                    sim.look_from(
                        view.map_or(-28.0, |o| o.yaw),
                        view.map_or(20.0, |o| o.pitch),
                        view.map_or(12.0, |o| o.distance),
                    );
                    sim.options_mut().background = if ui.get_is_dark() {
                        [27, 30, 36, 255]
                    } else {
                        [228, 233, 240, 255]
                    };
                    sim.options_mut().draw_arrows = ui.get_show_arrows();
                    sim.options_mut().draw_edges = ui.get_show_edges();
                    // Initialized while the puzzle is in its solved configuration
                    // to minimize permutation analysis overhead.
                    state.names = sim.symbolic().ok();
                    state.solved = Solvedness::No;
                    state.sim = Some(sim);
                    if let Some(o) = start {
                        opening_moves(&mut state, &o);
                    }
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
            sim.hovered_arrow().map(|h| (h / 2, if h % 2 == 0 { -1 } else { 1 })),
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
///
/// Eliminates per-frame heap allocations. The rasterizer's color and depth
/// scratch buffers reside in `App::scratch`: rendering at 1280x1280 with 2x
/// supersampling rasterizes 2560x2560 pixels (~50 MB), which would incur
/// substantial allocation overhead if reallocated at 60 Hz. The rasterizer writes
/// directly to the shared pixel buffer displayed by the window system, eliminating
/// redundant blit passes.
///
/// Double-buffering ping-pongs between two shared surfaces. Because the window
/// system retains a reference to the active frame, drawing into the active buffer
/// would trigger copy-on-write reallocation. Alternating buffers guarantees that
/// the target buffer is unreferenced and reused in place.
fn redraw(ui: &MainWindow, app: &Shared) {
    let mut state = lock(app);
    let (width, height) = (state.width, state.height);
    if state.sim.is_none() {
        ui.set_frame(Image::default());
        return;
    }
    let slot = state.surface;
    state.surface ^= 1;
    let mut buffer = match state.surfaces[slot].take() {
        Some(b) if b.width() == width && b.height() == height => b,
        _ => SharedPixelBuffer::<Rgba8Pixel>::new(width, height),
    };
    let App {
        sim: Some(sim),
        scratch,
        ..
    } = &mut *state
    else {
        return;
    };
    let drawn = sim.render_into(width, height, scratch, buffer.make_mut_bytes());
    state.surfaces[slot] = Some(buffer.clone());
    let solved = state.solved == Solvedness::Yes;
    drop(state);
    if drawn.is_ok() {
        ui.set_frame(Image::from_rgba8(buffer));
    }
    ui.set_solved(solved);
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

/// Detect whether the host operating system is configured for dark mode.
fn detect_system_is_dark(ui: &MainWindow) -> bool {
    let mut theme = None;
    ui.window().with_winit_window(|w| {
        theme = w.theme();
    });
    match theme {
        Some(slint::winit_030::winit::window::Theme::Dark) => true,
        Some(slint::winit_030::winit::window::Theme::Light) => false,
        None => platform_system_is_dark().unwrap_or(false),
    }
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

    /// Frames go into two surfaces in turn, and into the rasterizer's kept
    /// buffers. Both are reused, so either could serve a stale picture: turn
    /// the puzzle and redraw more times than there are surfaces, and require
    /// every frame to be the one the puzzle would draw fresh.
    fn frames_are_never_stale(ui: &MainWindow, app: &Shared) {
        let mut seen: Vec<*const u8> = Vec::new();
        for turn in 0..5 {
            {
                let mut state = lock(app);
                let sim = state.sim.as_mut().unwrap();
                sim.begin_move(turn % sim.grip_count(), 1).expect("turn");
                sim.settle().expect("settle");
            }
            redraw(ui, app);
            let state = lock(app);
            let (width, height) = (state.width, state.height);
            let fresh = state.sim.as_ref().unwrap().render(width, height);
            let drawn = state.surfaces[state.surface ^ 1]
                .as_ref()
                .expect("the surface just drawn into");
            assert_eq!(
                drawn.as_bytes(),
                fresh.as_bytes(),
                "frame {turn} must be the picture of the puzzle as it now stands"
            );
            seen.push(drawn.as_bytes().as_ptr());
        }
        // Two surfaces, alternating: the fifth frame is in the first one's
        // memory, which is the point of keeping them.
        assert_eq!(seen[0], seen[2], "surfaces must be reused, not reallocated");
        assert_eq!(seen[1], seen[3]);
        assert_ne!(seen[0], seen[1], "consecutive frames must not share memory");
    }

    #[test]
    fn gui_hover_and_pointer_callbacks() {
        let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
        let _ = slint::platform::set_platform(Box::new(Headless { window: window.clone() }));

        let entries: Vec<Entry> = catalog::entries()
            .map(|e| Entry {
                name: e.name.to_string(),
                kind: e.kind.to_string(),
                recipe: e.recipe.to_string(),
            })
            .collect();
        let ui = MainWindow::new().expect("create window");
        let app: Shared = Arc::new(Mutex::new(App::new(entries, Options::default())));

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
        assert!(!ui.get_arrow_hovered(), "moving away must clear arrow_hovered");
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
        assert!(!ui.get_arrow_hovered(), "pointer exit must clear arrow_hovered");

        // Toggle arrows off
        ui.set_show_arrows(false);
        ui.invoke_options_changed();
        ui.invoke_pointer_moved(0.050, 0.775);
        assert!(!ui.get_arrow_hovered(), "hidden arrows must not be hoverable");

        // Window controls invocation
        assert!(!ui.get_is_maximized());
        ui.invoke_toggle_maximize();
        assert!(ui.get_is_maximized());
        ui.invoke_toggle_maximize();
        assert!(!ui.get_is_maximized());
        ui.invoke_minimize_window();
        ui.invoke_drag_window();

        // Animate scramble switch tests
        assert!(ui.get_animate_scramble(), "animate scramble must default to true");
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

        frames_are_never_stale(&ui, &app);
    }

    #[test]
    fn gui_theme_switching_and_system_mode() {
        let window = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
        let _ = slint::platform::set_platform(Box::new(Headless { window }));

        let ui = MainWindow::new().expect("create window");
        assert_eq!(ui.get_theme_mode(), 2);

        // System theme defaults to light in our tests
        ui.invoke_set_system_theme(false);
        assert!(!ui.get_is_dark());

        // When system is dark, system mode evaluates to dark
        ui.invoke_set_system_theme(true);
        assert!(ui.get_is_dark());

        // When system is light again, system mode evaluates to light
        ui.invoke_set_system_theme(false);
        assert!(!ui.get_is_dark());

        // Explicit dark mode (0) stays dark regardless of system theme
        ui.invoke_apply_theme(0);
        assert_eq!(ui.get_theme_mode(), 0);
        assert!(ui.get_is_dark());
        ui.invoke_set_system_theme(false);
        assert!(ui.get_is_dark());

        // Explicit light mode (1) stays light regardless of system theme
        ui.invoke_apply_theme(1);
        assert_eq!(ui.get_theme_mode(), 1);
        assert!(!ui.get_is_dark());
        ui.invoke_set_system_theme(true);
        assert!(!ui.get_is_dark());

        // Switching back to system mode (2) restores the system theme
        ui.invoke_apply_theme(2);
        assert_eq!(ui.get_theme_mode(), 2);
        assert!(ui.get_is_dark());
        ui.invoke_set_system_theme(false);
        assert!(!ui.get_is_dark());
    }
}
