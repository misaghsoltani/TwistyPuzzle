//! Software rendering: a tiled, parallel rasterizer plus framebuffer blitting.
//!
//! Everything is drawn on the CPU, so the package stays importable anywhere,
//! with no GPU, display server or browser required, and stays deterministic,
//! which a GPU pipeline is not.
//!
//! The scene's lighting makes that cheap: it is purely ambient, with no
//! directional light, so every fragment's color is a fixed function of its
//! vertex color and no shading model is needed at all (`SEMANTICS.md` §7).
//! What is left is geometry, depth ordering and coverage.

pub mod camera;
pub mod font;
mod font_data;
pub mod framebuffer;
pub mod png;
pub mod raster;
pub mod scene;
pub mod trackball;

pub use framebuffer::{BlendMode, Framebuffer, Rect};
