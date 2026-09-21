//! Software rendering: a tiled, parallel rasterizer plus framebuffer blitting.
//!
//! Everything is drawn on the CPU, so the package stays importable anywhere,
//! with no GPU, display server or browser required, and stays deterministic,
//! which a GPU pipeline is not.
//!
//! The scene's lighting model reduces per-pixel complexity: lighting is purely
//! ambient without directional sources, ensuring each fragment's color is a
//! deterministic function of its vertex color without per-pixel shading
//! calculations (`SEMANTICS.md` §7). Rendering reduces to geometric projection,
//! depth sorting, and pixel coverage.

pub mod camera;
pub mod font;
mod font_data;
pub mod framebuffer;
pub mod lut;
pub mod png;
pub mod raster;
pub mod scene;
pub mod trackball;

pub use framebuffer::{BlendMode, Framebuffer, Rect};
