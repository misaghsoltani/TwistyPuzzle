//! # twistypuzzle
//!
//! An exact-arithmetic twisty-puzzle simulator with an integrated software
//! renderer.
//!
//! Every move is computed in an algebraic number field, so the geometry is
//! exact: no cut ever drifts and no rotation ever accumulates round-off. A
//! face that should meet another meets it exactly, and a fifth of a turn is a
//! fifth of a turn.
//!
//! Deciding a comparison can *narrow* the isolating interval of the number
//! field the two values share, so the arithmetic carries mutable state that
//! every number in a field sees. Converting to `f64` refines until the answer
//! is the correctly rounded double, which makes the result independent of how
//! much narrowing came before it, and that is what keeps sort order, cache
//! lookups and evaluation order ordinary implementation details rather than
//! part of the contract. `SEMANTICS.md` lists the few things that are.

pub mod batch;
pub mod builder;
pub mod catalog;
mod catalog_data;
pub mod color;
mod color_tables;
pub mod error;
pub mod exact;
pub mod fdlibm;
pub mod make;
pub mod math;
pub mod movement;
pub mod num;
pub mod parse;
pub mod piece;
pub mod polyhedra;
mod polyhedra_data;
pub mod render;
pub mod simulator;
pub mod sort;
pub mod symbolic;

#[cfg(feature = "python")]
mod python;

pub use error::{Error, Result};
