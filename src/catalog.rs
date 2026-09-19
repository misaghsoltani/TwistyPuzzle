//! The catalog of named puzzles.
//!
//! About eighty-five twisty puzzles, each expressed as a shell-and-cuts
//! recipe. Anything else expressible as a recipe builds just as well, and these
//! are the ones with names.
//!
//! Ten of them carry the placeholder name `"Unknown"`, so names are not
//! unique. Recipes are, which is why every function that takes a puzzle takes
//! a recipe as readily as a name.

pub use crate::catalog_data::CATALOG;

/// One cataloged puzzle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Common name, e.g. `"Rubik's Cube (3x3x3)"`.
    pub name: &'static str,
    /// Shell family, e.g. `"Cubes"`.
    pub family: &'static str,
    /// Turning style, e.g. `"Face-turning"`.
    pub kind: &'static str,
    /// Recipe query string, e.g. `"?shell=C$1&cut=C$1/3"`.
    pub recipe: &'static str,
}

/// Every cataloged puzzle, in catalog order.
pub fn entries() -> impl Iterator<Item = CatalogEntry> {
    CATALOG
        .iter()
        .map(|&(name, family, kind, recipe)| CatalogEntry {
            name,
            family,
            kind,
            recipe,
        })
}

/// Look up a puzzle by name, case-insensitively.
///
/// Names are not unique: ten entries are labeled `"Unknown"`. This returns the first match, so use
/// [`find_all`] when a name may be ambiguous, or build from the recipe.
pub fn find(name: &str) -> Option<CatalogEntry> {
    entries().find(|e| e.name.eq_ignore_ascii_case(name))
}

/// Every cataloged puzzle with the given name, case-insensitively.
///
/// Recipes *are* unique, so the result identifies the puzzles unambiguously
/// even when the name does not.
pub fn find_all(name: &str) -> impl Iterator<Item = CatalogEntry> + '_ {
    entries().filter(move |e| e.name.eq_ignore_ascii_case(name))
}

/// The number of cataloged puzzles.
pub fn len() -> usize {
    CATALOG.len()
}

/// Always false, and present so `len` reads naturally alongside it.
pub fn is_empty() -> bool {
    CATALOG.is_empty()
}
