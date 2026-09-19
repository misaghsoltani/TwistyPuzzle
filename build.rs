//! Tell the linker where to find `libpython`, and teach `cfg` about PyO3.
//!
//! A wheel does not link against the interpreter: maturin turns on
//! `pyo3/extension-module` and the symbols are resolved by whichever
//! interpreter loads the module. Other workflows, such as `cargo test`, `cargo run`
//! with examples, and `cargo bench`, do link against it, and without an rpath the
//! resulting binary cannot find the shared library at run time.
//!
//! `use_pyo3_cfgs` is what makes `#[cfg(Py_GIL_DISABLED)]` work in this crate,
//! which is how the free-threading-specific paths are selected.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "python")]
    {
        pyo3_build_config::use_pyo3_cfgs();
        pyo3_build_config::print_expected_cfgs();
        pyo3_build_config::add_extension_module_link_args();
        pyo3_build_config::add_libpython_rpath_link_args();
    }
}
