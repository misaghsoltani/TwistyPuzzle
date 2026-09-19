//! Compile the interface description into Rust.

fn main() {
    println!("cargo:rerun-if-changed=ui/app.slint");
    println!("cargo:rerun-if-changed=ui/app-icon.svg");
    println!("cargo:rerun-if-changed=ui/app-icon.png");
    println!("cargo:rerun-if-changed=ui/logo.svg");
    println!("cargo:rerun-if-changed=ui/icon-moon.svg");
    println!("cargo:rerun-if-changed=ui/icon-sun.svg");
    println!("cargo:rerun-if-changed=ui/icon-both.svg");
    println!("cargo:rerun-if-changed=ui/icon-min.svg");
    println!("cargo:rerun-if-changed=ui/icon-max.svg");
    println!("cargo:rerun-if-changed=ui/icon-restore.svg");
    println!("cargo:rerun-if-changed=ui/icon-close.svg");
    slint_build::compile("ui/app.slint").expect("compiling ui/app.slint");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=src/dock_macos.m");
        let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
        if !out_dir.is_empty() {
            let obj = format!("{out_dir}/dock_macos.o");
            let lib = format!("{out_dir}/libdock_macos.a");
            let status = std::process::Command::new("clang")
                .args(["-c", "src/dock_macos.m", "-o", &obj])
                .status();
            if let Ok(s) = status {
                if s.success() {
                    let _ = std::process::Command::new("ar")
                        .args(["rcs", &lib, &obj])
                        .status();
                    println!("cargo:rustc-link-search=native={out_dir}");
                    println!("cargo:rustc-link-lib=static=dock_macos");
                    println!("cargo:rustc-link-lib=framework=AppKit");
                }
            }
        }
    }
}
