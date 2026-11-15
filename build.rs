// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

fn main() {
    // Embed the application icon and version info into the Windows executable.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/rustypano.ico");
        res.set("FileDescription", "RustyPano");
        res.set("ProductName", "RustyPano");
        if let Err(e) = res.compile() {
            panic!("failed to embed Windows resources: {e}");
        }
    }
    println!("cargo:rerun-if-changed=assets/rustypano.ico");
    println!("cargo:rerun-if-changed=build.rs");
}
