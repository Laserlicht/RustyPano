// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

// Release builds on Windows are GUI apps: no console window.
#![cfg_attr(all(windows, not(debug_assertions), not(test)), windows_subsystem = "windows")]

#[macro_use]
mod i18n;
mod inertia;

mod app;
mod fsv;
mod loader;
mod pyramid;
mod render;
mod settings;
mod source;
mod store;
mod tiles;
mod view;
#[cfg(target_arch = "wasm32")]
mod web;
mod xmp;

use std::sync::Arc;

use eframe::{egui_wgpu, wgpu};

/// wgpu setup shared by the native and the web build: ask for the adapter's
/// real texture limits, because panoramas are tiled into a texture array and
/// bigger tiles / more layers mean fewer downscales.
fn wgpu_setup() -> egui_wgpu::WgpuSetup {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| {
        let a = adapter.limits();
        let base = if adapter.get_info().backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits::default()
        };
        wgpu::DeviceDescriptor {
            label: Some("rustypano device"),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: a.max_texture_dimension_2d.min(16384),
                max_texture_array_layers: a.max_texture_array_layers,
                ..base
            },
            ..Default::default()
        }
    });
    egui_wgpu::WgpuSetup::CreateNew(setup)
}

#[cfg(not(target_arch = "wasm32"))]
fn usage() -> &'static str {
    tr!("Usage: rustypano [options] [panorama image | tour.fsv]

Options:
  -y <yaw>        initial yaw (0..360)
  -p <pitch>      initial pitch (-90..90)
  -hf <hfov>      initial horizontal field of view
  -fs             start in fullscreen
  -hson           show hotspots
  -left <x> -top <y> -width <w> -height <h>
                  window position and size (all four required)
  -h, --help      show this help")
}

#[cfg(not(target_arch = "wasm32"))]
struct Parsed {
    cli: app::CliOptions,
    geometry: Option<[f32; 4]>,
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_args() -> Result<Parsed, String> {
    let mut cli = app::CliOptions { file: None, yaw: None, pitch: None, hfov: None, fullscreen: false, hotspots_on: false };
    let mut geo: [Option<f32>; 4] = [None; 4];
    let mut args = std::env::args().skip(1).peekable();
    while let Some(a) = args.next() {
        let (key, inline) = match a.split_once('=') {
            Some((k, v)) if k.starts_with('-') => (k.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let key = key.trim_start_matches('-').to_lowercase();
        let mut value = |name: &str| -> Result<f64, String> {
            let v = inline.clone().or_else(|| args.next()).ok_or(tr!("{name} needs a value", name = name))?;
            v.parse::<f64>().map_err(|_| tr!("{name} expects a number", name = name))
        };
        if !a.starts_with('-') || a == "-" {
            cli.file = Some(source::Source::Path(std::path::PathBuf::from(&a)));
            continue;
        }
        match key.as_str() {
            "y" | "yaw" => {
                let v = value("-y")?;
                if !(0.0..=360.0).contains(&v) {
                    return Err(tr!("-y expects a yaw between 0 and 360").into());
                }
                cli.yaw = Some(v);
            }
            "p" | "pitch" => {
                let v = value("-p")?;
                if !(-90.0..=90.0).contains(&v) {
                    return Err(tr!("-p expects a pitch between -90 and 90").into());
                }
                cli.pitch = Some(v);
            }
            "hf" | "hfov" => {
                let v = value("-hf")?;
                if !(0.1..=view::MAX_HFOV).contains(&v) {
                    return Err(tr!("-hf expects a field of view between 0.1 and {max}", max = view::MAX_HFOV));
                }
                cli.hfov = Some(v);
            }
            "fs" => cli.fullscreen = true,
            "hson" => cli.hotspots_on = true,
            "left" => geo[0] = Some(value("-left")? as f32),
            "top" => geo[1] = Some(value("-top")? as f32),
            "width" => geo[2] = Some(value("-width")? as f32),
            "height" => geo[3] = Some(value("-height")? as f32),
            "h" | "help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("{}\n\n{}", tr!("Unknown option {option}", option = a), usage())),
        }
    }
    let geometry = match geo {
        [Some(l), Some(t), Some(w), Some(h)] => Some([l, t, w, h]),
        [None, None, None, None] => None,
        _ => return Err(tr!("Window geometry needs -left, -top, -width and -height together").into()),
    };
    if let Some(f) = cli.file.as_ref().and_then(|f| f.as_path())
        && !f.exists()
    {
        return Err(tr!("No such file: {path}", path = f.display()));
    }
    Ok(Parsed { cli, geometry })
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    use eframe::egui;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rustypano=info")).init();
    let settings = settings::Settings::load();
    i18n::apply(settings.language);
    store::start_session();
    let parsed = match parse_args() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let icon = image::load_from_memory(include_bytes!("../assets/icon256.png")).ok().map(|i| {
        let i = i.into_rgba8();
        egui::IconData { width: i.width(), height: i.height(), rgba: i.into_raw() }
    });
    let mut vp = egui::ViewportBuilder::default()
        .with_title("RustyPano")
        .with_app_id("rustypano")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([320.0, 240.0])
        .with_drag_and_drop(true);
    if let Some(icon) = icon {
        vp = vp.with_icon(Arc::new(icon));
    }
    if let Some([l, t, w, h]) = parsed.geometry {
        vp = vp.with_position([l, t]).with_inner_size([w, h]);
    }

    let options = eframe::NativeOptions {
        viewport: vp,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            surface: egui_wgpu::SurfaceConfig {
                present_mode: if settings.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
                ..egui_wgpu::SurfaceConfig::LOW_LATENCY
            },
            wgpu_setup: wgpu_setup(),
            ..Default::default()
        },
        ..Default::default()
    };
    let cli = parsed.cli;
    eframe::run_native("RustyPano", options, Box::new(move |cc| Ok(Box::new(app::RustyPanoApp::new(cc, cli)))))
}

/// Browser entry point: start the app on the `<canvas id="app">` of index.html
/// and remove the startup screen once it runs.
#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    let settings = settings::Settings::load();
    i18n::apply(settings.language);
    store::start_session();

    let mut cli = app::CliOptions { file: None, yaw: None, pitch: None, hfov: None, fullscreen: false, hotspots_on: false };
    for (k, v) in web::url_params() {
        match k.as_str() {
            "pano" if !v.is_empty() => cli.file = Some(source::Source::Url(v)),
            "yaw" => cli.yaw = v.parse().ok().filter(|v| (0.0..=360.0).contains(v)),
            "pitch" => cli.pitch = v.parse().ok().filter(|v| (-90.0..=90.0).contains(v)),
            "hfov" => cli.hfov = v.parse().ok().filter(|v| (0.1..=view::MAX_HFOV).contains(v)),
            "fs" => cli.fullscreen = v != "0",
            _ => {}
        }
    }

    let options = eframe::WebOptions {
        wgpu_options: egui_wgpu::WgpuConfiguration { wgpu_setup: wgpu_setup(), ..Default::default() },
        ..Default::default()
    };
    wasm_bindgen_futures::spawn_local(async move {
        let Some(canvas) = web::canvas() else {
            web::show_startup_error("RustyPano: missing <canvas id=\"app\">");
            return;
        };
        let result = eframe::WebRunner::new()
            .start(canvas, options, Box::new(move |cc| Ok(Box::new(app::RustyPanoApp::new(cc, cli)))))
            .await;
        match result {
            Ok(()) => web::hide_loading_screen(),
            Err(e) => {
                let detail = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                web::show_startup_error(&tr!("RustyPano could not start: {error}", error = detail));
            }
        }
    });
}
