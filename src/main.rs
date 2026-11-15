// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

mod app;
mod fsv;
mod loader;
mod render;
mod settings;
mod view;
mod xmp;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::{egui, egui_wgpu, wgpu};

const USAGE: &str = "\
Usage: rustypano [options] [file.jpg|file.tif|file.fsv]

Options:
  -y <yaw>        initial yaw (0..360)
  -p <pitch>      initial pitch (-90..90)
  -hf <hfov>      initial horizontal field of view
  -fs             start at full screen
  -hson           show hotspots
  -left <x> -top <y> -width <w> -height <h>
                  window position and size (all four required)
  -h, --help      show this help";

struct Parsed {
    cli: app::CliOptions,
    geometry: Option<[f32; 4]>,
}

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
            let v = inline.clone().or_else(|| args.next()).ok_or(format!("{name} needs a value"))?;
            v.parse::<f64>().map_err(|_| format!("{name} expects a number"))
        };
        if !a.starts_with('-') || a == "-" {
            cli.file = Some(PathBuf::from(&a));
            continue;
        }
        match key.as_str() {
            "y" | "yaw" => {
                let v = value("-y")?;
                if !(0.0..=360.0).contains(&v) {
                    return Err("-y expects a yaw between 0 and 360".into());
                }
                cli.yaw = Some(v);
            }
            "p" | "pitch" => {
                let v = value("-p")?;
                if !(-90.0..=90.0).contains(&v) {
                    return Err("-p expects a pitch between -90 and 90".into());
                }
                cli.pitch = Some(v);
            }
            "hf" | "hfov" => {
                let v = value("-hf")?;
                if !(0.1..=view::MAX_HFOV).contains(&v) {
                    return Err(format!("-hf expects a field of view between 0.1 and {}", view::MAX_HFOV));
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
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("Unknown option {a}\n\n{USAGE}")),
        }
    }
    let geometry = match geo {
        [Some(l), Some(t), Some(w), Some(h)] => Some([l, t, w, h]),
        [None, None, None, None] => None,
        _ => return Err("Window geometry needs -left, -top, -width and -height together".into()),
    };
    if let Some(f) = &cli.file
        && !f.exists() {
            return Err(format!("No such file: {}", f.display()));
        }
    Ok(Parsed { cli, geometry })
}

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,rustypano=info")).init();
    let parsed = match parse_args() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let settings = settings::Settings::load();

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

    // Ask for the adapter's real texture limits: panoramas are tiled into a
    // texture array, bigger tiles / more layers mean fewer downscales.
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
    let options = eframe::NativeOptions {
        viewport: vp,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            surface: egui_wgpu::SurfaceConfig {
                present_mode: if settings.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
                ..egui_wgpu::SurfaceConfig::LOW_LATENCY
            },
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup),
            ..Default::default()
        },
        ..Default::default()
    };
    let cli = parsed.cli;
    eframe::run_native("RustyPano", options, Box::new(move |cc| Ok(Box::new(app::RustyPanoApp::new(cc, cli)))))
}
