// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanningMode {
    /// Speed proportional to the distance from the click point (QuickTime VR style).
    AutoPanning,
    /// The image follows the mouse (Google Street View style).
    ClickAndDrag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolator {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl Interpolator {
    pub const ALL: [Interpolator; 4] =
        [Interpolator::Nearest, Interpolator::Bilinear, Interpolator::Bicubic, Interpolator::Lanczos3];
    pub fn label(self) -> &'static str {
        match self {
            Interpolator::Nearest => tr!("Nearest neighbour (blocky)"),
            Interpolator::Bilinear => tr!("Trilinear (smooth, quickest)"),
            Interpolator::Bicubic => tr!("Bicubic Catmull-Rom (sharp)"),
            Interpolator::Lanczos3 => tr!("Lanczos3 (sharpest)"),
        }
    }
    pub fn shader_id(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonitorProfile {
    Srgb,
    File,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GridSettings {
    pub show_rows: bool,
    pub show_columns: bool,
    /// Degrees between rows (pitch) / columns (yaw).
    pub rows_distance: f32,
    pub columns_distance: f32,
    /// Line width in screen pixels.
    pub line_width: f32,
    pub color: [u8; 4],
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            show_rows: true,
            show_columns: true,
            rows_distance: 10.0,
            columns_distance: 10.0,
            line_width: 1.5,
            color: [255, 40, 40, 200],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub language: crate::i18n::LanguageChoice,
    pub default_yaw: f64,
    pub default_pitch: f64,
    pub default_hfov: f64,
    pub show_yaw_pitch_in_status_bar: bool,
    pub show_hotspot_tooltips: bool,
    pub panning_mode: PanningMode,
    /// 1..10, higher = faster auto-panning.
    pub mouse_sensitivity: f64,
    pub interpolator_steady: Interpolator,
    pub interpolator_panning: Interpolator,
    pub enable_color_management: bool,
    pub monitor_profile: MonitorProfile,
    pub monitor_profile_file: String,
    pub grid: GridSettings,
    pub show_toolbar: bool,
    pub show_status_bar: bool,
    pub vsync: bool,
    /// Upper bound for GPU memory used by one panorama, in megabytes.
    pub max_gpu_memory_mb: u32,
    pub mru: Vec<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: crate::i18n::LanguageChoice::Auto,
            default_yaw: 180.0,
            default_pitch: 0.0,
            default_hfov: 70.0,
            show_yaw_pitch_in_status_bar: true,
            show_hotspot_tooltips: true,
            panning_mode: PanningMode::ClickAndDrag,
            mouse_sensitivity: 5.0,
            interpolator_steady: Interpolator::Lanczos3,
            interpolator_panning: Interpolator::Bilinear,
            enable_color_management: true,
            monitor_profile: MonitorProfile::Srgb,
            monitor_profile_file: String::new(),
            grid: GridSettings::default(),
            show_toolbar: true,
            show_status_bar: true,
            vsync: true,
            // Browsers give a page far less memory than a native process.
            max_gpu_memory_mb: if cfg!(target_arch = "wasm32") { 1024 } else { 3072 },
            mru: Vec::new(),
        }
    }
}

pub const MRU_LEN: usize = 9;
#[cfg(target_arch = "wasm32")]
const STORAGE_KEY: &str = "rustypano.settings";

impl Settings {
    #[cfg(not(target_arch = "wasm32"))]
    fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("rustypano").join("settings.toml"))
    }

    pub fn load() -> Settings {
        #[cfg(not(target_arch = "wasm32"))]
        let text = Self::path().and_then(|p| std::fs::read_to_string(p).ok());
        #[cfg(target_arch = "wasm32")]
        let text = crate::web::storage_get(STORAGE_KEY);
        text.and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save(&self) {
        let text = match toml::to_string_pretty(self) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Cannot serialize settings: {e}");
                return;
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(p) = Self::path() else { return };
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Err(e) = std::fs::write(&p, text) {
                log::warn!("Cannot save settings to {}: {e}", p.display());
            }
        }
        #[cfg(target_arch = "wasm32")]
        crate::web::storage_set(STORAGE_KEY, &text);
    }

    pub fn push_mru(&mut self, path: PathBuf) {
        let path = path.canonicalize().unwrap_or(path);
        self.mru.retain(|p| p != &path);
        self.mru.insert(0, path);
        self.mru.truncate(MRU_LEN);
    }
}
