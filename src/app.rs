// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use web_time::Instant;

use eframe::egui::{
    self, Align2, Color32, CursorIcon, Key, KeyboardShortcut, Modifiers, Pos2, Rect, RichText, Sense,
    Stroke, Vec2,
};
use eframe::{egui_wgpu, wgpu};

use crate::fsv::HotspotDef;
use crate::i18n::{self, LanguageChoice};
use crate::loader::{self, LoadJob, LoadedPano, Overrides};
use crate::render::{PanoCallback, PanoRenderer, Uniforms};
use crate::settings::{Interpolator, MonitorProfile, PanningMode, Settings};
use crate::source::Source;
use crate::tiles;
use crate::view::{MAX_HFOV, PanoGeometry, UserLimits, View};
use crate::xmp::{GPANO_TAGS, GPano};

const HOTSPOT_RADIUS: f32 = 12.0;

pub struct CliOptions {
    pub file: Option<Source>,
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub hfov: Option<f64>,
    pub fullscreen: bool,
    /// `-hson`: accepted for compatibility, hotspots are shown by default.
    #[allow(dead_code)]
    pub hotspots_on: bool,
}

struct Hotspot {
    yaw: f64,
    pitch: f64,
    def: HotspotDef,
}

struct Pano {
    loaded: LoadedPano,
    view: View,
    hotspots: Vec<Hotspot>,
    /// Degrees per 1/60 s.
    autospin: f64,
}

impl Pano {
    /// Local path of the displayed image, if it is a local file.
    fn image_path(&self) -> Option<&Path> {
        self.loaded.image_source().as_path()
    }
    fn file_name(&self) -> String {
        self.loaded.source.file_name()
    }
}

struct SpeedTest {
    t0: Instant,
    frames: u32,
    turned: f64,
}

struct MetadataDraft {
    tags: GPano,
    horizon: f64,
    note: Option<String>,
    dirty: bool,
}

struct ViewParamsDraft {
    yaw: f64,
    pitch: f64,
    hfov: f64,
    confirm_default: bool,
}

struct Message {
    title: String,
    text: String,
}

pub struct RustyPanoApp {
    settings: Settings,
    device: wgpu::Device,
    queue: wgpu::Queue,
    encode_srgb: bool,

    pano: Option<Pano>,
    job: Option<LoadJob>,

    show_hotspots: bool,
    show_grid: bool,
    fullscreen: bool,
    title: String,

    drag_origin: Option<Pos2>,
    /// Momentum after a quick drag (grab mode).
    inertia: crate::inertia::Inertia,
    last_motion: Instant,
    hovered_hotspot: Option<usize>,
    horizon_tool: bool,
    speed_test: Option<SpeedTest>,
    fps: f32,

    message: Option<Message>,
    options: Option<Settings>,
    options_tab: usize,
    view_params: Option<ViewParamsDraft>,
    metadata: Option<MetadataDraft>,
    hotspot_list: Option<Option<usize>>,
    about: bool,
    /// Debug aid: RUSTYPANO_SCREENSHOT=<png> saves one frame once the
    /// panorama is shown, then quits.
    screenshot: Option<(PathBuf, u32)>,
    /// Size of the viewer area in physical pixels, from the last frame.
    viewport_px: (f64, f64),
    /// Files read asynchronously (browser file picker / drag and drop) that
    /// should be opened on the next frame.
    incoming: Arc<Mutex<Option<Source>>>,
    /// Tiles for the current view are still being loaded.
    tiles_streaming: bool,
    /// "Clear" was pressed in the settings.
    clear_cache_requested: bool,
}

/// Of several files opened together, the one to show: the first tour file,
/// else the first supported image. All of them stay available to the tour.
#[cfg(target_arch = "wasm32")]
fn main_file(files: Vec<(String, Vec<u8>)>) -> Option<Source> {
    let group = Source::memory_group(files);
    group
        .iter()
        .find(|s| s.is_fsv())
        .or_else(|| group.iter().find(|s| loader::is_supported_ext(&s.extension())))
        .cloned()
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

impl RustyPanoApp {
    pub fn new(cc: &eframe::CreationContext<'_>, cli: CliOptions) -> Self {
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu backend required");
        let encode_srgb = PanoRenderer::install(rs);
        let settings = Settings::load();
        let mut app = Self {
            device: rs.device.clone(),
            queue: rs.queue.clone(),
            encode_srgb,
            pano: None,
            job: None,
            show_hotspots: true, // -hson is accepted but hotspots are on by default
            show_grid: false,
            fullscreen: false,
            title: "RustyPano".into(),
            drag_origin: None,
            inertia: Default::default(),
            last_motion: Instant::now(),
            hovered_hotspot: None,
            horizon_tool: false,
            speed_test: None,
            fps: 0.0,
            message: None,
            options: None,
            options_tab: 0,
            view_params: None,
            metadata: None,
            hotspot_list: None,
            about: false,
            screenshot: std::env::var_os("RUSTYPANO_SCREENSHOT").map(|p| (PathBuf::from(p), 0)),
            viewport_px: (1280.0, 720.0),
            incoming: Arc::new(Mutex::new(None)),
            tiles_streaming: false,
            clear_cache_requested: false,
            settings,
        };
        if let Ok(i) = std::env::var("RUSTYPANO_INTERP") {
            let it = match i.as_str() {
                "nearest" => Interpolator::Nearest,
                "bilinear" => Interpolator::Bilinear,
                "bicubic" => Interpolator::Bicubic,
                _ => Interpolator::Lanczos3,
            };
            app.settings.interpolator_steady = it;
        }
        if std::env::var_os("RUSTYPANO_GRID").is_some() {
            app.show_grid = true;
        }
        if cli.fullscreen {
            app.set_fullscreen(&cc.egui_ctx, true);
        }
        if let Some(f) = cli.file {
            let ov = Overrides { yaw: cli.yaw, pitch: cli.pitch, hfov: cli.hfov };
            app.open(&cc.egui_ctx, f, Some(ov));
        }
        app
    }

    // ------------------------------------------------------------------ files

    fn open(&mut self, ctx: &egui::Context, source: Source, overrides: Option<Overrides>) {
        if let Some(j) = self.job.take() {
            j.cancel.store(true, Ordering::Relaxed);
        }
        self.speed_test = None;
        self.job = Some(loader::start_load(
            source,
            &self.settings,
            self.device.clone(),
            self.queue.clone(),
            ctx.clone(),
            overrides,
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_dialog(&mut self, ctx: &egui::Context) {
        let mut d = rfd::FileDialog::new()
            .set_title(tr!("Choose a panorama"))
            .add_filter(tr!("Panoramas"), &[loader::IMAGE_EXTENSIONS, &["fsv"]].concat())
            .add_filter(tr!("Panorama tour files"), &["fsv"])
            .add_filter(tr!("All files"), &["*"]);
        if let Some(dir) = self.current_dir() {
            d = d.set_directory(dir);
        }
        if let Some(p) = d.pick_file() {
            self.open(ctx, Source::Path(p), None);
        }
    }

    /// In the browser, several files can be picked at once so that a tour
    /// file finds the images it refers to.
    #[cfg(target_arch = "wasm32")]
    fn open_dialog(&mut self, ctx: &egui::Context) {
        let dialog = rfd::AsyncFileDialog::new()
            .set_title(tr!("Choose a panorama"))
            .add_filter(tr!("Panoramas"), &[loader::IMAGE_EXTENSIONS, &["fsv"]].concat());
        let incoming = self.incoming.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let Some(handles) = dialog.pick_files().await else { return };
            let mut files = Vec::new();
            for h in handles {
                files.push((h.file_name(), h.read().await));
            }
            *incoming.lock().unwrap() = main_file(files);
            ctx.request_repaint();
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn current_dir(&self) -> Option<PathBuf> {
        self.pano
            .as_ref()
            .and_then(|p| p.image_path()?.parent().map(Path::to_path_buf))
            .or_else(|| self.settings.mru.first().and_then(|p| p.parent().map(Path::to_path_buf)))
    }

    /// Previous/next image in the same folder (local files only).
    fn folder_neighbour(&self, step: i32) -> Option<PathBuf> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let cur = self.pano.as_ref()?.image_path()?.to_path_buf();
        let dir = cur.parent()?;
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && loader::is_supported(p)
                    && !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("fsv"))
            })
            .collect();
        files.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
        let i = files.iter().position(|p| p == &cur)? as i32 + step;
        (0..files.len() as i32).contains(&i).then(|| files[i as usize].clone())
    }

    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        let res = match job.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(_) => Err(tr!("Loader thread crashed").into()),
        };
        let job = self.job.take().unwrap();
        match res {
            Ok(loaded) => self.install_pano(ctx, loaded, job.overrides),
            Err(e) if e == "cancelled" => {}
            Err(e) => {
                self.message = Some(Message {
                    title: tr!("Could not open").into(),
                    text: format!("{}\n\n{e}", job.name),
                })
            }
        }
    }

    fn install_pano(&mut self, ctx: &egui::Context, loaded: LoadedPano, ov: Option<Overrides>) {
        let p = &loaded.params;
        let geo = PanoGeometry::new(loaded.width, loaded.height, p.horizon_position);
        let mut view = View::new(geo);
        (view.vw, view.vh) = self.viewport_px;
        view.limits = UserLimits {
            min_yaw: p.min_yaw,
            max_yaw: p.max_yaw,
            min_pitch: p.min_pitch,
            max_pitch: p.max_pitch,
        };
        let ov = ov.unwrap_or(Overrides { yaw: None, pitch: None, hfov: None });
        let s = &self.settings;
        view.hfov = ov.hfov.or(p.hfov).unwrap_or(s.default_hfov);
        view.pitch = ov.pitch.or(p.pitch).unwrap_or(s.default_pitch);
        view.yaw = ov.yaw.or(p.yaw).unwrap_or(s.default_yaw);
        view.set_hfov(view.hfov);

        let hotspots = p
            .hotspots
            .iter()
            .map(|h| {
                let x = h.x / 100.0 * (geo.width - 1.0);
                let y = h.y / 100.0 * (geo.height - 1.0);
                let (yaw, pitch) = geo.pixel_to_angles(x, y);
                Hotspot { yaw, pitch, def: h.clone() }
            })
            .collect();
        let autospin = p.autospin.unwrap_or(0.0);
        self.title = match &p.window_title {
            Some(t) if !t.is_empty() => t.clone(),
            _ => format!("RustyPano - {}", loaded.source.file_name()),
        };
        #[cfg(not(target_arch = "wasm32"))]
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(self.title.clone()));
        #[cfg(target_arch = "wasm32")]
        {
            let _ = ctx;
            crate::web::set_title(&self.title);
        }
        if let Some(path) = loaded.source.as_path() {
            self.settings.push_mru(path.to_path_buf());
            self.settings.save();
        }
        self.metadata = None;
        self.hotspot_list = None;
        self.inertia.stop();
        self.pano = Some(Pano { loaded, view, hotspots, autospin });
    }

    fn reload(&mut self, ctx: &egui::Context) {
        if let Some(p) = &self.pano {
            let ov = Overrides { yaw: Some(p.view.yaw), pitch: Some(p.view.pitch), hfov: Some(p.view.hfov) };
            let src = p.loaded.source.clone();
            self.open(ctx, src, Some(ov));
        }
    }

    /// Delete all cached tiles, then rebuild the current panorama's tiles.
    fn clear_tile_cache(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(crate::store::clear_all());
            self.reload(ctx);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let ctx2 = ctx.clone();
            let reload = self.incoming.clone();
            let current = self.pano.as_ref().map(|p| p.loaded.source.clone());
            wasm_bindgen_futures::spawn_local(async move {
                crate::store::clear_all().await;
                *reload.lock().unwrap() = current;
                ctx2.request_repaint();
            });
        }
    }

    fn set_fullscreen(&mut self, ctx: &egui::Context, on: bool) {
        self.fullscreen = on;
        #[cfg(not(target_arch = "wasm32"))]
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(on));
        #[cfg(target_arch = "wasm32")]
        {
            let _ = ctx;
            crate::web::set_fullscreen(on);
        }
    }

    // ------------------------------------------------------------------ input

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        let sc = |m, k| KeyboardShortcut::new(m, k);
        // Single-letter shortcuts must not steal keys from focused text fields.
        let typing = ctx.egui_wants_keyboard_input();
        let (open, quit, prev, next, fs, esc, home, grid, hs) = ctx.input_mut(|i| {
            (
                i.consume_shortcut(&sc(Modifiers::COMMAND, Key::O)),
                i.consume_shortcut(&sc(Modifiers::COMMAND, Key::Q)),
                i.consume_shortcut(&sc(Modifiers::ALT, Key::ArrowLeft)),
                i.consume_shortcut(&sc(Modifiers::ALT, Key::ArrowRight)),
                i.consume_shortcut(&sc(Modifiers::NONE, Key::F11)),
                i.key_pressed(Key::Escape),
                !typing && i.consume_shortcut(&sc(Modifiers::NONE, Key::Home)),
                !typing && i.consume_shortcut(&sc(Modifiers::NONE, Key::G)),
                !typing && i.consume_shortcut(&sc(Modifiers::NONE, Key::H)),
            )
        });
        if open {
            self.open_dialog(ctx);
        }
        if quit && !cfg!(target_arch = "wasm32") {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if prev {
            self.go_neighbour(ctx, -1);
        }
        if next {
            self.go_neighbour(ctx, 1);
        }
        if fs {
            self.set_fullscreen(ctx, !self.fullscreen);
        }
        if esc {
            if self.horizon_tool {
                self.horizon_tool = false;
            } else if self.fullscreen {
                self.set_fullscreen(ctx, false);
            }
        }
        if home {
            self.reset_view();
        }
        if grid {
            self.show_grid = !self.show_grid;
        }
        if hs {
            self.show_hotspots = !self.show_hotspots;
        }
    }

    fn go_neighbour(&mut self, ctx: &egui::Context, step: i32) {
        if let Some(p) = self.folder_neighbour(step) {
            self.open(ctx, Source::Path(p), None);
        }
    }

    fn reset_view(&mut self) {
        let s = &self.settings;
        if let Some(p) = &mut self.pano {
            let prm = &p.loaded.params;
            let v = &mut p.view;
            v.hfov = prm.hfov.unwrap_or(s.default_hfov);
            v.pitch = prm.pitch.unwrap_or(s.default_pitch);
            v.yaw = prm.yaw.unwrap_or(s.default_yaw);
            v.set_hfov(v.hfov);
        }
    }

    /// Keyboard navigation; returns true if the view moved.
    fn keyboard_navigation(&mut self, ctx: &egui::Context, dt: f64) -> bool {
        if ctx.egui_wants_keyboard_input() {
            return false;
        }
        let Some(p) = &mut self.pano else { return false };
        let (l, r, u, d, zin, zout, mods, any) = ctx.input(|i| {
            (
                i.key_down(Key::ArrowLeft),
                i.key_down(Key::ArrowRight),
                i.key_down(Key::ArrowUp),
                i.key_down(Key::ArrowDown),
                i.key_down(Key::A) || i.key_down(Key::Plus) || i.key_down(Key::Equals),
                i.key_down(Key::Z) || i.key_down(Key::Minus),
                i.modifiers,
                i.keys_down.len(),
            )
        });
        if any > 0 && p.autospin != 0.0 {
            p.autospin = 0.0;
        }
        if any > 0 {
            self.inertia.stop();
        }
        let v = &mut p.view;
        let mut moved = false;
        let pan = 1.2 * dt; // fraction of the FOV per second
        if !mods.alt {
            if l {
                moved |= v.add_yaw(-v.hfov * pan);
            }
            if r {
                moved |= v.add_yaw(v.hfov * pan);
            }
            if u {
                moved |= v.add_pitch(v.vfov() * pan);
            }
            if d {
                moved |= v.add_pitch(-v.vfov() * pan);
            }
        }
        let zf = 1.03f64.powf(dt * 45.0);
        let shift_only = mods.shift && !mods.ctrl && !mods.alt;
        let ctrl_only = mods.ctrl && !mods.shift && !mods.alt;
        if zin || shift_only {
            moved |= v.zoom(1.0 / zf);
        }
        if zout || ctrl_only {
            moved |= v.zoom(zf);
        }
        moved
    }

    // ------------------------------------------------------------------ viewer

    fn viewer(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let ppp = ctx.pixels_per_point() as f64;
        let rect = ui.max_rect();
        self.viewport_px = (rect.width() as f64 * ppp, rect.height() as f64 * ppp);
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let dt = ctx.input(|i| i.stable_dt).clamp(0.001, 0.1) as f64;
        if dt > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt as f32) * 0.1;
        }

        // Drag & drop.
        self.handle_dropped_files(&ctx);
        let incoming = self.incoming.lock().unwrap().take();
        if let Some(src) = incoming {
            self.open(&ctx, src, None);
        }

        if self.pano.is_none() {
            painter.rect_filled(rect, 0.0, Color32::from_gray(24));
            let msg = if self.job.is_some() {
                tr!("Loading…")
            } else {
                tr!("Open a panorama with Ctrl+O, or drop an image or .fsv file here")
            };
            painter.text(rect.center(), Align2::CENTER_CENTER, msg, egui::FontId::proportional(18.0), Color32::GRAY);
            if response.double_clicked() && self.job.is_none() {
                self.open_dialog(&ctx);
            }
            self.loading_overlay(&painter, rect);
            return;
        }

        let mut moved = self.keyboard_navigation(&ctx, dt);
        let settings = &self.settings;
        let pano = self.pano.as_mut().unwrap();
        pano.view.set_viewport(rect.width() as f64 * ppp, rect.height() as f64 * ppp);
        let to_px = |pos: Pos2| ((pos.x - rect.min.x) as f64 * ppp, (pos.y - rect.min.y) as f64 * ppp);

        // Mouse wheel zoom.
        if response.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y + i.smooth_scroll_delta.x * 0.0) as f64;
            if scroll != 0.0 {
                moved |= pano.view.zoom(1.1f64.powf(-scroll / 40.0));
            }
        }

        // Hotspot hover.
        let hover = response.hover_pos();
        self.hovered_hotspot = None;
        if self.show_hotspots && self.drag_origin.is_none()
            && let Some(h) = hover {
                let (hx, hy) = to_px(h);
                for (i, hs) in pano.hotspots.iter().enumerate() {
                    if let Some((sx, sy)) = pano.view.angles_to_screen(hs.yaw, hs.pitch) {
                        let r = HOTSPOT_RADIUS as f64 * ppp;
                        if (sx - hx).hypot(sy - hy) <= r {
                            self.hovered_hotspot = Some(i);
                        }
                    }
                }
            }

        // Clicks.
        let mut open_target: Option<(Result<Source, String>, Overrides)> = None;
        if response.clicked() {
            if self.horizon_tool {
                if let Some(pos) = response.interact_pointer_pos() {
                    let (x, y) = to_px(pos);
                    let (_, pitch) = pano.view.screen_to_angles(x, y);
                    let (_, row) = pano.view.geo.angles_to_pixel(0.0, pitch);
                    let hp = (row / pano.view.geo.height * 100.0).round();
                    self.horizon_tool = false;
                    self.message = Some(Message {
                        title: tr!("Horizon Finder").into(),
                        text: if (0.0..=100.0).contains(&hp) {
                            tr!(
                                "Horizon detected at {hp}% of the image height.\n\n\
                                 Use HORIZONPOSITION={hp} in a .fsv file, or save it into the image \
                                 via Tools → Photo Sphere Tags.",
                                hp = hp
                            )
                        } else {
                            tr!("The clicked point is outside the image.").into()
                        },
                    });
                }
            } else if let Some(i) = self.hovered_hotspot {
                let d = &pano.hotspots[i].def;
                open_target = Some((
                    pano.loaded.source.resolve(&d.target),
                    Overrides { yaw: d.initial_yaw, pitch: d.initial_pitch, hfov: d.initial_hfov },
                ));
            }
        }

        // Panning.
        let now = ctx.input(|i| i.time);
        if response.drag_started() {
            pano.autospin = 0.0;
            self.drag_origin = response.interact_pointer_pos();
            self.inertia.start(now);
        }
        if response.is_pointer_button_down_on() && !response.dragged() {
            // Pressing the button (or a finger) catches a gliding view.
            pano.autospin = 0.0;
            self.inertia.stop();
        }
        if response.dragged() {
            let v = &mut pano.view;
            match settings.panning_mode {
                PanningMode::ClickAndDrag => {
                    let d = response.drag_delta();
                    if d != Vec2::ZERO {
                        let (dx, dy) = (d.x as f64 * ppp, d.y as f64 * ppp);
                        let degpp = v.hfov / v.vw;
                        let vdegpp = v.vfov() / v.vh;
                        moved |= v.add_yaw(-dx * degpp);
                        moved |= v.add_pitch(dy * vdegpp);
                        self.inertia.drag(now, [dx, dy]);
                    }
                }
                PanningMode::AutoPanning => {
                    if let (Some(o), Some(c)) = (self.drag_origin, response.interact_pointer_pos()) {
                        let dx = (c.x - o.x) as f64 * ppp;
                        let dy = (c.y - o.y) as f64 * ppp;
                        let k = 0.6 * settings.mouse_sensitivity * dt;
                        // Axis snapping: ignore the minor axis
                        // when it is under 1/5 of the major one.
                        if dx.abs() > (dy / 5.0).abs() {
                            moved |= v.add_yaw(v.hfov * dx / v.vw * k);
                        }
                        if dy.abs() > (dx / 5.0).abs() {
                            moved |= v.add_pitch(-v.vfov() * dy / v.vh * k);
                        }
                    }
                    ctx.request_repaint();
                }
            }
        }
        if response.drag_stopped() {
            self.drag_origin = None;
            if settings.smooth_panning && settings.panning_mode == PanningMode::ClickAndDrag {
                self.inertia.release(now);
            }
        }
        // Glide on after a quick drag.
        if !response.dragged()
            && let Some([dx, dy]) = self.inertia.step(now)
        {
            let v = &mut pano.view;
            let (degpp, vdegpp) = (v.hfov / v.vw, v.vfov() / v.vh);
            // Stop the part of the movement that hits a limit of the view.
            if dx != 0.0 && !v.add_yaw(-dx * degpp) {
                self.inertia.stop_axis(0);
            }
            if dy != 0.0 && !v.add_pitch(dy * vdegpp) {
                self.inertia.stop_axis(1);
            }
            moved = true;
            ctx.request_repaint();
        }

        // Auto spin / speed test.
        if let Some(st) = &mut self.speed_test {
            let step = 2.0;
            pano.view.add_yaw(step);
            st.turned += step;
            st.frames += 1;
            moved = true;
            if st.turned >= 360.0 {
                let secs = st.t0.elapsed().as_secs_f64();
                let fps = st.frames as f64 / secs;
                self.message = Some(Message {
                    title: tr!("Benchmark").into(),
                    text: tr!(
                        "{fps} fps\n({frames} frames rendered in {secs} s{note})",
                        fps = format!("{fps:.1}"),
                        frames = st.frames,
                        secs = format!("{secs:.2}"),
                        note = if settings.vsync { tr!(", limited by vsync") } else { "" }
                    ),
                });
                self.speed_test = None;
            }
        } else if pano.autospin != 0.0 && self.drag_origin.is_none() {
            if !pano.view.add_yaw(pano.autospin * dt * 60.0) {
                pano.autospin = -pano.autospin;
            }
            moved = true;
        }

        if moved {
            self.last_motion = Instant::now();
            ctx.request_repaint();
        }
        let animating = self.last_motion.elapsed().as_secs_f32() < 0.25;
        if animating {
            ctx.request_repaint_after(std::time::Duration::from_millis(260));
        }

        // Cursor.
        let cursor = if self.horizon_tool {
            CursorIcon::Crosshair
        } else if self.hovered_hotspot.is_some() {
            CursorIcon::PointingHand
        } else if response.dragged() {
            match settings.panning_mode {
                PanningMode::ClickAndDrag => CursorIcon::Grabbing,
                PanningMode::AutoPanning => CursorIcon::Move,
            }
        } else if response.hovered() {
            match settings.panning_mode {
                PanningMode::ClickAndDrag => CursorIcon::Grab,
                PanningMode::AutoPanning => CursorIcon::Crosshair,
            }
        } else {
            CursorIcon::Default
        };
        if response.hovered() || response.dragged() {
            ctx.set_cursor_icon(cursor);
        }

        // Stream the tiles this view needs into the GPU cache.
        self.tiles_streaming = pano.loaded.tiles.update(&pano.view, &self.queue);
        if self.tiles_streaming {
            ctx.request_repaint();
        }

        // GPU draw.
        let v = &pano.view;
        let g = &pano.loaded.tiles.gpu;
        let interp = if animating || response.dragged() || self.speed_test.is_some() {
            settings.interpolator_panning
        } else {
            settings.interpolator_steady
        };
        let gs = &settings.grid;
        let grid_flags = if self.show_grid { (gs.show_rows as u32) | ((gs.show_columns as u32) << 1) } else { 0 };
        let bg = if self.encode_srgb {
            [0.06, 0.06, 0.06, 1.0]
        } else {
            [srgb_to_linear(15), srgb_to_linear(15), srgb_to_linear(15), 1.0]
        };
        let mut uniforms = Uniforms {
            vp: [
                (rect.min.x as f64 * ppp).round() as f32,
                (rect.min.y as f64 * ppp).round() as f32,
                v.vw as f32,
                v.vh as f32,
            ],
            img: [g.width as f32, g.height as f32, v.geo.horizon_row as f32, v.focal() as f32],
            cam: [
                v.yaw.to_radians() as f32,
                v.pitch.to_radians() as f32,
                tiles::GUTTER as f32,
                (g.grid.mip_levels() - 1) as f32,
            ],
            tile: [g.grid.cx as f32, g.grid.cy as f32, g.grid.sx as f32, g.grid.sy as f32],
            flags: [interp.shader_id(), self.encode_srgb as u32, grid_flags, g.grid.base() as u32],
            grid: [gs.rows_distance.max(0.1), gs.columns_distance.max(0.1), gs.line_width, 0.0],
            grid_color: [
                srgb_to_linear(gs.color[0]),
                srgb_to_linear(gs.color[1]),
                srgb_to_linear(gs.color[2]),
                gs.color[3] as f32 / 255.0,
            ],
            bg,
            ..Default::default()
        };
        uniforms.set_levels(g);
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            PanoCallback { uniforms, pano: Some(Arc::clone(g)) },
        ));

        // Hotspot markers.
        if self.show_hotspots {
            for (i, hs) in pano.hotspots.iter().enumerate() {
                if let Some((sx, sy)) = v.angles_to_screen(hs.yaw, hs.pitch) {
                    let c = Pos2::new(rect.min.x + (sx / ppp) as f32, rect.min.y + (sy / ppp) as f32);
                    if !rect.expand(HOTSPOT_RADIUS).contains(c) {
                        continue;
                    }
                    let hot = self.hovered_hotspot == Some(i);
                    let fill = if hot {
                        Color32::from_rgba_unmultiplied(255, 200, 40, 140)
                    } else {
                        Color32::from_rgba_unmultiplied(255, 255, 255, 60)
                    };
                    painter.circle(c, HOTSPOT_RADIUS, fill, Stroke::new(2.0, Color32::WHITE));
                    painter.circle_filled(c, 3.0, Color32::WHITE);
                }
            }
            if let (Some(i), Some(h)) = (self.hovered_hotspot, hover) {
                let d = &pano.hotspots[i].def.description;
                if settings.show_hotspot_tooltips && !d.is_empty() {
                    let galley = painter.layout_no_wrap(d.clone(), egui::FontId::proportional(14.0), Color32::WHITE);
                    let pos = h + Vec2::new(14.0, 18.0);
                    let r = Rect::from_min_size(pos, galley.size()).expand(5.0);
                    painter.rect_filled(r, 4.0, Color32::from_black_alpha(200));
                    painter.galley(pos, galley, Color32::WHITE);
                }
            }
        }

        if self.horizon_tool {
            let t = tr!("Click on the horizon (Esc to cancel)");
            let pos = rect.center_top() + Vec2::new(0.0, 24.0);
            let galley = painter.layout_no_wrap(t.into(), egui::FontId::proportional(16.0), Color32::WHITE);
            let r = Rect::from_center_size(pos, galley.size()).expand(8.0);
            painter.rect_filled(r, 6.0, Color32::from_black_alpha(190));
            painter.galley(r.min + Vec2::splat(8.0), galley, Color32::WHITE);
        }

        self.loading_overlay(&painter, rect);

        match open_target {
            Some((Ok(target), ov)) => self.open(&ctx, target, Some(ov)),
            Some((Err(e), _)) => self.message = Some(Message { title: tr!("Could not open").into(), text: e }),
            None => {}
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect()
        });
        if let Some(f) = dropped.into_iter().next() {
            self.open(ctx, Source::Path(f), None);
        }
    }

    /// Browsers hand over file contents asynchronously; all files dropped at
    /// once are kept together so a tour finds its images.
    #[cfg(target_arch = "wasm32")]
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        let incoming = self.incoming.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let mut files = Vec::new();
            for f in &dropped {
                if let Ok(bytes) = f.bytes_async().await {
                    files.push((f.path().to_string_lossy().into_owned(), bytes));
                }
            }
            *incoming.lock().unwrap() = main_file(files);
            ctx.request_repaint();
        });
    }

    /// Overlay while files are dragged over the window.
    fn drop_hint(&self, ctx: &egui::Context) {
        // (name if known, MIME type if known)
        let mut items: Vec<(Option<String>, String)> = ctx.input(|i| {
            i.raw
                .hovered_files
                .iter()
                .map(|f| (f.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()), f.mime.clone()))
                .collect()
        });
        // Debug aid for screenshots: RUSTYPANO_DROP_HINT=a.jpg,b.fsv
        if items.is_empty()
            && let Ok(names) = std::env::var("RUSTYPANO_DROP_HINT")
        {
            items = names.split(',').map(|n| (Some(n.to_string()), String::new())).collect();
        }
        if items.is_empty() {
            return;
        }

        let ext = |n: &str| n.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default();
        let known = |(name, mime): &(Option<String>, String)| -> Option<bool> {
            match name {
                Some(n) => Some(loader::is_supported_ext(&ext(n))),
                None if mime.starts_with("image/") => Some(true),
                None if !mime.is_empty() && mime != "application/octet-stream" => Some(false),
                None => None, // browsers often reveal nothing while dragging
            }
        };
        let states: Vec<Option<bool>> = items.iter().map(known).collect();
        let unsupported = !states.is_empty() && states.iter().all(|s| *s == Some(false));
        let tour = items.iter().any(|(n, _)| n.as_deref().is_some_and(|n| ext(n) == "fsv"));

        let screen = ctx.content_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop-hint")));
        painter.rect_filled(screen, 0.0, Color32::from_black_alpha(175));

        let t = ctx.input(|i| i.time) as f32;
        let pulse = 0.75 + 0.25 * (t * 3.0).sin();
        let accent = if unsupported { Color32::from_rgb(235, 90, 80) } else { Color32::from_rgb(90, 160, 255) };
        let stroke = Stroke::new(3.0, accent.gamma_multiply(pulse));
        let frame = screen.shrink(24.0);
        let c = [frame.left_top(), frame.right_top(), frame.right_bottom(), frame.left_bottom(), frame.left_top()];
        painter.extend(egui::Shape::dashed_line(&c, stroke, 14.0, 10.0));

        let title = if unsupported {
            tr!("This file type cannot be opened")
        } else if tour {
            tr!("Drop to open the tour")
        } else {
            tr!("Drop to open the panorama")
        };
        let names: Vec<&str> = items.iter().filter_map(|(n, _)| n.as_deref()).collect();
        let files = if names.is_empty() {
            tr!("{n} file(s)", n = items.len())
        } else if names.len() <= 3 {
            names.join(", ")
        } else {
            format!("{}, {}", names[..3].join(", "), tr!("and {n} more", n = names.len() - 3))
        };
        let hint = if cfg!(target_arch = "wasm32") {
            tr!("Drop a tour together with its images.")
        } else {
            tr!("JPEG, TIFF, PNG, WebP, BMP, JPEG XL or .fsv tour")
        };
        let title = painter.layout_no_wrap(title.into(), egui::FontId::proportional(26.0), Color32::WHITE);
        let files = painter.layout_no_wrap(files, egui::FontId::proportional(16.0), Color32::from_gray(225));
        let hint = painter.layout_no_wrap(hint.into(), egui::FontId::proportional(14.0), Color32::from_gray(165));

        // Card behind icon and text, so the panorama does not compete with it.
        let width = title.size().x.max(files.size().x).max(hint.size().x).min(screen.width() - 80.0) + 64.0;
        let card = Rect::from_center_size(screen.center(), Vec2::new(width.max(320.0), 230.0));
        painter.rect(card, 14.0, Color32::from_rgba_unmultiplied(28, 28, 32, 240), Stroke::new(1.0, accent.gamma_multiply(0.5)), egui::StrokeKind::Inside);

        // Tray with an arrow pointing into it (a cross if unsupported).
        let icon = card.center_top() + Vec2::new(0.0, 62.0);
        let ink = Stroke::new(5.0, accent);
        let tray = [
            icon + Vec2::new(-34.0, 6.0),
            icon + Vec2::new(-34.0, 30.0),
            icon + Vec2::new(34.0, 30.0),
            icon + Vec2::new(34.0, 6.0),
        ];
        painter.add(egui::Shape::line(tray.to_vec(), ink));
        if unsupported {
            painter.line_segment([icon + Vec2::new(-14.0, -30.0), icon + Vec2::new(14.0, -2.0)], ink);
            painter.line_segment([icon + Vec2::new(14.0, -30.0), icon + Vec2::new(-14.0, -2.0)], ink);
        } else {
            let bob = 4.0 * (t * 3.0).sin();
            let tip = icon + Vec2::new(0.0, 14.0 + bob);
            painter.line_segment([icon + Vec2::new(0.0, -34.0 + bob), tip], ink);
            painter.add(egui::Shape::line(vec![tip + Vec2::new(-13.0, -13.0), tip, tip + Vec2::new(13.0, -13.0)], ink));
        }

        let mut y = icon.y + 50.0;
        for g in [title, files, hint] {
            let h = g.size().y;
            painter.galley(egui::pos2(card.center().x - g.size().x / 2.0, y), g, Color32::WHITE);
            y += h + 8.0;
        }
        ctx.request_repaint();
    }

    fn loading_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let Some(job) = &self.job else { return };
        let p = job.progress.load(Ordering::Relaxed) as f32 / 1000.0;
        let stage = job.stage.lock().unwrap().clone();
        let w = 260.0;
        let r = Rect::from_center_size(rect.center() + Vec2::new(0.0, 40.0), Vec2::new(w, 8.0));
        painter.rect_filled(r.expand(10.0).translate(Vec2::new(0.0, -8.0)), 6.0, Color32::from_black_alpha(160));
        painter.rect_filled(r, 3.0, Color32::from_gray(60));
        painter.rect_filled(Rect::from_min_size(r.min, Vec2::new(w * p, 8.0)), 3.0, Color32::from_rgb(90, 160, 255));
        painter.text(
            r.center_top() - Vec2::new(0.0, 6.0),
            Align2::CENTER_BOTTOM,
            format!("{stage}…"),
            egui::FontId::proportional(13.0),
            Color32::WHITE,
        );
    }

    // ------------------------------------------------------------------ chrome

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button(tr!("File"), |ui| {
                if ui.add(egui::Button::new(tr!("Open…")).shortcut_text(tr!("Ctrl+O"))).clicked() {
                    self.open_dialog(&ctx);
                }
                // Recent files and Exit only make sense for the desktop app.
                #[cfg(not(target_arch = "wasm32"))]
                ui.menu_button(tr!("Recent Files"), |ui| {
                    if self.settings.mru.is_empty() {
                        ui.label(tr!("(none)"));
                    }
                    let mut chosen = None;
                    for (i, p) in self.settings.mru.iter().enumerate() {
                        let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                        if ui.button(format!("{} {name}", i + 1)).on_hover_text(p.display().to_string()).clicked() {
                            chosen = Some(p.clone());
                        }
                    }
                    if let Some(p) = chosen {
                        if p.exists() {
                            self.open(&ctx, Source::Path(p), None);
                        } else {
                            self.settings.mru.retain(|m| m != &p);
                            self.message =
                                Some(Message { title: tr!("Recent Files").into(), text: tr!("The file could not be found. It may have been moved or deleted.").into() });
                        }
                    }
                });
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.separator();
                    if ui.add(egui::Button::new(tr!("Exit")).shortcut_text(tr!("Ctrl+Q"))).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            });
            ui.menu_button(tr!("Edit"), |ui| {
                if ui.button(tr!("Settings…")).clicked() {
                    self.options = Some(self.settings.clone());
                }
            });
            ui.menu_button(tr!("View"), |ui| {
                ui.checkbox(&mut self.settings.show_toolbar, tr!("Button Bar"));
                ui.checkbox(&mut self.settings.show_status_bar, tr!("Info Bar"));
                if ui.add(egui::Button::new(tr!("Fullscreen")).shortcut_text("F11")).clicked() {
                    self.set_fullscreen(&ctx, !self.fullscreen);
                }
                ui.separator();
                let has = self.pano.is_some();
                let has_hs = self.pano.as_ref().is_some_and(|p| !p.hotspots.is_empty());
                ui.add_enabled_ui(has_hs, |ui| {
                    if ui.add(egui::Button::new(tr!("Hotspot Markers")).selected(self.show_hotspots).shortcut_text("H")).clicked() {
                        self.show_hotspots = !self.show_hotspots;
                    }
                    if ui.button(tr!("Jump to Hotspot…")).clicked() {
                        self.hotspot_list = Some(None);
                    }
                });
                ui.add_enabled_ui(has, |ui| {
                    if ui.add(egui::Button::new(tr!("Grid Overlay")).selected(self.show_grid).shortcut_text("G")).clicked() {
                        self.show_grid = !self.show_grid;
                    }
                    if ui.add(egui::Button::new(tr!("Reset View")).shortcut_text(tr!("Home"))).clicked() {
                        self.reset_view();
                    }
                });
                ui.separator();
                let prev = self.folder_neighbour(-1);
                let next = self.folder_neighbour(1);
                if ui.add_enabled(prev.is_some(), egui::Button::new(tr!("Previous in Folder")).shortcut_text(tr!("Alt+Left"))).clicked() {
                    self.open(&ctx, Source::Path(prev.unwrap()), None);
                }
                if ui.add_enabled(next.is_some(), egui::Button::new(tr!("Next in Folder")).shortcut_text(tr!("Alt+Right"))).clicked() {
                    self.open(&ctx, Source::Path(next.unwrap()), None);
                }
            });
            ui.menu_button(tr!("Tools"), |ui| {
                let has = self.pano.is_some();
                ui.add_enabled_ui(has, |ui| {
                    if ui.button(tr!("Go to View…")).clicked() {
                        let v = &self.pano.as_ref().unwrap().view;
                        self.view_params =
                            Some(ViewParamsDraft { yaw: v.yaw, pitch: v.pitch, hfov: v.hfov, confirm_default: false });
                    }
                    if ui.button(tr!("Photo Sphere Tags…")).clicked() {
                        self.open_metadata();
                    }
                    if ui.button(tr!("Find Horizon…")).clicked() {
                        self.message = Some(Message {
                            title: tr!("Horizon Finder").into(),
                            text: tr!(
                                "Click the horizon in the panorama to measure where it lies in the \
                                 image.\n\nThe result is displayed only; nothing gets saved. \
                                 Press Esc to cancel."
                            )
                            .into(),
                        });
                        self.horizon_tool = true;
                    }
                    if ui.button(tr!("Benchmark")).clicked() {
                        self.speed_test = Some(SpeedTest { t0: Instant::now(), frames: 0, turned: 0.0 });
                        ctx.request_repaint();
                    }
                });
            });
            ui.menu_button(tr!("Help"), |ui| {
                if ui.button(tr!("About RustyPano…")).clicked() {
                    self.about = true;
                }
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button(tr!("Open")).on_hover_text(tr!("Open a panorama (Ctrl+O)")).clicked() {
                self.open_dialog(&ctx);
            }
            let prev = self.folder_neighbour(-1);
            let next = self.folder_neighbour(1);
            if ui.add_enabled(prev.is_some(), egui::Button::new(tr!("< Prev"))).on_hover_text(tr!("Previous in folder (Alt+Left)")).clicked() {
                self.open(&ctx, Source::Path(prev.unwrap()), None);
            }
            if ui.add_enabled(next.is_some(), egui::Button::new(tr!("Next >"))).on_hover_text(tr!("Next in folder (Alt+Right)")).clicked() {
                self.open(&ctx, Source::Path(next.unwrap()), None);
            }
            ui.separator();
            if ui.button(tr!("Fullscreen")).on_hover_text(tr!("Fullscreen (F11)")).clicked() {
                self.set_fullscreen(&ctx, true);
            }
            let has_hs = self.pano.as_ref().is_some_and(|p| !p.hotspots.is_empty());
            ui.add_enabled_ui(has_hs, |ui| {
                ui.toggle_value(&mut self.show_hotspots, tr!("Hotspots")).on_hover_text(tr!("Hotspot markers (H)"));
            });
            ui.add_enabled_ui(self.pano.is_some(), |ui| {
                ui.toggle_value(&mut self.show_grid, tr!("Grid")).on_hover_text(tr!("Grid overlay (G)"));
                ui.separator();
                if ui.button(" + ").on_hover_text(tr!("Zoom in (A / Shift / wheel)")).clicked()
                    && let Some(p) = &mut self.pano {
                        p.view.zoom(1.0 / 1.25);
                    }
                if ui.button(" - ").on_hover_text(tr!("Zoom out (Z / Ctrl / wheel)")).clicked()
                    && let Some(p) = &mut self.pano {
                        p.view.zoom(1.25);
                    }
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(job) = &self.job {
                let p = job.progress.load(Ordering::Relaxed) as f32 / 1000.0;
                ui.add(egui::ProgressBar::new(p).desired_width(140.0).desired_height(12.0));
                ui.label(format!("{} {}", job.stage.lock().unwrap(), job.name));
            } else if let (Some(i), Some(p)) = (self.hovered_hotspot, &self.pano) {
                let d = &p.hotspots[i].def;
                let t = if d.description.is_empty() { d.target.clone() } else { d.description.clone() };
                ui.label(tr!("Hotspot: {t}", t = t));
            } else if let Some(p) = &self.pano {
                ui.label(p.file_name());
            } else {
                ui.label(tr!("Ready"));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(p) = &self.pano {
                    let v = &p.view;
                    let l = &p.loaded;
                    ui.label(RichText::new(format!("{}×{}", l.width, l.height)).weak());
                    if let Some((done, total)) = l.tiles.preload_progress() {
                        ui.separator();
                        ui.label(tr!("Loading tiles to the GPU: {done} / {total}", done = done, total = total));
                    }
                    ui.separator();
                    ui.label(format!("{:.0}%", v.zoom_percent()));
                    ui.separator();
                    if self.settings.show_yaw_pitch_in_status_bar {
                        ui.label(format!("V: {:.1}", v.vfov()));
                        ui.label(format!("H: {:.1}", v.hfov));
                        ui.label(format!("P: {:.1}", v.pitch));
                        ui.label(format!("Y: {:.1}", v.yaw));
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------ dialogs

    fn open_metadata(&mut self) {
        if let Some(p) = &self.pano {
            self.metadata = Some(MetadataDraft {
                tags: p.loaded.gpano.clone(),
                horizon: p.view.geo.horizon_percent().round(),
                note: None,
                dirty: false,
            });
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        if let Some(m) = &self.message {
            let mut close = false;
            egui::Modal::new(egui::Id::unique("message")).show(ctx, |ui| {
                ui.set_max_width(420.0);
                ui.heading(&m.title);
                ui.add_space(6.0);
                ui.label(&m.text);
                ui.add_space(8.0);
                if ui.button(tr!("OK")).clicked() || ui.input(|i| i.key_pressed(Key::Enter)) {
                    close = true;
                }
            });
            if close {
                self.message = None;
            }
        }
        self.options_dialog(ctx);
        self.view_params_dialog(ctx);
        self.metadata_dialog(ctx);
        self.hotspots_dialog(ctx);
        if self.about {
            let mut open = true;
            egui::Window::new(tr!("About RustyPano"))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.heading(format!("RustyPano {}", env!("CARGO_PKG_VERSION")));
                    ui.label(tr!("A GPU-accelerated viewer for equirectangular panoramas."));
                    ui.label(tr!("Built with wgpu, winit and egui."));
                    ui.label(
                        RichText::new(tr!("Tile encoding is based in part on the work of the Independent JPEG Group."))
                            .small()
                            .weak(),
                    );
                    ui.add_space(6.0);
                    if let Some(p) = &self.pano {
                        let l = &p.loaded;
                        let g = &l.tiles.gpu;
                        ui.label(tr!(
                            "Current image: {w}×{h} px in {levels} resolution levels, loaded in {secs} s",
                            w = l.width,
                            h = l.height,
                            levels = g.grid.levels.len(),
                            secs = format!("{:.2}", l.load_time)
                        ));
                        if l.tiles.keeps_all() {
                            ui.label(tr!("All tiles are kept in GPU memory."));
                        }
                        ui.label(if !l.tiles.is_persistent() {
                            tr!("Tiles are kept in memory (no tile cache available).")
                        } else if l.from_cache {
                            tr!("Tiles loaded from the tile cache.")
                        } else {
                            tr!("Tiles built and saved to the tile cache.")
                        });
                        ui.label(tr!(
                            "GPU tile cache: {used} of {slots} slots in use ({mb} MB), tiles of {tw}×{th} px",
                            used = l.tiles.used_slots(),
                            slots = g.slots,
                            mb = (g.slots as u64 * g.grid.slot_bytes()) >> 20,
                            tw = g.grid.cx,
                            th = g.grid.cy
                        ));
                        if let Some(icc) = &l.icc_used {
                            ui.label(tr!("Colour management: {icc}", icc = icc));
                        }
                    }
                    ui.label(tr!("Render: {fps} fps", fps = format!("{:.0}", self.fps)));
                });
            self.about = open;
        }
    }

    fn options_dialog(&mut self, ctx: &egui::Context) {
        let Some(draft) = &mut self.options else { return };
        let mut action = 0; // 1 = ok, 2 = cancel
        let mut error = None;
        egui::Window::new(tr!("Settings"))
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (i, t) in [tr!("View"), tr!("Mouse"), tr!("Quality"), tr!("Colour"), tr!("Grid"), tr!("System")].iter().enumerate() {
                        ui.selectable_value(&mut self.options_tab, i, *t);
                    }
                });
                ui.separator();
                ui.set_min_width(440.0);
                match self.options_tab {
                    0 => {
                        egui::Grid::new("gen").num_columns(2).show(ui, |ui| {
                            ui.label(tr!("Language"));
                            let name = |c: LanguageChoice| match c {
                                LanguageChoice::Auto => {
                                    tr!("Automatic ({language})", language = i18n::detect().native_name())
                                }
                                other => other.resolve().native_name().to_string(),
                            };
                            egui::ComboBox::from_id_salt("language").selected_text(name(draft.language)).show_ui(ui, |ui| {
                                for c in LanguageChoice::ALL {
                                    ui.selectable_value(&mut draft.language, c, name(c));
                                }
                            });
                            ui.end_row();
                            ui.label(tr!("Start yaw"));
                            ui.add(egui::DragValue::new(&mut draft.default_yaw).range(0.0..=360.0).speed(0.5));
                            ui.end_row();
                            ui.label(tr!("Start pitch"));
                            ui.add(egui::DragValue::new(&mut draft.default_pitch).range(-90.0..=90.0).speed(0.5));
                            ui.end_row();
                            ui.label(tr!("Start field of view"));
                            ui.add(egui::DragValue::new(&mut draft.default_hfov).range(1.0..=MAX_HFOV).speed(0.5));
                            ui.end_row();
                        });
                        ui.checkbox(&mut draft.show_yaw_pitch_in_status_bar, tr!("Display view angles in the info bar"));
                        ui.checkbox(&mut draft.show_hotspot_tooltips, tr!("Describe hotspots on hover"));
                    }
                    1 => {
                        ui.label(tr!("Mouse behaviour"));
                        ui.radio_value(&mut draft.panning_mode, PanningMode::ClickAndDrag, tr!("Grab: the image follows the pointer"));
                        ui.radio_value(&mut draft.panning_mode, PanningMode::AutoPanning, tr!("Joystick: speed grows with distance from the click"));
                        ui.add_space(6.0);
                        ui.add_enabled_ui(draft.panning_mode == PanningMode::ClickAndDrag, |ui| {
                            ui.checkbox(&mut draft.smooth_panning, tr!("Smooth panning: the image glides on after a quick drag"));
                        });
                        ui.add_space(6.0);
                        ui.add_enabled_ui(draft.panning_mode == PanningMode::AutoPanning, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(tr!("Joystick speed:"));
                                ui.add(egui::Slider::new(&mut draft.mouse_sensitivity, 1.0..=10.0).step_by(0.5));
                            });
                        });
                    }
                    2 => {
                        let combo = |ui: &mut egui::Ui, id: &str, v: &mut Interpolator| {
                            egui::ComboBox::from_id_salt(id).selected_text(v.label()).width(260.0).show_ui(ui, |ui| {
                                for i in Interpolator::ALL {
                                    ui.selectable_value(v, i, i.label());
                                }
                            });
                        };
                        egui::Grid::new("interp").num_columns(2).show(ui, |ui| {
                            ui.label(tr!("When still"));
                            combo(ui, "steady", &mut draft.interpolator_steady);
                            ui.end_row();
                            ui.label(tr!("While moving"));
                            combo(ui, "panning", &mut draft.interpolator_panning);
                            ui.end_row();
                        });
                        ui.label(RichText::new(tr!("All interpolators run on the GPU with mipmapping.")).weak());
                    }
                    3 => {
                        ui.checkbox(&mut draft.enable_color_management, tr!("Colour-manage images"));
                        ui.add_enabled_ui(draft.enable_color_management, |ui| {
                            ui.label(tr!("Display profile"));
                            ui.radio_value(&mut draft.monitor_profile, MonitorProfile::Srgb, tr!("Standard sRGB display"));
                            // Browsers cannot read local ICC files.
                            #[cfg(not(target_arch = "wasm32"))]
                            ui.horizontal(|ui| {
                                ui.radio_value(&mut draft.monitor_profile, MonitorProfile::File, tr!("ICC file:"));
                                ui.add_enabled(
                                    draft.monitor_profile == MonitorProfile::File,
                                    egui::TextEdit::singleline(&mut draft.monitor_profile_file).desired_width(200.0),
                                );
                                if ui.add_enabled(draft.monitor_profile == MonitorProfile::File, egui::Button::new("…")).clicked() {
                                    let mut d = rfd::FileDialog::new()
                                        .set_title(tr!("Choose display ICC profile"))
                                        .add_filter(tr!("ICC profiles"), &["icc", "icm"]);
                                    if Path::new("/usr/share/color/icc").is_dir() {
                                        d = d.set_directory("/usr/share/color/icc");
                                    }
                                    if let Some(p) = d.pick_file() {
                                        draft.monitor_profile_file = p.display().to_string();
                                    }
                                }
                            });
                        });
                        ui.label(RichText::new(tr!("Changes apply when the panorama is reloaded.")).weak());
                    }
                    4 => {
                        let g = &mut draft.grid;
                        egui::Grid::new("grid").num_columns(2).show(ui, |ui| {
                            ui.checkbox(&mut g.show_rows, tr!("Horizontal lines"));
                            ui.add(egui::DragValue::new(&mut g.rows_distance).range(0.5..=90.0).suffix("°").prefix(tr!("every ")));
                            ui.end_row();
                            ui.checkbox(&mut g.show_columns, tr!("Vertical lines"));
                            ui.add(egui::DragValue::new(&mut g.columns_distance).range(0.5..=180.0).suffix("°").prefix(tr!("every ")));
                            ui.end_row();
                            ui.label(tr!("Line thickness"));
                            ui.add(egui::DragValue::new(&mut g.line_width).range(0.5..=10.0).speed(0.1).suffix(" px"));
                            ui.end_row();
                            ui.label(tr!("Colour"));
                            let mut c = Color32::from_rgba_unmultiplied(g.color[0], g.color[1], g.color[2], g.color[3]);
                            if ui.color_edit_button_srgba(&mut c).changed() {
                                g.color = c.to_srgba_unmultiplied();
                            }
                            ui.end_row();
                        });
                    }
                    _ => {
                        ui.horizontal(|ui| {
                            ui.label(if cfg!(target_arch = "wasm32") {
                                tr!("Tile cache in the browser")
                            } else {
                                tr!("Tile cache on disk")
                            });
                            ui.add(egui::DragValue::new(&mut draft.tile_cache_mb).range(256..=1_048_576).suffix(" MB").speed(64));
                            if ui.button(tr!("Clear")).clicked() {
                                self.clear_cache_requested = true;
                            }
                        });
                        let mut note = tr!("Tiles of panoramas opened in this session are kept here; they are deleted when the app is closed.").to_string();
                        if let Some(dir) = crate::store::location() {
                            note = format!("{note}\n{dir}");
                        }
                        ui.label(RichText::new(note).weak());
                        ui.add_space(4.0);
                        ui.checkbox(&mut draft.keep_all_tiles_on_gpu, tr!("Keep all tiles in GPU memory"));
                        let need = self.pano.as_ref().map(|p| {
                            let g = &p.loaded.tiles.gpu.grid;
                            (g.tile_count() as u64 * g.slot_bytes()) >> 20
                        });
                        let mut note = tr!("No waiting for tiles while panning or zooming, but needs much more video memory; falls back to loading on demand if the GPU cannot hold them all.").to_string();
                        if let Some(mb) = need {
                            note = format!("{note} {}", tr!("The current panorama needs {mb} MB.", mb = mb));
                        }
                        ui.label(RichText::new(note).weak());
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(tr!("GPU tile cache"));
                            ui.add(egui::DragValue::new(&mut draft.max_gpu_memory_mb).range(64..=4096).suffix(" MB").speed(16));
                        });
                        ui.label(RichText::new(tr!("Only the visible part is kept on the GPU in full resolution; a bigger cache reloads less while panning.")).weak());
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.checkbox(&mut draft.vsync, tr!("Vertical sync (needs restart)"));
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(tr!("OK")).clicked() {
                        action = 1;
                    }
                    if ui.button(tr!("Cancel")).clicked() {
                        action = 2;
                    }
                });
            });
        if action == 1 {
            if draft.default_hfov > MAX_HFOV {
                error = Some(tr!("The start field of view must not exceed {max}°.", max = MAX_HFOV));
            } else if draft.enable_color_management
                && draft.monitor_profile == MonitorProfile::File
                && draft.monitor_profile_file.trim().is_empty()
            {
                error = Some(
                    tr!("Pick an ICC file for the display profile, or switch to sRGB.").into(),
                );
            } else if draft.enable_color_management
                && draft.monitor_profile == MonitorProfile::File
                && !Path::new(draft.monitor_profile_file.trim()).exists()
            {
                error = Some(tr!("The selected ICC file was not found.").into());
            }
            if let Some(e) = error {
                self.message = Some(Message { title: tr!("Settings").into(), text: e });
                return;
            }
            let new = self.options.take().unwrap();
            let cm_changed = new.enable_color_management != self.settings.enable_color_management
                || new.monitor_profile != self.settings.monitor_profile
                || new.monitor_profile_file != self.settings.monitor_profile_file
                || new.max_gpu_memory_mb != self.settings.max_gpu_memory_mb
                || new.keep_all_tiles_on_gpu != self.settings.keep_all_tiles_on_gpu;
            let mru = std::mem::take(&mut self.settings.mru);
            self.settings = new;
            self.settings.mru = mru;
            i18n::apply(self.settings.language);
            self.settings.save();
            if cm_changed {
                self.reload(ctx);
            }
        } else if action == 2 {
            self.options = None;
        }
    }

    fn view_params_dialog(&mut self, ctx: &egui::Context) {
        let Some(d) = &mut self.view_params else { return };
        let mut action = 0;
        egui::Window::new(tr!("Go to View"))
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                egui::Grid::new("vp").num_columns(2).show(ui, |ui| {
                    ui.label(tr!("Yaw"));
                    ui.add(egui::DragValue::new(&mut d.yaw).range(0.0..=360.0).speed(0.5).max_decimals(2));
                    ui.end_row();
                    ui.label(tr!("Pitch"));
                    ui.add(egui::DragValue::new(&mut d.pitch).range(-90.0..=90.0).speed(0.5).max_decimals(2));
                    ui.end_row();
                    ui.label(tr!("Field of view"));
                    ui.add(egui::DragValue::new(&mut d.hfov).range(0.5..=MAX_HFOV).speed(0.5).max_decimals(2));
                    ui.end_row();
                });
                if d.confirm_default {
                    ui.separator();
                    ui.label(
                        tr!(
                            "Use these values as the start view for every image\n\
                             that has no view settings of its own?"
                        ),
                    );
                    ui.horizontal(|ui| {
                        if ui.button(tr!("Yes")).clicked() {
                            action = 3;
                        }
                        if ui.button(tr!("No")).clicked() {
                            d.confirm_default = false;
                        }
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(tr!("OK")).clicked() {
                        action = 1;
                    }
                    if ui.button(tr!("Cancel")).clicked() {
                        action = 2;
                    }
                    if ui.button(tr!("Make Start View")).clicked() {
                        d.confirm_default = true;
                    }
                });
            });
        match action {
            1 => {
                let d = self.view_params.take().unwrap();
                if let Some(p) = &mut self.pano {
                    let v = &mut p.view;
                    v.hfov = d.hfov;
                    v.pitch = d.pitch;
                    v.yaw = d.yaw;
                    v.set_hfov(d.hfov);
                    let mut issues = Vec::new();
                    if (v.hfov - d.hfov).abs() > 0.01 {
                        issues.push(tr!("field of view"));
                    }
                    if (v.pitch - d.pitch).abs() > 0.01 {
                        issues.push(tr!("pitch"));
                    }
                    if (v.yaw - d.yaw.rem_euclid(360.0)).abs() > 0.01 {
                        issues.push(tr!("yaw"));
                    }
                    if !issues.is_empty() {
                        self.message = Some(Message {
                            title: tr!("Go to View").into(),
                            text: tr!("Adjusted to fit the image and window: {list}.", list = issues.join(", ")),
                        });
                    }
                }
            }
            2 => self.view_params = None,
            3 => {
                let d = self.view_params.as_mut().unwrap();
                d.confirm_default = false;
                self.settings.default_yaw = d.yaw;
                self.settings.default_pitch = d.pitch;
                self.settings.default_hfov = d.hfov;
                self.settings.save();
            }
            _ => {}
        }
    }

    fn metadata_dialog(&mut self, ctx: &egui::Context) {
        let Some(m) = &mut self.metadata else { return };
        let Some(p) = &self.pano else {
            self.metadata = None;
            return;
        };
        let is_jpeg = ["jpg", "jpeg", "jpe"].contains(&p.loaded.image_source().extension().as_str());
        // Tags are written back into the file, which needs a local file.
        let writable = is_jpeg && p.image_path().is_some();
        let mut action = 0;
        egui::Window::new(tr!("Photo Sphere Tags"))
            .collapsible(false)
            .resizable(true)
            .default_width(480.0)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(tr!("Photo Sphere (GPano) XMP tags stored in this file:"));
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                        egui::Grid::new("tags").striped(true).num_columns(2).min_col_width(120.0).show(ui, |ui| {
                            ui.strong(tr!("Tag"));
                            ui.strong(tr!("Value"));
                            ui.end_row();
                            let mut names: Vec<&String> = m.tags.tags.keys().collect();
                            names.sort_by_key(|n| GPANO_TAGS.iter().position(|t| t == n).unwrap_or(usize::MAX));
                            for n in names {
                                ui.label(format!("GPano:{n}"));
                                ui.label(&m.tags.tags[n]);
                                ui.end_row();
                            }
                            if m.tags.is_empty() {
                                ui.label(RichText::new(tr!("(none)")).weak());
                                ui.end_row();
                            }
                        });
                    });
                });
                ui.add_space(4.0);
                ui.group(|ui| {
                    ui.label(tr!("Horizon"));
                    ui.horizontal(|ui| {
                        ui.label(tr!("Height from top"));
                        ui.add(egui::DragValue::new(&mut m.horizon).range(0.0..=100.0).suffix(" %"));
                        if ui
                            .button(tr!("Write Crop Tags"))
                            .on_hover_text(tr!("Store the horizon height as GPano cropped-area tags"))
                            .clicked()
                        {
                            let w = p.loaded.width as f64;
                            let h = p.loaded.height as f64;
                            let full_h = (w / 2.0).round();
                            let top = (full_h / 2.0 - m.horizon / 100.0 * h).round().max(0.0);
                            m.tags.set("ProjectionType", "equirectangular");
                            m.tags.set("UsePanoramaViewer", "True");
                            m.tags.set("FullPanoWidthPixels", w as u64);
                            m.tags.set("FullPanoHeightPixels", full_h as u64);
                            m.tags.set("CroppedAreaImageWidthPixels", w as u64);
                            m.tags.set("CroppedAreaImageHeightPixels", h as u64);
                            m.tags.set("CroppedAreaLeftPixels", 0);
                            m.tags.set("CroppedAreaTopPixels", top as u64);
                            m.dirty = true;
                            m.note = Some(
                                tr!(
                                    "Tags changed. Press OK to write them into the file, then reopen the image \
                                     to apply the new horizon."
                                )
                                .into(),
                            );
                        }
                    });
                });
                ui.group(|ui| {
                    ui.label(tr!("Start view"));
                    let v = &p.view;
                    ui.label(tr!("Yaw {yaw}°   Pitch {pitch}°   Field of view {fov}°", yaw = format!("{:.1}", v.yaw), pitch = format!("{:.1}", v.pitch), fov = format!("{:.1}", v.hfov)));
                    if ui
                        .button(tr!("Use Current View"))
                        .on_hover_text(tr!("Save the current yaw, pitch and field of view as the view shown when the file is opened"))
                        .clicked()
                    {
                        let pose = m.tags.get("PoseHeadingDegrees").unwrap_or(0.0);
                        let heading = (v.yaw - 180.0 + pose).rem_euclid(360.0);
                        m.tags.set("InitialViewHeadingDegrees", format!("{heading:.1}"));
                        m.tags.set("InitialViewPitchDegrees", format!("{:.1}", v.pitch));
                        m.tags.set("InitialHorizontalFOVDegrees", format!("{:.1}", v.hfov));
                        m.dirty = true;
                        m.note = Some(
                            tr!("Tags changed. Press OK to write them into the file.")
                                .into(),
                        );
                    }
                });
                if let Some(n) = &m.note {
                    ui.label(RichText::new(n).color(Color32::from_rgb(230, 190, 80)));
                }
                if !is_jpeg {
                    ui.label(RichText::new(tr!("Writing tags is supported for JPEG files only.")).weak());
                } else if !writable {
                    ui.label(RichText::new(tr!("Tags can only be saved into local files in the desktop version.")).weak());
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.add_enabled(writable || !m.dirty, egui::Button::new(tr!("OK"))).clicked() {
                        action = 1;
                    }
                    if ui.button(tr!("Cancel")).clicked() {
                        action = 2;
                    }
                });
            });
        match action {
            1 => {
                #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
                let m = self.metadata.take().unwrap();
                #[cfg(not(target_arch = "wasm32"))]
                if m.dirty
                    && let Some(path) = self.pano.as_ref().and_then(|p| p.image_path()).map(Path::to_path_buf)
                {
                    match crate::xmp::write_gpano_jpeg(&path, &m.tags) {
                        Ok(()) => {
                            if let Some(p) = &mut self.pano {
                                p.loaded.gpano = GPano::read(&path);
                            }
                        }
                        Err(e) => {
                            self.message = Some(Message { title: tr!("Photo Sphere Tags").into(), text: tr!("Could not save the tags:\n{error}", error = e) })
                        }
                    }
                }
            }
            2 => self.metadata = None,
            _ => {}
        }
    }

    fn hotspots_dialog(&mut self, ctx: &egui::Context) {
        let Some(sel) = &mut self.hotspot_list else { return };
        let Some(p) = &mut self.pano else {
            self.hotspot_list = None;
            return;
        };
        let mut action = 0;
        egui::Window::new(tr!("Jump to Hotspot"))
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(tr!("Destinations in this panorama:"));
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    ui.set_min_width(300.0);
                    for (i, h) in p.hotspots.iter().enumerate() {
                        let name = if h.def.description.is_empty() {
                            Path::new(&h.def.target.replace('\\', "/"))
                                .file_name()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_default()
                        } else {
                            h.def.description.clone()
                        };
                        let r = ui.selectable_label(*sel == Some(i), name);
                        if r.clicked() {
                            *sel = Some(i);
                        }
                        if r.double_clicked() {
                            action = 1;
                        }
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.add_enabled(sel.is_some(), egui::Button::new(tr!("OK"))).clicked() {
                        action = 1;
                    }
                    if ui.button(tr!("Cancel")).clicked() {
                        action = 2;
                    }
                });
            });
        if action == 1 {
            if let Some(i) = sel.take() {
                let (y, pt) = (p.hotspots[i].yaw, p.hotspots[i].pitch);
                p.view.set_pitch(pt);
                p.view.set_yaw(y);
                p.autospin = 0.0;
            }
            self.hotspot_list = None;
        } else if action == 2 {
            self.hotspot_list = None;
        }
    }
}

impl RustyPanoApp {
    fn debug_screenshot(&mut self, ctx: &egui::Context) {
        let Some((_, frames)) = &mut self.screenshot else { return };
        if self.pano.is_none() || self.job.is_some() {
            ctx.request_repaint();
            return;
        }
        if *frames == 0 {
            match std::env::var("RUSTYPANO_DIALOG").as_deref() {
                Ok("options") => {
                    self.options = Some(self.settings.clone());
                    self.options_tab = std::env::var("RUSTYPANO_TAB").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
                }
                Ok("metadata") => self.open_metadata(),
                Ok("hotspots") => self.hotspot_list = Some(Some(0)),
                Ok("view") => {
                    let v = &self.pano.as_ref().unwrap().view;
                    self.view_params = Some(ViewParamsDraft { yaw: v.yaw, pitch: v.pitch, hfov: v.hfov, confirm_default: false });
                }
                Ok("about") => self.about = true,
                _ => {}
            }
        }
        let Some((path, frames)) = &mut self.screenshot else { return };
        *frames += 1;
        let steady = self.last_motion.elapsed().as_secs_f32() > 0.4 && !self.tiles_streaming;
        // RUSTYPANO_SCREENSHOT_FRAMES: take the screenshot after this many
        // frames even if the view keeps moving (default 150).
        let max_frames: u32 = std::env::var("RUSTYPANO_SCREENSHOT_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
        if ((*frames >= 20 && steady && max_frames == 150) || *frames == max_frames) && *frames < 1_000_000 {
            *frames = 1_000_000;
            let path = path.clone();
            let c2 = ctx.clone();
            ctx.request_screenshot(move |img| {
                let [w, h] = img.size;
                let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                if let Some(buf) = image::RgbaImage::from_raw(w as u32, h as u32, rgba)
                    && let Err(e) = buf.save(&path) {
                        log::error!("screenshot: {e}");
                    }
                c2.send_viewport_cmd(egui::ViewportCommand::Close);
            });
        }
        ctx.request_repaint();
    }
}

impl eframe::App for RustyPanoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // The browser leaves fullscreen on Esc by itself; follow its state.
        #[cfg(target_arch = "wasm32")]
        {
            self.fullscreen = crate::web::is_fullscreen();
        }
        self.poll_job(&ctx);
        if std::mem::take(&mut self.clear_cache_requested) {
            self.clear_tile_cache(&ctx);
        }
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.handle_shortcuts(&ctx);

        if !self.fullscreen {
            egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));
            if self.settings.show_toolbar {
                egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
            }
            if self.settings.show_status_bar {
                egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
            }
        }
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui));
        self.dialogs(&ctx);
        self.drop_hint(&ctx);
        self.debug_screenshot(&ctx);
    }

    fn on_exit(&mut self) {
        self.settings.save();
        // The tile cache only lives as long as the app.
        if let Some(job) = self.job.take() {
            job.cancel.store(true, Ordering::Relaxed);
        }
        self.pano = None;
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(crate::store::end_session());
    }
}
