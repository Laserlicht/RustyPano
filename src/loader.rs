// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Background loading: decode → colour-manage → split into GPU tiles with
//! gutters → build mipmaps → upload into one `texture_2d_array`.
//!
//! Panoramas are usually far bigger than the GPU's maximum texture size, so the
//! equirectangular image is cut into square tiles of `tile_content` pixels,
//! each stored in an array layer of `tile_size = tile_content + 2*GUTTER`
//! pixels. The gutter holds real neighbouring pixels (wrapping horizontally,
//! clamped vertically), so filtering never shows seams up to mip level
//! `MAX_LOD`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use eframe::wgpu;
use image::{ImageDecoder, ImageReader};

use crate::fsv::PanoParams;
use crate::settings::{MonitorProfile, Settings};
use crate::xmp::GPano;

pub const GUTTER: u32 = 32;
pub const MIP_LEVELS: u32 = 6; // levels 0..=5; gutter is 1 texel at level 5
pub const MAX_LOD: u32 = MIP_LEVELS - 1;
const MAX_TILE: u32 = 4096;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

pub struct GpuPano {
    pub id: u32,
    pub view: wgpu::TextureView,
    _texture: wgpu::Texture,
    /// Size of the image stored on the GPU (may be downscaled).
    pub gpu_width: u32,
    pub gpu_height: u32,
    pub tile_size: u32,
    pub tile_content: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
}

pub struct LoadedPano {
    pub source: PathBuf,
    pub params: PanoParams,
    pub gpano: GPano,
    pub width: u32,
    pub height: u32,
    pub gpu: Arc<GpuPano>,
    pub downscale: u32,
    pub icc_used: Option<String>,
    pub load_time: f32,
}

#[derive(Clone)]
pub struct Overrides {
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub hfov: Option<f64>,
}

pub struct LoadJob {
    pub path: PathBuf,
    pub progress: Arc<AtomicU32>, // 0..=1000
    pub stage: Arc<Mutex<String>>,
    pub cancel: Arc<AtomicBool>,
    pub rx: mpsc::Receiver<Result<LoadedPano, String>>,
    pub overrides: Option<Overrides>,
}

pub fn start_load(
    path: PathBuf,
    settings: &Settings,
    device: wgpu::Device,
    queue: wgpu::Queue,
    ctx: eframe::egui::Context,
    overrides: Option<Overrides>,
) -> LoadJob {
    let progress = Arc::new(AtomicU32::new(0));
    let stage = Arc::new(Mutex::new("Reading".to_string()));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let s = settings.clone();
    let (p2, st2, c2, path2) = (progress.clone(), stage.clone(), cancel.clone(), path.clone());
    std::thread::Builder::new()
        .name("pano-loader".into())
        .spawn(move || {
            let r = load(&path2, &s, &device, &queue, &p2, &st2, &c2, &ctx);
            let _ = tx.send(r);
            ctx.request_repaint();
        })
        .expect("spawn loader thread");
    LoadJob { path, progress, stage, cancel, rx, overrides }
}

/// Image file extensions the viewer opens (lower case).
pub const IMAGE_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "jpe", "tif", "tiff", "png", "webp", "bmp", "jxl"];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .is_some_and(|e| e == "fsv" || IMAGE_EXTENSIONS.contains(&e.as_str()))
}

/// Container formats recognised from the first bytes of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sniffed {
    Jxl,
    Other,
}

fn sniff(path: &Path) -> Sniffed {
    let mut head = [0u8; 12];
    let n = std::fs::File::open(path).and_then(|mut f| std::io::Read::read(&mut f, &mut head)).unwrap_or(0);
    let head = &head[..n];
    if head.starts_with(&[0xFF, 0x0A]) || head.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n") {
        Sniffed::Jxl
    } else {
        Sniffed::Other
    }
}

/// Decode any supported image, returning the pixels and the embedded ICC profile.
fn decode(path: &Path) -> Result<(image::DynamicImage, Option<Vec<u8>>), String> {
    let err = |e: &dyn std::fmt::Display| format!("Cannot decode this image: {e}");
    match sniff(path) {
        Sniffed::Jxl => {
            let f = std::io::BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
            let mut decoder = jxl_oxide::integration::JxlDecoder::new(f).map_err(|e| err(&e))?;
            let icc = decoder.icc_profile().ok().flatten();
            let img = image::DynamicImage::from_decoder(decoder).map_err(|e| err(&e))?;
            Ok((img, icc))
        }
        Sniffed::Other => {
            let mut reader = ImageReader::open(path).map_err(|e| e.to_string())?;
            reader = reader.with_guessed_format().map_err(|e| e.to_string())?;
            reader.no_limits();
            let mut decoder = reader.into_decoder().map_err(|e| err(&e))?;
            let icc = decoder.icc_profile().ok().flatten();
            let img = image::DynamicImage::from_decoder(decoder).map_err(|e| err(&e))?;
            Ok((img, icc))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load(
    path: &Path,
    settings: &Settings,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    progress: &AtomicU32,
    stage: &Mutex<String>,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
) -> Result<LoadedPano, String> {
    let t0 = Instant::now();
    let set_stage = |s: &str| {
        *stage.lock().unwrap() = s.to_string();
        ctx.request_repaint();
    };
    if !path.exists() {
        return Err(format!("No such file: {}", path.display()));
    }
    let is_fsv = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fsv"));
    let mut params = if is_fsv { PanoParams::read_fsv(path)? } else { PanoParams::default() };
    let image_path = if is_fsv { params.image.clone() } else { path.to_path_buf() };
    if !image_path.exists() {
        return Err(format!("No such file: {}", image_path.display()));
    }
    let gpano = GPano::read(&image_path);
    if !is_fsv {
        params = PanoParams::from_image(&image_path, &gpano);
    }

    set_stage("Decoding");
    let (img, icc) = decode(&image_path)?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let (width, height) = (img.width(), img.height());
    let mut rgb = img.into_rgb8();
    progress.store(150, Ordering::Relaxed);

    // ---- colour management -------------------------------------------------
    let mut icc_used = None;
    if settings.enable_color_management {
        set_stage("Colour management");
        let output = match settings.monitor_profile {
            MonitorProfile::File if !settings.monitor_profile_file.is_empty() => {
                qcms::Profile::new_from_path(&settings.monitor_profile_file)
            }
            _ => None,
        };
        let out_is_srgb = output.is_none();
        let input = icc.as_deref().and_then(|b| qcms::Profile::new_from_slice(b, false));
        if input.is_some() || !out_is_srgb {
            let input = input.unwrap_or_else(qcms::Profile::new_sRGB);
            let mut output = output.unwrap_or_else(qcms::Profile::new_sRGB);
            output.precache_output_transform();
            if let Some(xf) =
                qcms::Transform::new(&input, &output, qcms::DataType::RGB8, qcms::Intent::Perceptual)
            {
                let buf: &mut [u8] = &mut rgb;
                let n = std::thread::available_parallelism().map_or(4, |n| n.get());
                let row = width as usize * 3;
                let rows_per = (height as usize).div_ceil(n).max(1);
                std::thread::scope(|s| {
                    for chunk in buf.chunks_mut(rows_per * row) {
                        let xf = &xf;
                        s.spawn(move || xf.apply(chunk));
                    }
                });
                icc_used = Some(if icc.is_some() { "embedded ICC".into() } else { "sRGB (assumed)".into() });
            }
        }
    }
    progress.store(250, Ordering::Relaxed);

    // ---- fit into the GPU budget, upload, verify ----------------------------
    let limits = device.limits();
    let max_tile = MAX_TILE.min(limits.max_texture_dimension_2d);
    let max_layers = limits.max_texture_array_layers;
    let budget = settings.max_gpu_memory_mb.max(256) as u64 * 1024 * 1024;
    let mut downscale = 1u32;
    let (mut gw, mut gh) = (width, height);
    let (texture, layout) = loop {
        let layout = TileLayout::new(gw, gh, max_tile);
        let bytes = layout.tile_size as u64 * layout.tile_size as u64 * 4 * 4 / 3 * layout.layers() as u64;
        if (layout.layers() > max_layers || bytes > budget) && gw >= 1024 {
            set_stage("Downscaling");
            rgb = halve(&rgb);
            downscale *= 2;
            (gw, gh) = (rgb.width(), rgb.height());
            continue;
        }
        set_stage("Uploading to GPU");
        let tex = upload(&rgb, &layout, device, queue, progress, cancel, ctx);
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if verify(&tex, &rgb, &layout, device, queue) || gw < 1024 {
            break (tex, layout);
        }
        // Some drivers (notably OpenGL) fail texture allocation silently when
        // out of video memory: the readback did not match, so try again smaller.
        log::warn!("GPU upload of {gw}x{gh} failed verification (out of video memory?), downscaling");
        tex.destroy();
        drop(tex);
        set_stage("Downscaling");
        rgb = halve(&rgb);
        downscale *= 2;
        (gw, gh) = (rgb.width(), rgb.height());
    };
    if downscale > 1 {
        log::info!("Panorama downscaled by {downscale} to fit the GPU ({gw}x{gh})");
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("pano tiles view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let gpu = Arc::new(GpuPano {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        view,
        _texture: texture,
        gpu_width: gw,
        gpu_height: gh,
        tile_size: layout.tile_size,
        tile_content: layout.tile_content,
        tiles_x: layout.tiles_x,
        tiles_y: layout.tiles_y,
    });
    Ok(LoadedPano {
        source: path.to_path_buf(),
        params,
        gpano,
        width,
        height,
        gpu,
        downscale,
        icc_used,
        load_time: t0.elapsed().as_secs_f32(),
    })
}

fn mip_levels_for(layout: &TileLayout) -> u32 {
    MIP_LEVELS.min(32 - layout.tile_size.leading_zeros())
}

fn upload(
    rgb: &image::RgbImage,
    layout: &TileLayout,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    progress: &AtomicU32,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
) -> wgpu::Texture {
    let (gw, gh) = (rgb.width(), rgb.height());
    let mip_levels = mip_levels_for(layout);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pano tiles"),
        size: wgpu::Extent3d {
            width: layout.tile_size,
            height: layout.tile_size,
            depth_or_array_layers: layout.layers(),
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let lut = Lut::new();
    let n_tiles = layout.tiles_x * layout.tiles_y;
    let done = AtomicU32::new(0);
    let next = AtomicU32::new(0);
    // Each worker holds one tile (+ mips) at a time: ~90 MB at 4096².
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(n_tiles as usize).clamp(1, 8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let layer = next.fetch_add(1, Ordering::Relaxed);
                    if layer >= n_tiles {
                        return;
                    }
                    let tx = layer % layout.tiles_x;
                    let ty = layer / layout.tiles_x;
                    let mut level = build_tile(rgb, gw, gh, layout, tx, ty);
                    let mut size = layout.tile_size;
                    for mip in 0..mip_levels {
                        queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &texture,
                                mip_level: mip,
                                origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                                aspect: wgpu::TextureAspect::All,
                            },
                            &level,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(size * 4),
                                rows_per_image: Some(size),
                            },
                            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                        );
                        if mip + 1 < mip_levels {
                            level = downsample_rgba(&level, size, &lut);
                            size /= 2;
                        }
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.store(250 + 750 * d / n_tiles, Ordering::Relaxed);
                    ctx.request_repaint();
                }
            });
        }
    });
    queue.submit([]);
    texture
}

/// Read back a few texels of the last tile and compare them with the source.
fn verify(
    tex: &wgpu::Texture,
    rgb: &image::RgbImage,
    layout: &TileLayout,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> bool {
    let layer = layout.tiles_x * layout.tiles_y - 1;
    let (tx, ty) = (layer % layout.tiles_x, layer / layout.tiles_x);
    let (w, h) = (rgb.width() as i64, rgb.height() as i64);
    let probes: Vec<(u32, u32)> = [(0u32, 0u32), (7, 3), (13, 29), (2, 17)]
        .iter()
        .map(|&(a, b)| (GUTTER + a, GUTTER + b))
        .collect();
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verify readback"),
        size: 256 * probes.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    for (i, &(x, y)) in probes.iter().enumerate() {
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 256 * i as u64, bytes_per_row: Some(256), rows_per_image: None },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
    }
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (stx, srx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = stx.send(r.is_ok());
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) });
    if !srx.recv().unwrap_or(false) {
        return false;
    }
    let Ok(data) = slice.get_mapped_range() else { return false };
    let ok = probes.iter().enumerate().all(|(i, &(x, y))| {
        let sx = ((tx * layout.tile_content + x - GUTTER) as i64).rem_euclid(w) as u32;
        let sy = ((ty * layout.tile_content + y - GUTTER) as i64).clamp(0, h - 1) as u32;
        let p = rgb.get_pixel(sx, sy).0;
        data[256 * i..256 * i + 3] == p && data[256 * i + 3] == 255
    });
    drop(data);
    buf.unmap();
    ok
}

#[derive(Clone, Copy, Debug)]
struct TileLayout {
    tile_size: u32,
    tile_content: u32,
    tiles_x: u32,
    tiles_y: u32,
}

impl TileLayout {
    fn new(w: u32, h: u32, max_tile: u32) -> Self {
        let align = 1 << MAX_LOD;
        let max_content = (max_tile - 2 * GUTTER) / align * align;
        let content = w.max(h).div_ceil(align).saturating_mul(align).min(max_content);
        let tile_size = content + 2 * GUTTER;
        Self { tile_size, tile_content: content, tiles_x: w.div_ceil(content), tiles_y: h.div_ceil(content) }
    }
    fn layers(&self) -> u32 {
        let n = self.tiles_x * self.tiles_y;
        // wgpu's GL backend infers the view type from the layer count: 1 layer
        // means a plain 2D texture and multiples of 6 mean cube maps. Pad the
        // array with an unused layer so it is always treated as a 2D array.
        if n == 1 || n.is_multiple_of(6) { n + 1 } else { n }
    }
}

fn build_tile(rgb: &image::RgbImage, w: u32, h: u32, l: &TileLayout, tx: u32, ty: u32) -> Vec<u8> {
    let t = l.tile_size as usize;
    let mut out = vec![255u8; t * t * 4];
    let src = rgb.as_raw();
    let x0 = (tx * l.tile_content) as i64 - GUTTER as i64;
    let y0 = (ty * l.tile_content) as i64 - GUTTER as i64;
    // Precompute wrapped source columns.
    let cols: Vec<usize> = (0..t as i64).map(|i| (x0 + i).rem_euclid(w as i64) as usize).collect();
    for j in 0..t {
        let sy = (y0 + j as i64).clamp(0, h as i64 - 1) as usize;
        let srow = &src[sy * w as usize * 3..(sy + 1) * w as usize * 3];
        let drow = &mut out[j * t * 4..(j + 1) * t * 4];
        for (i, &sx) in cols.iter().enumerate() {
            drow[i * 4] = srow[sx * 3];
            drow[i * 4 + 1] = srow[sx * 3 + 1];
            drow[i * 4 + 2] = srow[sx * 3 + 2];
        }
    }
    out
}

/// sRGB ↔ linear lookup tables for gamma-correct mip generation.
struct Lut {
    to_lin: [f32; 256],
    to_srgb: Vec<u8>, // 4096 entries over [0,1]
}

impl Lut {
    fn new() -> Self {
        let mut to_lin = [0f32; 256];
        for (i, v) in to_lin.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        }
        let to_srgb = (0..4096)
            .map(|i| {
                let l = i as f32 / 4095.0;
                let s = if l <= 0.0031308 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
                (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            })
            .collect();
        Self { to_lin, to_srgb }
    }
    #[inline]
    fn enc(&self, l: f32) -> u8 {
        self.to_srgb[(l * 4095.0 + 0.5) as usize]
    }
}

fn downsample_rgba(src: &[u8], size: u32, lut: &Lut) -> Vec<u8> {
    let s = size as usize;
    let d = s / 2;
    let mut out = vec![255u8; d * d * 4];
    for y in 0..d {
        let r0 = &src[(2 * y) * s * 4..(2 * y + 1) * s * 4];
        let r1 = &src[(2 * y + 1) * s * 4..(2 * y + 2) * s * 4];
        let o = &mut out[y * d * 4..(y + 1) * d * 4];
        for x in 0..d {
            for c in 0..3 {
                let a = lut.to_lin[r0[8 * x + c] as usize]
                    + lut.to_lin[r0[8 * x + 4 + c] as usize]
                    + lut.to_lin[r1[8 * x + c] as usize]
                    + lut.to_lin[r1[8 * x + 4 + c] as usize];
                o[4 * x + c] = lut.enc(a * 0.25);
            }
        }
    }
    out
}

/// 2× box downscale of an RGB image (used only when it does not fit on the GPU).
fn halve(src: &image::RgbImage) -> image::RgbImage {
    let (w, h) = (src.width(), src.height());
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = image::RgbImage::new(nw, nh);
    let s = src.as_raw();
    let o: &mut [u8] = &mut out;
    for y in 0..nh as usize {
        let y0 = (2 * y).min(h as usize - 1);
        let y1 = (2 * y + 1).min(h as usize - 1);
        for x in 0..nw as usize {
            let x0 = (2 * x).min(w as usize - 1);
            let x1 = (2 * x + 1).min(w as usize - 1);
            for c in 0..3 {
                let v = s[(y0 * w as usize + x0) * 3 + c] as u32
                    + s[(y0 * w as usize + x1) * 3 + c] as u32
                    + s[(y1 * w as usize + x0) * 3 + c] as u32
                    + s[(y1 * w as usize + x1) * 3 + c] as u32;
                o[(y * nw as usize + x) * 3 + c] = ((v + 2) / 4) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_alignment() {
        for &(w, h) in &[(100, 50), (8000, 4000), (40000, 20000), (12345, 3000)] {
            let l = TileLayout::new(w, h, 4096);
            assert_eq!(l.tile_content % 32, 0);
            assert!(l.tile_size <= 4096);
            assert!(l.tiles_x * l.tile_content >= w && l.tiles_y * l.tile_content >= h);
        }
    }

    #[test]
    fn tile_wraps_horizontally() {
        let mut img = image::RgbImage::new(64, 32);
        img.put_pixel(63, 0, image::Rgb([9, 8, 7]));
        let l = TileLayout::new(64, 32, 4096);
        let t = build_tile(&img, 64, 32, &l, 0, 0);
        // Pixel just left of x=0 in the gutter is the image's last column.
        let i = ((GUTTER as usize) * l.tile_size as usize + (GUTTER as usize - 1)) * 4;
        assert_eq!(&t[i..i + 3], &[9, 8, 7]);
    }
}
