// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Loading: read → decode → colour-manage → build the resolution pyramid →
//! allocate the GPU tile cache and upload the coarsest level. Finer tiles are
//! streamed in while viewing (see `tiles.rs`).
//!
//! The pipeline is one async function. Natively it runs on a background thread
//! (and uses worker threads for the heavy parts); in the browser it runs on the
//! main thread and yields between steps so the page stays responsive.

use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use eframe::wgpu;
use image::{ImageDecoder, ImageReader};
use web_time::Instant;

use crate::fsv::{PanoParams, decode_text};
use crate::settings::{MonitorProfile, Settings};
use crate::source::Source;
use crate::tiles::{GpuTiles, Lut, Pyramid, TileCache, TileGrid};
use crate::xmp::GPano;

/// The browser build has a 4 GB address space; refuse images that cannot fit.
#[cfg(target_arch = "wasm32")]
const MAX_WEB_PIXELS: u64 = 300_000_000;

pub struct LoadedPano {
    /// What was opened: the image itself or the tour file.
    pub source: Source,
    pub params: PanoParams,
    pub gpano: GPano,
    pub width: u32,
    pub height: u32,
    /// Multi-resolution tiles: CPU pyramid plus the GPU cache.
    pub tiles: TileCache,
    pub icc_used: Option<String>,
    pub load_time: f32,
}

impl LoadedPano {
    /// The displayed image (for a tour, the image it points to).
    pub fn image_source(&self) -> &Source {
        self.params.image.as_ref().unwrap_or(&self.source)
    }
}

#[derive(Clone)]
pub struct Overrides {
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub hfov: Option<f64>,
}

pub struct LoadJob {
    pub name: String,
    pub progress: Arc<AtomicU32>, // 0..=1000
    pub stage: Arc<Mutex<String>>,
    pub cancel: Arc<AtomicBool>,
    pub rx: mpsc::Receiver<Result<LoadedPano, String>>,
    pub overrides: Option<Overrides>,
}

pub fn start_load(
    source: Source,
    settings: &Settings,
    device: wgpu::Device,
    queue: wgpu::Queue,
    ctx: eframe::egui::Context,
    overrides: Option<Overrides>,
) -> LoadJob {
    let progress = Arc::new(AtomicU32::new(0));
    let stage = Arc::new(Mutex::new(tr!("Reading").to_string()));
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let s = settings.clone();
    let name = source.file_name();
    let (p2, st2, c2) = (progress.clone(), stage.clone(), cancel.clone());
    let job = async move {
        let r = load(&source, &s, &device, &queue, &p2, &st2, &c2, &ctx).await;
        let _ = tx.send(r);
        ctx.request_repaint();
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new()
        .name("pano-loader".into())
        .spawn(move || pollster::block_on(job))
        .expect("spawn loader thread");
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(job);
    LoadJob { name, progress, stage, cancel, rx, overrides }
}

/// Image file extensions the viewer opens (lower case).
pub const IMAGE_EXTENSIONS: &[&str] =
    &["jpg", "jpeg", "jpe", "tif", "tiff", "png", "webp", "bmp", "jxl"];

pub fn is_supported_ext(ext: &str) -> bool {
    ext == "fsv" || IMAGE_EXTENSIONS.contains(&ext)
}

pub fn is_supported(path: &Path) -> bool {
    path.extension().is_some_and(|e| is_supported_ext(&e.to_string_lossy().to_lowercase()))
}

/// Let other work run (browser only; native loading has its own thread).
async fn yield_now() {
    #[cfg(target_arch = "wasm32")]
    crate::web::yield_now().await;
}

/// Run `f(chunk, offset)` over chunks of `data` in parallel where threads
/// are available.
pub(crate) fn par_chunks(data: &mut [u8], chunk: usize, f: impl Fn(&mut [u8], usize) + Sync) {
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|s| {
        for (i, c) in data.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || f(c, i * chunk));
        }
    });
    #[cfg(target_arch = "wasm32")]
    for (i, c) in data.chunks_mut(chunk).enumerate() {
        f(c, i * chunk);
    }
}

pub(crate) fn threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    return std::thread::available_parallelism().map_or(4, |n| n.get());
    #[cfg(target_arch = "wasm32")]
    1
}

fn is_jxl(data: &[u8]) -> bool {
    data.starts_with(&[0xFF, 0x0A]) || data.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
}

fn check_size(_w: u32, _h: u32) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    if _w as u64 * _h as u64 > MAX_WEB_PIXELS {
        return Err(tr!(
            "This panorama ({w}×{h}) is too large for the browser version.",
            w = _w,
            h = _h
        ));
    }
    Ok(())
}

/// Decode any supported image, returning the pixels and the embedded ICC profile.
fn decode(data: &[u8]) -> Result<(image::DynamicImage, Option<Vec<u8>>), String> {
    let err = |e: &dyn std::fmt::Display| tr!("Cannot decode this image: {error}", error = e);
    if is_jxl(data) {
        let mut decoder = jxl_oxide::integration::JxlDecoder::new(Cursor::new(data)).map_err(|e| err(&e))?;
        let (w, h) = decoder.dimensions();
        check_size(w, h)?;
        let icc = decoder.icc_profile().ok().flatten();
        let img = image::DynamicImage::from_decoder(decoder).map_err(|e| err(&e))?;
        Ok((img, icc))
    } else {
        let mut reader = ImageReader::new(Cursor::new(data)).with_guessed_format().map_err(|e| e.to_string())?;
        reader.no_limits();
        let mut decoder = reader.into_decoder().map_err(|e| err(&e))?;
        let (w, h) = decoder.dimensions();
        check_size(w, h)?;
        let icc = decoder.icc_profile().ok().flatten();
        let img = image::DynamicImage::from_decoder(decoder).map_err(|e| err(&e))?;
        Ok((img, icc))
    }
}

#[allow(clippy::too_many_arguments)]
async fn load(
    source: &Source,
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
    let cancelled = || cancel.load(Ordering::Relaxed);

    let (mut params, image_source, data) = if source.is_fsv() {
        let text = decode_text(&source.read().await?);
        let mut params = PanoParams::parse_fsv(&text)?;
        let image = source.resolve(&params.image_name)?;
        let data = image.read().await?;
        params.image = Some(image.clone());
        (params, image, data)
    } else {
        (PanoParams::default(), source.clone(), source.read().await?)
    };
    let gpano = GPano::from_bytes(&data, &image_source.extension());
    if !source.is_fsv() {
        params = PanoParams::from_image(image_source, &gpano);
    }
    progress.store(50, Ordering::Relaxed);

    set_stage(tr!("Decoding"));
    yield_now().await;
    let (img, icc) = decode(&data)?;
    drop(data);
    if cancelled() {
        return Err("cancelled".into());
    }
    let (width, height) = (img.width(), img.height());
    let mut rgb = img.into_rgb8();
    progress.store(150, Ordering::Relaxed);

    // ---- colour management -------------------------------------------------
    let mut icc_used = None;
    if settings.enable_color_management {
        set_stage(tr!("Colour management"));
        yield_now().await;
        let output = match settings.monitor_profile {
            MonitorProfile::File if !settings.monitor_profile_file.is_empty() => {
                std::fs::read(&settings.monitor_profile_file)
                    .ok()
                    .and_then(|b| qcms::Profile::new_from_slice(&b, false))
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
                let row = width as usize * 3;
                let rows_per = (height as usize).div_ceil(threads()).max(1);
                par_chunks(&mut rgb, rows_per * row, |c, _| xf.apply(c));
                icc_used = Some(if icc.is_some() { tr!("embedded ICC").into() } else { tr!("sRGB (assumed)").into() });
            }
        }
    }
    progress.store(250, Ordering::Relaxed);

    // ---- resolution pyramid ----------------------------------------------------
    set_stage(tr!("Building resolution levels"));
    yield_now().await;
    let grid = TileGrid::new(width, height);
    let lut = Lut::new();
    let levels = grid.levels.len();
    let pyramid = Pyramid::build(rgb, &grid, &lut, |i| {
        progress.store(250 + (450 * i / levels) as u32, Ordering::Relaxed);
        ctx.request_repaint();
    });
    if cancelled() {
        return Err("cancelled".into());
    }
    progress.store(700, Ordering::Relaxed);

    // ---- GPU tile cache ------------------------------------------------------
    set_stage(tr!("Uploading to GPU"));
    yield_now().await;
    let max_layers = device.limits().max_texture_array_layers;
    let mut budget = settings.max_gpu_memory_mb.max(64) as u64 * 1024 * 1024;
    let mut slots = grid.slots_for_budget(budget, max_layers);
    let gpu = loop {
        let gpu = GpuTiles::allocate(device, grid.clone(), width, height, slots);
        if verify(&gpu, device, queue) {
            break gpu;
        }
        // Some drivers (notably OpenGL) fail texture allocation silently when
        // out of video memory: the readback did not match, so use fewer slots.
        budget /= 2;
        let fewer = grid.slots_for_budget(budget, max_layers);
        log::warn!("GPU tile cache with {slots} slots failed verification (out of video memory?), trying {fewer}");
        gpu.texture.destroy();
        if fewer >= slots {
            break GpuTiles::allocate(device, grid.clone(), width, height, slots);
        }
        slots = fewer;
    };
    log::info!(
        "{width}x{height}: {} levels, tiles {}x{}, {} of {} tiles cached on the GPU ({} MB)",
        grid.levels.len(),
        grid.cx,
        grid.cy,
        slots,
        grid.tile_count(),
        (slots as u64 * grid.slot_bytes()) >> 20
    );
    let tiles = TileCache::new(gpu, pyramid, lut, queue, ctx);
    progress.store(1000, Ordering::Relaxed);

    Ok(LoadedPano {
        source: source.clone(),
        params,
        gpano,
        width,
        height,
        tiles,
        icc_used,
        load_time: t0.elapsed().as_secs_f32(),
    })
}

/// The browser cannot wait for a GPU readback; WebGPU reports allocation
/// failures itself, so the check is skipped there.
#[cfg(target_arch = "wasm32")]
fn verify(_: &GpuTiles, _: &wgpu::Device, _: &wgpu::Queue) -> bool {
    true
}

/// Write a texel into the last cache slot and read it back.
#[cfg(not(target_arch = "wasm32"))]
fn verify(gpu: &GpuTiles, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    let probe = [11u8, 222, 33, 255];
    let layer = gpu.slots - 1;
    let at = wgpu::TexelCopyTextureInfo {
        texture: &gpu.texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 7, y: 5, z: layer },
        aspect: wgpu::TextureAspect::All,
    };
    let one = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
    queue.write_texture(at, &probe, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: None }, one);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verify readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        at,
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None },
        },
        one,
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r.is_ok());
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) });
    if !rx.recv().unwrap_or(false) {
        return false;
    }
    let Ok(data) = slice.get_mapped_range() else { return false };
    let ok = data[..4] == probe;
    drop(data);
    buf.unmap();
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_chunks_passes_offsets() {
        let mut data = vec![0u8; 10];
        par_chunks(&mut data, 4, |c, off| c.iter_mut().enumerate().for_each(|(i, b)| *b = (off + i) as u8));
        assert_eq!(data, (0..10).collect::<Vec<u8>>());
    }

    #[test]
    fn detects_jpeg_xl() {
        assert!(is_jxl(&[0xFF, 0x0A, 1]));
        assert!(!is_jxl(&[0xFF, 0xD8, 0xFF]));
    }
}
