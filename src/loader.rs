// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Loading a panorama.
//!
//! 1. Work out a cache key from the file's identity and the colour settings.
//! 2. If the tile store has the panorama, use it: the original file is not
//!    read at all.
//! 3. Otherwise read the image in stripes (`pyramid.rs`), colour-manage each
//!    stripe, cut all pyramid levels into tiles and store them as JPEG
//!    (`store.rs`). The decoded image is never kept as a whole (except for
//!    formats whose decoders can only produce the whole image).
//! 4. Allocate the GPU tile cache and upload the coarsest level; finer tiles
//!    are streamed in while viewing (`tiles.rs`).
//!
//! Natively this runs on a background thread (with worker threads for
//! encoding); in the browser it runs on the main thread and yields regularly
//! so the page stays responsive.

use std::io::BufReader;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use eframe::wgpu;
use web_time::Instant;

use crate::fsv::{PanoParams, decode_text};
use crate::pyramid::{Input, PyramidBuilder, ReadSeek, open_rows};
use crate::settings::{MonitorProfile, Settings};
use crate::source::Source;
use crate::store::{self, CacheMeta, FORMAT_VERSION, TileStore};
use crate::tiles::{GpuTiles, Lut, TileCache, TileGrid, TileId, slot_mips};
use crate::xmp::GPano;

pub struct LoadedPano {
    /// What was opened: the image itself or the tour file.
    pub source: Source,
    pub params: PanoParams,
    pub gpano: GPano,
    pub width: u32,
    pub height: u32,
    /// Multi-resolution tiles: tile store plus the GPU cache.
    pub tiles: TileCache,
    pub icc_used: Option<String>,
    pub load_time: f32,
    /// The tiles came from the tile cache (the image was not decoded).
    pub from_cache: bool,
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
    // The future is created on the thread that runs it, so it need not be Send.
    let run = move || async move {
        let r = load(&source, &s, &device, &queue, &p2, &st2, &c2, &ctx).await;
        let _ = tx.send(r);
        ctx.request_repaint();
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new()
        .name("pano-loader".into())
        .spawn(move || pollster::block_on(run()))
        .expect("spawn loader thread");
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(run());
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

// ---------------------------------------------------------------------------
// Cache key

/// What makes the cached pixels differ besides the source file.
fn colour_signature(settings: &Settings) -> String {
    if !settings.enable_color_management {
        return "cm=off".into();
    }
    match settings.monitor_profile {
        MonitorProfile::Srgb => "cm=srgb".into(),
        MonitorProfile::File => {
            let p = &settings.monitor_profile_file;
            let stamp = std::fs::metadata(p).ok().and_then(|m| m.modified().ok()).map(|t| format!("{t:?}"));
            format!("cm=file:{p}:{stamp:?}")
        }
    }
}

/// Identity of the image file, without reading it where possible.
async fn source_identity(src: &Source) -> Result<String, String> {
    match src {
        Source::Path(p) => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let m = std::fs::metadata(p).map_err(|_| tr!("No such file: {path}", path = p.display()))?;
                let path = p.canonicalize().unwrap_or_else(|_| p.clone());
                Ok(format!("path:{}|{}|{:?}", path.display(), m.len(), m.modified().ok()))
            }
            #[cfg(target_arch = "wasm32")]
            Err(tr!("No such file: {path}", path = p.display()))
        }
        Source::Memory { name, data, .. } => Ok(format!("mem:{name}|{}|{:x}", data.len(), store::hash_bytes(data))),
        Source::Url(u) => {
            #[cfg(target_arch = "wasm32")]
            let fp = crate::web::fingerprint(u).await.unwrap_or_default();
            #[cfg(not(target_arch = "wasm32"))]
            let fp = String::new();
            Ok(format!("url:{u}|{fp}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline

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

    let mut params = PanoParams::default();
    let image_source = if source.is_fsv() {
        let text = decode_text(&source.read().await?);
        params = PanoParams::parse_fsv(&text)?;
        let image = source.resolve(&params.image_name)?;
        params.image = Some(image.clone());
        image
    } else {
        source.clone()
    };

    let identity = source_identity(&image_source).await?;
    let key = format!(
        "{:016x}",
        store::hash_str(&format!("v{FORMAT_VERSION}|{identity}|{}", colour_signature(settings)))
    );
    let (tile_store, meta, from_cache) = match store::lookup(&key).await {
        Some((s, m)) => {
            set_stage(tr!("Loading from tile cache"));
            (s, m, true)
        }
        None => {
            let (s, m) = build_tiles(&image_source, &key, settings, progress, &set_stage, cancel, ctx).await?;
            (s, m, false)
        }
    };
    let (width, height) = (meta.width, meta.height);
    let gpano = GPano { tags: meta.gpano.clone() };
    if !source.is_fsv() {
        params = PanoParams::from_image(image_source, &gpano);
    }
    progress.store(900, Ordering::Relaxed);

    // ---- GPU tile cache ------------------------------------------------------
    set_stage(tr!("Uploading to GPU"));
    yield_now().await;
    let grid = TileGrid::new(width, height);
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
        "{width}x{height}: {} levels, tiles {}x{}, {} of {} tiles cached on the GPU ({} MB), {}",
        grid.levels.len(),
        grid.cx,
        grid.cy,
        slots,
        grid.tile_count(),
        (slots as u64 * grid.slot_bytes()) >> 20,
        if from_cache { "tiles from cache" } else { "tiles built" }
    );

    // The always-resident base level.
    let lut = Lut::new();
    let b = grid.base();
    let mut base = Vec::new();
    for ty in 0..grid.levels[b].ny {
        for tx in 0..grid.levels[b].nx {
            let t = TileId { level: b as u8, tx: tx as u16, ty: ty as u16 };
            let jpeg = tile_store.get(t).await.map_err(|e| format!("{}: {e}", tr!("Tile cache is damaged")))?;
            let rgb = store::decode_tile(&jpeg)?;
            base.push((t, slot_mips(&rgb, &grid, &lut)));
        }
    }
    let tiles = TileCache::new(gpu, tile_store, base, lut, queue, ctx);
    progress.store(1000, Ordering::Relaxed);

    Ok(LoadedPano {
        source: source.clone(),
        params,
        gpano,
        width,
        height,
        tiles,
        icc_used: meta.icc_used,
        load_time: t0.elapsed().as_secs_f32(),
        from_cache,
    })
}

/// Open the image file for reading, plus its GPano tags.
async fn open_input(src: &Source) -> Result<(Box<dyn FnMut() -> Result<Input, String>>, GPano), String> {
    let ext = src.extension();
    #[cfg(not(target_arch = "wasm32"))]
    if let Source::Path(p) = src {
        // Read straight from the file: huge images never have to fit in memory.
        let p = p.clone();
        let gpano = GPano::read(&p);
        let open = move || -> Result<Input, String> {
            let f = std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            Ok(BufReader::with_capacity(1 << 20, Box::new(f) as Box<dyn ReadSeek>))
        };
        return Ok((Box::new(open), gpano));
    }
    let data = src.read().await?;
    let gpano = GPano::from_bytes(&data, &ext);
    let open = move || -> Result<Input, String> {
        Ok(BufReader::with_capacity(1 << 20, Box::new(std::io::Cursor::new(data.clone())) as Box<dyn ReadSeek>))
    };
    Ok((Box::new(open), gpano))
}

/// Colour transform from the image's profile to the display profile.
fn colour_transform(icc: Option<&[u8]>, settings: &Settings) -> (Option<qcms::Transform>, Option<String>) {
    if !settings.enable_color_management {
        return (None, None);
    }
    let output = match settings.monitor_profile {
        MonitorProfile::File if !settings.monitor_profile_file.is_empty() => {
            std::fs::read(&settings.monitor_profile_file).ok().and_then(|b| qcms::Profile::new_from_slice(&b, false))
        }
        _ => None,
    };
    let input = icc.and_then(|b| qcms::Profile::new_from_slice(b, false));
    if input.is_none() && output.is_none() {
        return (None, None); // sRGB to sRGB
    }
    let used = if input.is_some() { tr!("embedded ICC") } else { tr!("sRGB (assumed)") };
    let input = input.unwrap_or_else(qcms::Profile::new_sRGB);
    let mut output = output.unwrap_or_else(qcms::Profile::new_sRGB);
    output.precache_output_transform();
    let xf = qcms::Transform::new(&input, &output, qcms::DataType::RGB8, qcms::Intent::Perceptual);
    let used = xf.as_ref().map(|_| used.to_string());
    (xf, used)
}

/// Decode the image stripe by stripe and write all tiles to a new cache.
#[allow(clippy::too_many_arguments)]
async fn build_tiles(
    src: &Source,
    key: &str,
    settings: &Settings,
    progress: &AtomicU32,
    set_stage: &impl Fn(&str),
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
) -> Result<(TileStore, CacheMeta), String> {
    let (open, gpano) = open_input(src).await?;
    set_stage(tr!("Building tiles"));
    yield_now().await;
    let mut decoded = open_rows(open)?;
    let (w, h) = (decoded.width, decoded.height);
    let (xf, icc_used) = colour_transform(decoded.icc.as_deref(), settings);
    let grid = TileGrid::new(w, h);
    let lut = Lut::new();
    let mut builder = PyramidBuilder::new(grid.clone());
    let building = store::begin(key).await;
    log::info!(
        "Building tiles for {w}x{h} ({}), {} tiles",
        if decoded.streaming { "read in stripes" } else { "decoded at once" },
        grid.tile_count()
    );

    // About 8 MB of rows per step.
    let batch = ((8 << 20) / (w as usize * 3)).clamp(8, 512);
    let mut rows = Vec::with_capacity(batch * w as usize * 3);
    let mut done = 0usize;
    let mut bytes = 0u64;
    // Time spent per step, logged at the end.
    let (mut t_read, mut t_build, mut t_store) = (0f64, 0f64, 0f64);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        rows.clear();
        let t = Instant::now();
        let n = decoded.rows.read(batch, &mut rows)?;
        t_read += t.elapsed().as_secs_f64();
        if n == 0 {
            break;
        }
        if let Some(xf) = &xf {
            let per = n.div_ceil(threads()).max(1) * w as usize * 3;
            par_chunks(&mut rows, per, |c, _| xf.apply(c));
        }
        let t = Instant::now();
        builder.push_rows(&rows, &lut);
        let tiles = builder.take_tiles();
        t_build += t.elapsed().as_secs_f64();
        if !tiles.is_empty() {
            let t = Instant::now();
            bytes += store_tiles(&building.store, tiles, &grid, cancel).await?;
            t_store += t.elapsed().as_secs_f64();
        }
        done += n;
        progress.store(50 + (850 * done as u64 / h as u64) as u32, Ordering::Relaxed);
        ctx.request_repaint();
        yield_now().await;
    }
    drop(decoded);
    log::info!("Tiles built: decoding {t_read:.2} s, pyramid {t_build:.2} s, encoding and storing {t_store:.2} s");
    if !builder.is_complete() {
        return Err(tr!("Cannot decode this image: {error}", error = "image data ended early"));
    }
    let meta = CacheMeta { version: FORMAT_VERSION, width: w, height: h, gpano: gpano.tags, icc_used, bytes };
    let store = store::finish(building, &meta, settings.tile_cache_mb).await;
    Ok((store, meta))
}

/// Encode tiles as JPEG and store them; returns the bytes written.
async fn store_tiles(store: &TileStore, tiles: Vec<(TileId, Vec<u8>)>, grid: &TileGrid, cancel: &AtomicBool) -> Result<u64, String> {
    let (w, h) = (grid.sx, grid.sy);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let next = AtomicU32::new(0);
        let bytes = std::sync::atomic::AtomicU64::new(0);
        let error = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads().min(tiles.len()) {
                s.spawn(|| {
                    while !cancel.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                        let Some((t, rgb)) = tiles.get(i) else { return };
                        let r = store::encode_tile(rgb, w, h).and_then(|jpeg| {
                            bytes.fetch_add(jpeg.len() as u64, Ordering::Relaxed);
                            store.put_blocking(*t, jpeg)
                        });
                        if let Err(e) = r {
                            *error.lock().unwrap() = Some(e);
                            return;
                        }
                    }
                });
            }
        });
        if let Some(e) = error.into_inner().unwrap() {
            return Err(e);
        }
        Ok(bytes.into_inner())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cancel;
        let mut bytes = 0;
        for (t, rgb) in tiles {
            let jpeg = store::encode_tile(&rgb, w, h)?;
            bytes += jpeg.len() as u64;
            store.put(t, jpeg).await?;
            yield_now().await;
        }
        Ok(bytes)
    }
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

}
