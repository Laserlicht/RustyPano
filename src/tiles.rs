// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Multi-resolution tile streaming.
//!
//! The panorama is a pyramid (full resolution, half, quarter, …) whose levels
//! are cut into tiles of `cx × cy` pixels (see `pyramid.rs`). The tiles live
//! in the tile store (`store.rs`: disk, browser cache or memory), not in RAM.
//! On the GPU there is only a fixed-size cache: one `texture_2d_array` whose
//! layers ("slots") hold one tile each, plus a page table texture that maps
//! (level, tile x, tile y) to a slot.
//!
//! The coarsest level ("base") is always resident, so something can always be
//! drawn. Every frame the visible tiles at the resolution the screen needs are
//! determined on the CPU and missing ones are loaded from the store and
//! uploaded (natively by worker threads, in the browser asynchronously); the
//! least recently used tiles are evicted. The shader picks the finest resident level
//! for each pixel and falls back to coarser ones while tiles are streaming in.
//!
//! Each slot has a gutter of `GUTTER` real neighbouring pixels (wrapping
//! horizontally, clamped vertically) and its own mip chain, so filtering never
//! shows seams.

use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, mpsc};
use std::sync::atomic::{AtomicU32, Ordering};

use eframe::wgpu;

use crate::store::TileStore;
use crate::view::View;

pub const GUTTER: u32 = 32;
/// Mip levels inside a slot (0..=5); the gutter is still 1 texel at level 5.
pub const MIP_LEVELS: u32 = 6;
pub const MAX_LOD: u32 = MIP_LEVELS - 1;
/// Pyramid levels the shader can address.
pub const MAX_LEVELS: usize = 16;
/// The base level is the first one with at most this many tiles.
const BASE_TILES: u32 = 4;
/// More slots than this are never needed, even on 4K displays.
const MAX_SLOTS: u32 = 192;
/// Minimum number of streaming slots besides the base level.
const MIN_STREAM_SLOTS: u32 = 24;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileId {
    pub level: u8,
    pub tx: u16,
    pub ty: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelInfo {
    /// Size of this pyramid level in pixels.
    pub w: u32,
    pub h: u32,
    /// Tiles in each direction.
    pub nx: u32,
    pub ny: u32,
    /// First row of this level in the page table.
    pub row: u32,
}

/// How the pyramid is cut into tiles. Tile sizes are chosen per image so
/// that the finest levels divide (almost) evenly.
#[derive(Clone, Debug)]
pub struct TileGrid {
    /// Tile content size.
    pub cx: u32,
    pub cy: u32,
    /// Slot size including the gutter on both sides.
    pub sx: u32,
    pub sy: u32,
    pub levels: Vec<LevelInfo>,
    pub table_w: u32,
    pub table_h: u32,
}

const ALIGN: u32 = 1 << MAX_LOD;

fn half(n: u32) -> u32 {
    n.div_ceil(2).max(1)
}

fn round_up(n: u32, a: u32) -> u32 {
    n.div_ceil(a) * a
}

/// Tile content size along one axis: a multiple of `ALIGN` between 512 and
/// 1024 that wastes the least space on the two finest levels.
fn best_content(d0: u32) -> u32 {
    if d0 <= 1024 {
        return round_up(d0, ALIGN);
    }
    let d1 = half(d0);
    let waste = |d: u32, c: u32| (d.div_ceil(c) * c - d) as f64 / d as f64;
    (512..=1024)
        .step_by(ALIGN as usize)
        .min_by(|&a, &b| {
            let wa = 2.0 * waste(d0, a) + waste(d1, a);
            let wb = 2.0 * waste(d0, b) + waste(d1, b);
            // Prefer bigger tiles (fewer uploads) when equally good.
            wa.partial_cmp(&wb).unwrap().then(b.cmp(&a))
        })
        .unwrap()
}

impl TileGrid {
    pub fn new(w: u32, h: u32) -> Self {
        let (cx, cy) = (best_content(w), best_content(h));
        let mut levels = Vec::new();
        let (mut lw, mut lh, mut row) = (w, h, 0);
        loop {
            let (nx, ny) = (lw.div_ceil(cx), lh.div_ceil(cy));
            levels.push(LevelInfo { w: lw, h: lh, nx, ny, row });
            row += ny;
            if nx * ny <= BASE_TILES || levels.len() == MAX_LEVELS || (lw == 1 && lh == 1) {
                break;
            }
            (lw, lh) = (half(lw), half(lh));
        }
        let table_w = levels[0].nx;
        Self { cx, cy, sx: cx + 2 * GUTTER, sy: cy + 2 * GUTTER, levels, table_w, table_h: row }
    }

    pub fn base(&self) -> usize {
        self.levels.len() - 1
    }

    pub fn tile_count(&self) -> u32 {
        self.levels.iter().map(|l| l.nx * l.ny).sum()
    }

    pub fn base_tiles(&self) -> u32 {
        let b = self.levels[self.base()];
        b.nx * b.ny
    }

    /// GPU bytes per slot including mipmaps.
    pub fn slot_bytes(&self) -> u64 {
        self.sx as u64 * self.sy as u64 * 4 * 4 / 3
    }

    pub fn mip_levels(&self) -> u32 {
        MIP_LEVELS.min(32 - self.sx.min(self.sy).leading_zeros())
    }

    fn table_index(&self, t: TileId) -> usize {
        let l = self.levels[t.level as usize];
        ((l.row + t.ty as u32) * self.table_w + t.tx as u32) as usize
    }

    /// Number of slots for a memory budget: never more than useful, never
    /// fewer than needed to stream, and never a count that wgpu's GL backend
    /// would mistake for a 2D texture or a cube map.
    pub fn slots_for_budget(&self, budget_bytes: u64, max_layers: u32) -> u32 {
        let pinned = self.base_tiles();
        let want = (budget_bytes / self.slot_bytes()) as u32;
        let mut n = want
            .min(MAX_SLOTS)
            .min(self.tile_count())
            .max(pinned + MIN_STREAM_SLOTS.min(self.tile_count() - pinned))
            .min(max_layers);
        if n == 1 || n.is_multiple_of(6) {
            n = if n > pinned + 1 { n - 1 } else { n + 1 };
        }
        n.max(pinned)
    }
}

// ---------------------------------------------------------------------------
// Tile pixels

/// sRGB ↔ linear lookup tables for gamma-correct downsampling.
pub struct Lut {
    to_lin: [f32; 256],
    to_srgb: Vec<u8>, // 4096 entries over [0,1]
}

impl Lut {
    pub fn new() -> Self {
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
    #[inline]
    pub(crate) fn avg4(&self, a: u8, b: u8, c: u8, d: u8) -> u8 {
        let t = &self.to_lin;
        self.enc((t[a as usize] + t[b as usize] + t[c as usize] + t[d as usize]) * 0.25)
    }
}

/// Gamma-correct 2× box downscale of a whole RGB image (odd sizes round up).
/// The streaming builder does the same row by row; this is the reference.
#[cfg(test)]
pub(crate) fn halve(src: &image::RgbImage, lut: &Lut) -> image::RgbImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = image::RgbImage::new(nw as u32, nh as u32);
    let s = src.as_raw();
    for y in 0..nh {
        let (a, b) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
        for x in 0..nw {
            let (p, q) = ((2 * x).min(w - 1), (2 * x + 1).min(w - 1));
            for c in 0..3 {
                let v = lut.avg4(s[(a * w + p) * 3 + c], s[(a * w + q) * 3 + c], s[(b * w + p) * 3 + c], s[(b * w + q) * 3 + c]);
                out.get_pixel_mut(x as u32, y as u32).0[c] = v;
            }
        }
    }
    out
}

/// RGBA mip chain for a cache slot from a tile's RGB8 pixels (incl. gutter).
pub fn slot_mips(rgb: &[u8], grid: &TileGrid, lut: &Lut) -> Vec<Vec<u8>> {
    let (sx, sy) = (grid.sx as usize, grid.sy as usize);
    let mut level = vec![255u8; sx * sy * 4];
    for (d, s) in level.as_chunks_mut::<4>().0.iter_mut().zip(rgb.as_chunks::<3>().0) {
        d[..3].copy_from_slice(s);
    }
    let n = grid.mip_levels() as usize;
    let mut mips = Vec::with_capacity(n);
    let (mut mw, mut mh) = (sx, sy);
    mips.push(level);
    for _ in 1..n {
        let next = downsample_rgba(mips.last().unwrap(), mw, mh, lut);
        mw /= 2;
        mh /= 2;
        mips.push(next);
    }
    mips
}

/// Mip chain of a tile ready for upload, or `None` if it could not be loaded.
type TileMips = Option<Vec<Vec<u8>>>;

/// Load a tile from the store and prepare it for upload.
#[cfg(not(target_arch = "wasm32"))]
fn load_tile(store: &TileStore, grid: &TileGrid, t: TileId, lut: &Lut) -> TileMips {
    tile_from_bytes(store.get_blocking(t), grid, lut)
}

fn tile_from_bytes(data: Result<Vec<u8>, String>, grid: &TileGrid, lut: &Lut) -> TileMips {
    let rgb = crate::store::decode_tile(&data.ok()?).ok()?;
    (rgb.len() == grid.sx as usize * grid.sy as usize * 3).then(|| slot_mips(&rgb, grid, lut))
}

fn downsample_rgba(src: &[u8], w: usize, h: usize, lut: &Lut) -> Vec<u8> {
    let (dw, dh) = (w / 2, h / 2);
    let mut out = vec![255u8; dw * dh * 4];
    for y in 0..dh {
        let r0 = &src[(2 * y) * w * 4..(2 * y + 1) * w * 4];
        let r1 = &src[(2 * y + 1) * w * 4..(2 * y + 2) * w * 4];
        let o = &mut out[y * dw * 4..(y + 1) * dw * 4];
        for x in 0..dw {
            for c in 0..3 {
                o[4 * x + c] = lut.avg4(r0[8 * x + c], r0[8 * x + 4 + c], r1[8 * x + c], r1[8 * x + 4 + c]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GPU cache

pub struct GpuTiles {
    pub id: u32,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub table: wgpu::Texture,
    pub table_view: wgpu::TextureView,
    pub grid: TileGrid,
    /// Full-resolution image size.
    pub width: u32,
    pub height: u32,
    pub slots: u32,
}

impl GpuTiles {
    pub fn allocate(device: &wgpu::Device, grid: TileGrid, width: u32, height: u32, slots: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pano tile cache"),
            size: wgpu::Extent3d { width: grid.sx, height: grid.sy, depth_or_array_layers: slots },
            mip_level_count: grid.mip_levels(),
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("pano tile cache view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let table = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pano page table"),
            size: wgpu::Extent3d { width: grid.table_w, height: grid.table_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let table_view = table.create_view(&Default::default());
        Self { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), texture, view, table, table_view, grid, width, height, slots }
    }

    fn upload(&self, queue: &wgpu::Queue, slot: u32, mips: &[Vec<u8>]) {
        let (mut w, mut h) = (self.grid.sx, self.grid.sy);
        for (m, data) in mips.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: m as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: slot },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            w /= 2;
            h /= 2;
        }
    }

    fn write_table(&self, queue: &wgpu::Queue, table: &[u32]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.table,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(table),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.grid.table_w * 4),
                rows_per_image: Some(self.grid.table_h),
            },
            wgpu::Extent3d { width: self.grid.table_w, height: self.grid.table_h, depth_or_array_layers: 1 },
        );
    }
}

/// Streams tiles from the tile store into the GPU cache. Lives with the
/// displayed panorama.
pub struct TileCache {
    pub gpu: Arc<GpuTiles>,
    store: Arc<TileStore>,
    lut: Arc<Lut>,
    owner: Vec<Option<TileId>>,
    last_used: Vec<u64>,
    pinned: u32,
    resident: HashMap<TileId, u32>,
    pending: HashSet<TileId>,
    /// Tiles that could not be loaded; not requested again.
    failed: HashSet<TileId>,
    table: Vec<u32>,
    table_dirty: bool,
    frame: u64,
    last_view: Option<[f64; 5]>,
    needed: Vec<TileId>,
    /// View motion, for loading tiles before they come into view.
    motion: Motion,
    /// Number of tiles at the start of `needed` that are visible right now.
    visible_count: usize,
    /// Frames drawn, and frames where a visible tile was still missing (so a
    /// coarser level was shown); logged when the panorama is closed.
    frames_drawn: u64,
    frames_missing: u64,
    #[cfg(not(target_arch = "wasm32"))]
    requests: Option<mpsc::Sender<TileId>>,
    #[cfg(not(target_arch = "wasm32"))]
    results: mpsc::Receiver<(TileId, TileMips)>,
    #[cfg(not(target_arch = "wasm32"))]
    wanted: Arc<Mutex<HashSet<TileId>>>,
    #[cfg(target_arch = "wasm32")]
    queue: VecDeque<TileId>,
    #[cfg(target_arch = "wasm32")]
    ready: Arc<std::sync::Mutex<Vec<(TileId, TileMips)>>>,
    #[cfg(target_arch = "wasm32")]
    in_flight: usize,
    #[cfg(target_arch = "wasm32")]
    ctx: eframe::egui::Context,
}

impl TileCache {
    /// Upload the base level (always resident, already loaded by the caller)
    /// and start the tile workers.
    pub fn new(
        gpu: GpuTiles,
        store: TileStore,
        base: Vec<(TileId, Vec<Vec<u8>>)>,
        lut: Lut,
        queue: &wgpu::Queue,
        ctx: &eframe::egui::Context,
    ) -> Self {
        let grid = gpu.grid.clone();
        let slots = gpu.slots as usize;
        let mut cache = Self {
            gpu: Arc::new(gpu),
            store: Arc::new(store),
            lut: Arc::new(lut),
            owner: vec![None; slots],
            last_used: vec![0; slots],
            pinned: 0,
            resident: HashMap::new(),
            pending: HashSet::new(),
            failed: HashSet::new(),
            table: vec![0; (grid.table_w * grid.table_h) as usize],
            table_dirty: true,
            frame: 0,
            last_view: None,
            needed: Vec::new(),
            motion: Motion::default(),
            visible_count: 0,
            frames_drawn: 0,
            frames_missing: 0,
            #[cfg(not(target_arch = "wasm32"))]
            requests: None,
            #[cfg(not(target_arch = "wasm32"))]
            results: mpsc::channel().1,
            #[cfg(not(target_arch = "wasm32"))]
            wanted: Arc::new(Mutex::new(HashSet::new())),
            #[cfg(target_arch = "wasm32")]
            queue: VecDeque::new(),
            #[cfg(target_arch = "wasm32")]
            ready: Arc::new(std::sync::Mutex::new(Vec::new())),
            #[cfg(target_arch = "wasm32")]
            in_flight: 0,
            #[cfg(target_arch = "wasm32")]
            ctx: ctx.clone(),
        };
        for (t, mips) in base {
            cache.install(t, &mips, queue);
        }
        cache.pinned = cache.resident.len() as u32;
        cache.gpu.write_table(queue, &cache.table);
        cache.table_dirty = false;
        #[cfg(not(target_arch = "wasm32"))]
        cache.start_workers(ctx);
        cache
    }

    /// Whether the tiles survive closing the app (disk or browser cache).
    pub fn is_persistent(&self) -> bool {
        self.store.is_persistent()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_workers(&mut self, ctx: &eframe::egui::Context) {
        let (req_tx, req_rx) = mpsc::channel::<TileId>();
        let (res_tx, res_rx) = mpsc::channel();
        let req_rx = Arc::new(Mutex::new(req_rx));
        let workers = (crate::loader::threads() / 2).clamp(1, 4);
        for i in 0..workers {
            let (rx, tx) = (req_rx.clone(), res_tx.clone());
            let (store, lut, wanted, ctx) = (self.store.clone(), self.lut.clone(), self.wanted.clone(), ctx.clone());
            let grid = self.gpu.grid.clone();
            std::thread::Builder::new()
                .name(format!("tile-worker-{i}"))
                .spawn(move || {
                    loop {
                        let Ok(t) = rx.lock().unwrap().recv() else { return };
                        // Skip tiles that scrolled out of view while queued.
                        let data = if wanted.lock().unwrap().contains(&t) { load_tile(&store, &grid, t, &lut) } else { None };
                        if tx.send((t, data)).is_err() {
                            return;
                        }
                        ctx.request_repaint();
                    }
                })
                .expect("spawn tile worker");
        }
        self.requests = Some(req_tx);
        self.results = res_rx;
    }

    /// Slots in use (for the About dialog).
    pub fn used_slots(&self) -> u32 {
        self.resident.len() as u32
    }

    fn install(&mut self, t: TileId, mips: &[Vec<u8>], queue: &wgpu::Queue) {
        if self.resident.contains_key(&t) {
            return;
        }
        // A free slot, else the least recently used one not needed this frame.
        let slot = match self.owner.iter().position(Option::is_none) {
            Some(s) => s,
            None => {
                let candidate = (self.pinned as usize..self.owner.len())
                    .filter(|&s| self.last_used[s] < self.frame)
                    .min_by_key(|&s| self.last_used[s]);
                let Some(s) = candidate else { return };
                if let Some(old) = self.owner[s].take() {
                    self.resident.remove(&old);
                    let i = self.gpu.grid.table_index(old);
                    self.table[i] = 0;
                }
                s
            }
        };
        self.gpu.upload(queue, slot as u32, mips);
        self.owner[slot] = Some(t);
        self.last_used[slot] = self.frame;
        self.resident.insert(t, slot as u32);
        let i = self.gpu.grid.table_index(t);
        self.table[i] = slot as u32 + 1;
        self.table_dirty = true;
    }

    /// Bring the tiles needed for `view` into the cache. Returns true while
    /// tiles are still on their way (the caller should repaint).
    pub fn update(&mut self, view: &View, queue: &wgpu::Queue) -> bool {
        self.frame += 1;
        let key = [view.yaw, view.pitch, view.hfov, view.vw, view.vh];
        let was_moving = self.motion.is_moving(view);
        self.motion.track(view);
        // Re-plan when the view changed, and once more when it came to rest so
        // the look-ahead tiles are dropped.
        if self.last_view != Some(key) || was_moving {
            self.last_view = Some(key);
            let capacity = (self.gpu.slots - self.pinned) as usize;
            let motion = if prefetch_enabled() { self.motion } else { Motion::default() };
            (self.needed, self.visible_count) =
                plan_tiles(view, &motion, &self.gpu.grid, self.gpu.width, self.gpu.height, capacity);
        }
        self.frames_drawn += 1;
        if self.needed[..self.visible_count].iter().any(|t| !self.resident.contains_key(t)) {
            self.frames_missing += 1;
        }
        let mut missing = Vec::new();
        for t in &self.needed {
            match self.resident.get(t) {
                Some(&s) => self.last_used[s as usize] = self.frame,
                None if !self.failed.contains(t) => missing.push(*t),
                None => {}
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            *self.wanted.lock().unwrap() = missing.iter().copied().collect();
            if let Some(tx) = &self.requests {
                for t in &missing {
                    if self.pending.insert(*t) {
                        let _ = tx.send(*t);
                    }
                }
            }
            let wanted = self.wanted.lock().unwrap().clone();
            while let Ok((t, data)) = self.results.try_recv() {
                self.pending.remove(&t);
                match data {
                    Some(mips) => self.install(t, &mips, queue),
                    // Skipped because it scrolled away, or failed to load.
                    None if wanted.contains(&t) => {
                        self.failed.insert(t);
                    }
                    None => {}
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let wanted: HashSet<TileId> = missing.iter().copied().collect();
            self.queue.retain(|t| wanted.contains(t));
            for t in &missing {
                if !self.pending.contains(t) && !self.queue.contains(t) {
                    self.queue.push_back(*t);
                }
            }
            // Reading from the browser cache is asynchronous; decoding runs
            // on the main thread, so only a couple of tiles are in flight.
            while self.in_flight < 2 {
                let Some(t) = self.queue.pop_front() else { break };
                self.pending.insert(t);
                self.in_flight += 1;
                let (store, grid, lut, ready, ctx) =
                    (self.store.clone(), self.gpu.grid.clone(), self.lut.clone(), self.ready.clone(), self.ctx.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let mips = tile_from_bytes(store.get(t).await, &grid, &lut);
                    ready.lock().unwrap().push((t, mips));
                    ctx.request_repaint();
                });
            }
            let done = std::mem::take(&mut *self.ready.lock().unwrap());
            for (t, mips) in done {
                self.in_flight -= 1;
                self.pending.remove(&t);
                match mips {
                    Some(m) => self.install(t, &m, queue),
                    None => {
                        self.failed.insert(t);
                    }
                }
            }
        }

        if self.table_dirty {
            self.gpu.write_table(queue, &self.table);
            self.table_dirty = false;
        }
        #[cfg(target_arch = "wasm32")]
        return !self.pending.is_empty() || !self.queue.is_empty();
        #[cfg(not(target_arch = "wasm32"))]
        !self.pending.is_empty()
    }
}

/// Smoothed speed of the view, measured between frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct Motion {
    last: Option<(web_time::Instant, f64, f64, f64)>,
    /// Degrees per second.
    pub yaw: f64,
    pub pitch: f64,
    /// Change of ln(hfov) per second (negative: zooming in).
    pub zoom: f64,
}

impl Motion {
    pub fn track(&mut self, view: &View) {
        self.track_at(view, web_time::Instant::now());
    }

    fn track_at(&mut self, view: &View, now: web_time::Instant) {
        let cur = (now, view.yaw, view.pitch, view.hfov.ln());
        if let Some((t, yaw, pitch, lh)) = self.last {
            let dt = now.duration_since(t).as_secs_f64();
            if dt > 0.25 {
                // First frame after a pause: no speed yet.
                *self = Motion { last: Some(cur), ..Default::default() };
                return;
            }
            if dt > 1e-4 {
                let mut dyaw = view.yaw - yaw;
                if dyaw > 180.0 {
                    dyaw -= 360.0;
                } else if dyaw < -180.0 {
                    dyaw += 360.0;
                }
                // Exponential smoothing with a time constant of ~0.1 s.
                let a = (dt / 0.1).min(1.0);
                self.yaw += a * (dyaw / dt - self.yaw);
                self.pitch += a * ((view.pitch - pitch) / dt - self.pitch);
                self.zoom += a * ((cur.3 - lh) / dt - self.zoom);
            }
        }
        self.last = Some(cur);
    }

    /// Fast enough for looking ahead to pay off.
    pub fn is_moving(&self, view: &View) -> bool {
        let pan = self.yaw.hypot(self.pitch) / view.hfov.max(1e-6);
        pan > 0.02 || self.zoom.abs() > 0.02
    }

    /// Where the view will be in `t` seconds at the current speed. The
    /// distance is capped so a quick flick does not request half the sphere.
    pub fn predict(&self, view: &View, t: f64) -> View {
        let mut v = *view;
        let reach = 0.75 * view.hfov;
        let scale = (reach / (self.yaw.hypot(self.pitch) * t).max(1e-9)).min(1.0);
        v.set_hfov(view.hfov * (self.zoom * t).clamp(-0.7, 0.7).exp());
        v.set_pitch(view.pitch + self.pitch * t * scale);
        v.set_yaw(view.yaw + self.yaw * t * scale);
        v
    }
}

/// `RUSTYPANO_NO_PREFETCH` turns look-ahead off (for comparisons).
fn prefetch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RUSTYPANO_NO_PREFETCH").is_none())
}

impl Drop for TileCache {
    fn drop(&mut self) {
        if self.frames_drawn > 0 {
            log::info!(
                "{} of {} frames showed a visible tile at a coarser level",
                self.frames_missing,
                self.frames_drawn
            );
        }
    }
}

/// Look-ahead times for prefetching, in seconds.
const LOOK_AHEAD: [f64; 3] = [0.25, 0.5, 0.8];

/// Tiles to keep in the GPU cache: everything visible now (coarse levels and
/// the centre first), then the tiles of where the view is heading. The list is
/// cut to `capacity` so prefetching never displaces visible tiles. Also
/// returns how many of the tiles are visible now.
pub fn plan_tiles(
    view: &View,
    motion: &Motion,
    grid: &TileGrid,
    width: u32,
    height: u32,
    capacity: usize,
) -> (Vec<TileId>, usize) {
    let base = grid.base();
    let mut tiles: Vec<TileId> = visible_tiles(view, grid, width, height)
        .into_iter()
        .filter(|t| t.level as usize != base)
        .collect();
    let visible = tiles.len().min(capacity);
    if motion.is_moving(view) {
        let mut seen: HashSet<TileId> = tiles.iter().copied().collect();
        for t in LOOK_AHEAD {
            for tile in visible_tiles(&motion.predict(view, t), grid, width, height) {
                if tile.level as usize != base && seen.insert(tile) {
                    tiles.push(tile);
                }
            }
        }
    }
    tiles.truncate(capacity);
    (tiles, visible)
}

/// Tiles needed to draw `view` at the resolution the screen needs, coarse
/// levels first, then by distance from the centre of the view. The level per
/// screen position follows the same rule as the shader.
pub fn visible_tiles(view: &View, grid: &TileGrid, width: u32, height: u32) -> Vec<TileId> {
    let map = view.screen_mapper();
    let (w, h) = (width as f64, height as f64);
    let px_per_deg = w / 360.0;
    let base = grid.base();
    let step = 24.0;
    let margin = 0.1;
    let mut found: HashMap<TileId, f64> = HashMap::new();
    let (x0, x1) = (-view.vw * margin, view.vw * (1.0 + margin));
    let (y0, y1) = (-view.vh * margin, view.vh * (1.0 + margin));
    let wrap = |d: f64| if d > 180.0 { d - 360.0 } else if d < -180.0 { d + 360.0 } else { d };
    let mut sy = y0;
    while sy <= y1 {
        let mut sx = x0;
        while sx <= x1 {
            let (lon, lat) = map(sx, sy);
            let (lon_x, lat_x) = map(sx + 1.0, sy);
            let (lon_y, lat_y) = map(sx, sy + 1.0);
            let c = lat.to_radians().cos();
            let fx = (wrap(lon_x - lon) * c).hypot(lat_x - lat);
            let fy = (wrap(lon_y - lon) * c).hypot(lat_y - lat);
            let lod = (fx.max(fy) * px_per_deg).max(1e-9).log2();
            let gx = lon.rem_euclid(360.0) * px_per_deg;
            let gy = view.geo.horizon_row - lat * px_per_deg;
            if (0.0..h).contains(&gy) {
                let k = (lod.floor().max(0.0) as usize).min(base);
                let l = grid.levels[k];
                let tx = ((gx * l.w as f64 / w) as u32 / grid.cx).min(l.nx - 1);
                let ty = ((gy * l.h as f64 / h) as u32 / grid.cy).min(l.ny - 1);
                let d = (sx - view.vw * 0.5).hypot(sy - view.vh * 0.5);
                let e = found.entry(TileId { level: k as u8, tx: tx as u16, ty: ty as u16 }).or_insert(d);
                *e = e.min(d);
            }
            sx += step;
        }
        sy += step;
    }
    let mut tiles: Vec<(TileId, f64)> = found.into_iter().collect();
    tiles.sort_by(|a, b| b.0.level.cmp(&a.0.level).then(a.1.partial_cmp(&b.1).unwrap()));
    tiles.into_iter().map(|(t, _)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::PanoGeometry;

    #[test]
    fn grid_fits_image() {
        for &(w, h) in &[(100, 50), (8192, 4096), (20000, 10000), (40000, 20000), (12345, 3000)] {
            let g = TileGrid::new(w, h);
            assert_eq!(g.cx % ALIGN, 0);
            assert_eq!(g.cy % ALIGN, 0);
            assert!(g.cx <= 1024 && g.cy <= 1024);
            let l0 = g.levels[0];
            assert!(l0.nx * g.cx >= w && l0.ny * g.cy >= h);
            assert!(g.base_tiles() <= BASE_TILES || g.levels.len() == MAX_LEVELS);
            assert_eq!(g.table_h, g.levels.iter().map(|l| l.ny).sum::<u32>());
        }
    }

    #[test]
    fn adaptive_tiles_waste_little() {
        // 20000 = 25 × 800 and 10000 = 25 × 400 → ≤ 1 % waste on level 0.
        let g = TileGrid::new(20000, 10000);
        let l0 = g.levels[0];
        let waste = (l0.nx * g.cx * l0.ny * g.cy) as f64 / (20000.0 * 10000.0) - 1.0;
        assert!(waste < 0.05, "waste {waste}");
    }

    #[test]
    fn slot_count_avoids_gl_view_heuristics() {
        let g = TileGrid::new(20000, 10000);
        for budget_mb in [64u64, 256, 512, 1024, 4096] {
            let n = g.slots_for_budget(budget_mb << 20, 2048);
            assert!(n != 1 && n % 6 != 0, "{n}");
            assert!(n >= g.base_tiles());
            assert!(n <= MAX_SLOTS + 1);
        }
    }

    #[test]
    fn halve_is_gamma_correct() {
        let lut = Lut::new();
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 255, 255]));
        img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        // Half white, half black averages to linear 0.5 = sRGB 188, not 128.
        assert_eq!(halve(&img, &lut).get_pixel(0, 0).0[0], 188);
    }

    #[test]
    fn prefetches_in_the_direction_of_motion() {
        let (w, h) = (20000, 10000);
        let g = TileGrid::new(w, h);
        let mut v = View::new(PanoGeometry::new(w, h, None));
        v.set_viewport(1920.0, 1080.0);
        v.set_hfov(40.0);
        v.set_yaw(180.0);
        // Pan right at half a field of view per second.
        let mut m = Motion::default();
        let t0 = web_time::Instant::now();
        for i in 0..40 {
            let mut f = v;
            f.set_yaw(180.0 + 20.0 * i as f64 * 0.016);
            m.track_at(&f, t0 + std::time::Duration::from_millis(16 * i));
        }
        assert!((m.yaw - 20.0).abs() < 2.0, "measured {}", m.yaw);
        assert!(m.is_moving(&v));

        let (still, nv) = plan_tiles(&v, &Motion::default(), &g, w, h, 1000);
        let (moving, mv) = plan_tiles(&v, &m, &g, w, h, 1000);
        assert_eq!((nv, mv), (still.len(), still.len()));
        assert_eq!(&moving[..still.len()], &still[..], "visible tiles come first");
        let extra = &moving[still.len()..];
        assert!(!extra.is_empty(), "nothing prefetched");
        // The extra tiles lie to the right (increasing yaw) of the view.
        let l = g.levels[extra[0].level as usize];
        let yaw_of = |t: &TileId| (t.tx as f64 + 0.5) * g.cx as f64 / l.w as f64 * 360.0;
        assert!(extra.iter().all(|t| yaw_of(t) > 180.0), "{extra:?}");
        // Prefetching never pushes visible tiles out.
        let (capped, _) = plan_tiles(&v, &m, &g, w, h, still.len());
        assert_eq!(capped, still);
    }

    #[test]
    fn motion_resets_after_a_pause() {
        let mut v = View::new(PanoGeometry::new(8000, 4000, None));
        v.set_viewport(1000.0, 800.0);
        let mut m = Motion::default();
        let t0 = web_time::Instant::now();
        m.track_at(&v, t0);
        v.set_yaw(v.yaw + 1.0);
        m.track_at(&v, t0 + std::time::Duration::from_millis(16));
        assert!(m.yaw > 0.0);
        m.track_at(&v, t0 + std::time::Duration::from_secs(2));
        assert_eq!(m.yaw, 0.0);
        assert!(!m.is_moving(&v));
    }

    #[test]
    fn visible_tiles_cover_view_at_right_level() {
        let (w, h) = (20000, 10000);
        let g = TileGrid::new(w, h);
        let mut v = View::new(PanoGeometry::new(w, h, None));
        v.set_viewport(1920.0, 1080.0);
        // Zoomed in to ~100 %: the finest level must be requested.
        v.set_hfov(1920.0 / (w as f64 / 360.0));
        let tiles = visible_tiles(&v, &g, w, h);
        assert!(tiles.iter().any(|t| t.level == 0));
        // Zoomed out: the centre needs a coarse level (the magnified edges of
        // a wide rectilinear view may still need finer ones), and the request
        // fits into the cache.
        v.set_hfov(120.0);
        let tiles = visible_tiles(&v, &g, w, h);
        assert!(tiles.iter().any(|t| t.level >= 2), "{tiles:?}");
        assert!(tiles.len() as u32 <= MAX_SLOTS, "{}", tiles.len());
    }
}
