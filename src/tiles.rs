// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Multi-resolution tile streaming.
//!
//! The decoded panorama is kept on the CPU as a pyramid (full resolution,
//! half, quarter, …). Each pyramid level is cut into tiles of `cx × cy`
//! pixels. On the GPU there is only a fixed-size cache: one
//! `texture_2d_array` whose layers ("slots") hold one tile each, plus a page
//! table texture that maps (level, tile x, tile y) to a slot.
//!
//! The coarsest level ("base") is always resident, so something can always be
//! drawn. Every frame the visible tiles at the resolution the screen needs are
//! determined on the CPU and missing ones are built and uploaded (natively by
//! worker threads, in the browser within a time budget per frame); the least
//! recently used tiles are evicted. The shader picks the finest resident level
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
// CPU pyramid

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
    fn avg4(&self, a: u8, b: u8, c: u8, d: u8) -> u8 {
        let t = &self.to_lin;
        self.enc((t[a as usize] + t[b as usize] + t[c as usize] + t[d as usize]) * 0.25)
    }
}

/// The panorama at full, half, quarter, … resolution (one image per grid level).
pub struct Pyramid {
    pub levels: Vec<image::RgbImage>,
}

impl Pyramid {
    pub fn build(full: image::RgbImage, grid: &TileGrid, lut: &Lut, mut progress: impl FnMut(usize)) -> Self {
        let mut levels = vec![full];
        for i in 1..grid.levels.len() {
            let next = halve(&levels[i - 1], lut);
            debug_assert_eq!((next.width(), next.height()), (grid.levels[i].w, grid.levels[i].h));
            levels.push(next);
            progress(i);
        }
        Self { levels }
    }
}

/// Gamma-correct 2× box downscale of an RGB image (odd sizes round up).
pub fn halve(src: &image::RgbImage, lut: &Lut) -> image::RgbImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = image::RgbImage::new(nw as u32, nh as u32);
    let s = src.as_raw();
    let rows_per = nh.div_ceil(crate::loader::threads()).max(1);
    crate::loader::par_chunks(&mut out, rows_per * nw * 3, |chunk, first_byte| {
        let y0 = first_byte / (nw * 3);
        for (r, row) in chunk.chunks_mut(nw * 3).enumerate() {
            let y = y0 + r;
            let (a, b) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
            for x in 0..nw {
                let (p, q) = ((2 * x).min(w - 1), (2 * x + 1).min(w - 1));
                for c in 0..3 {
                    row[x * 3 + c] = lut.avg4(
                        s[(a * w + p) * 3 + c],
                        s[(a * w + q) * 3 + c],
                        s[(b * w + p) * 3 + c],
                        s[(b * w + q) * 3 + c],
                    );
                }
            }
        }
    });
    out
}

/// RGBA pixels of one tile with gutter, followed by its mip chain.
pub fn build_tile(pyr: &Pyramid, grid: &TileGrid, t: TileId, lut: &Lut) -> Vec<Vec<u8>> {
    let img = &pyr.levels[t.level as usize];
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (sx, sy) = (grid.sx as usize, grid.sy as usize);
    let x0 = (t.tx as u32 * grid.cx) as i64 - GUTTER as i64;
    let y0 = (t.ty as u32 * grid.cy) as i64 - GUTTER as i64;
    let src = img.as_raw();
    let cols: Vec<usize> = (0..sx as i64).map(|i| (x0 + i).rem_euclid(w) as usize).collect();
    let mut level = vec![255u8; sx * sy * 4];
    for j in 0..sy {
        let syy = (y0 + j as i64).clamp(0, h - 1) as usize;
        let srow = &src[syy * w as usize * 3..(syy + 1) * w as usize * 3];
        let drow = &mut level[j * sx * 4..(j + 1) * sx * 4];
        for (i, &c) in cols.iter().enumerate() {
            drow[i * 4..i * 4 + 3].copy_from_slice(&srow[c * 3..c * 3 + 3]);
        }
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

/// Streams tiles into the GPU cache. Lives with the displayed panorama.
pub struct TileCache {
    pub gpu: Arc<GpuTiles>,
    pyramid: Arc<Pyramid>,
    lut: Arc<Lut>,
    owner: Vec<Option<TileId>>,
    last_used: Vec<u64>,
    pinned: u32,
    resident: HashMap<TileId, u32>,
    pending: HashSet<TileId>,
    table: Vec<u32>,
    table_dirty: bool,
    frame: u64,
    last_view: Option<[f64; 5]>,
    needed: Vec<TileId>,
    #[cfg(not(target_arch = "wasm32"))]
    requests: Option<mpsc::Sender<TileId>>,
    #[cfg(not(target_arch = "wasm32"))]
    results: mpsc::Receiver<(TileId, Option<Vec<Vec<u8>>>)>,
    #[cfg(not(target_arch = "wasm32"))]
    wanted: Arc<Mutex<HashSet<TileId>>>,
    #[cfg(target_arch = "wasm32")]
    queue: VecDeque<TileId>,
}

impl TileCache {
    /// Upload the base level (always resident) and start the tile workers.
    pub fn new(
        gpu: GpuTiles,
        pyramid: Pyramid,
        lut: Lut,
        queue: &wgpu::Queue,
        ctx: &eframe::egui::Context,
    ) -> Self {
        let grid = gpu.grid.clone();
        let slots = gpu.slots as usize;
        let mut cache = Self {
            gpu: Arc::new(gpu),
            pyramid: Arc::new(pyramid),
            lut: Arc::new(lut),
            owner: vec![None; slots],
            last_used: vec![0; slots],
            pinned: 0,
            resident: HashMap::new(),
            pending: HashSet::new(),
            table: vec![0; (grid.table_w * grid.table_h) as usize],
            table_dirty: true,
            frame: 0,
            last_view: None,
            needed: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            requests: None,
            #[cfg(not(target_arch = "wasm32"))]
            results: mpsc::channel().1,
            #[cfg(not(target_arch = "wasm32"))]
            wanted: Arc::new(Mutex::new(HashSet::new())),
            #[cfg(target_arch = "wasm32")]
            queue: VecDeque::new(),
        };
        let base = grid.base();
        let b = grid.levels[base];
        for ty in 0..b.ny {
            for tx in 0..b.nx {
                let t = TileId { level: base as u8, tx: tx as u16, ty: ty as u16 };
                let mips = build_tile(&cache.pyramid, &grid, t, &cache.lut);
                cache.install(t, &mips, queue);
            }
        }
        cache.pinned = cache.resident.len() as u32;
        cache.gpu.write_table(queue, &cache.table);
        cache.table_dirty = false;
        #[cfg(not(target_arch = "wasm32"))]
        cache.start_workers(ctx);
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
        cache
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_workers(&mut self, ctx: &eframe::egui::Context) {
        let (req_tx, req_rx) = mpsc::channel::<TileId>();
        let (res_tx, res_rx) = mpsc::channel();
        let req_rx = Arc::new(Mutex::new(req_rx));
        let workers = (crate::loader::threads() / 2).clamp(1, 4);
        for i in 0..workers {
            let (rx, tx) = (req_rx.clone(), res_tx.clone());
            let (pyr, lut, wanted, ctx) = (self.pyramid.clone(), self.lut.clone(), self.wanted.clone(), ctx.clone());
            let grid = self.gpu.grid.clone();
            std::thread::Builder::new()
                .name(format!("tile-worker-{i}"))
                .spawn(move || {
                    loop {
                        let Ok(t) = rx.lock().unwrap().recv() else { return };
                        // Skip tiles that scrolled out of view while queued.
                        let data = wanted.lock().unwrap().contains(&t).then(|| build_tile(&pyr, &grid, t, &lut));
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
        if self.last_view != Some(key) {
            self.last_view = Some(key);
            let capacity = (self.gpu.slots - self.pinned) as usize;
            self.needed = visible_tiles(view, &self.gpu.grid, self.gpu.width, self.gpu.height);
            self.needed.retain(|t| t.level as usize != self.gpu.grid.base());
            self.needed.truncate(capacity);
        }
        let mut missing = Vec::new();
        for t in &self.needed {
            match self.resident.get(t) {
                Some(&s) => self.last_used[s as usize] = self.frame,
                None => missing.push(*t),
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
            while let Ok((t, data)) = self.results.try_recv() {
                self.pending.remove(&t);
                if let Some(mips) = data {
                    self.install(t, &mips, queue);
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let wanted: HashSet<TileId> = missing.iter().copied().collect();
            self.queue.retain(|t| wanted.contains(t));
            self.pending.retain(|t| wanted.contains(t));
            for t in &missing {
                if self.pending.insert(*t) {
                    self.queue.push_back(*t);
                }
            }
            // Build tiles on the main thread for at most ~8 ms per frame.
            let t0 = web_time::Instant::now();
            while let Some(t) = self.queue.pop_front() {
                self.pending.remove(&t);
                let mips = build_tile(&self.pyramid, &self.gpu.grid, t, &self.lut);
                self.install(t, &mips, queue);
                if t0.elapsed().as_millis() >= 8 {
                    break;
                }
            }
        }

        if self.table_dirty {
            self.gpu.write_table(queue, &self.table);
            self.table_dirty = false;
        }
        !self.pending.is_empty()
    }
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
    fn tile_gutter_wraps_horizontally() {
        let mut img = image::RgbImage::new(64, 32);
        img.put_pixel(63, 0, image::Rgb([9, 8, 7]));
        let g = TileGrid::new(64, 32);
        let lut = Lut::new();
        let pyr = Pyramid { levels: vec![img] };
        let mips = build_tile(&pyr, &g, TileId { level: 0, tx: 0, ty: 0 }, &lut);
        let i = ((GUTTER as usize) * g.sx as usize + (GUTTER as usize - 1)) * 4;
        assert_eq!(&mips[0][i..i + 3], &[9, 8, 7]);
        assert_eq!(mips.len() as u32, g.mip_levels());
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
