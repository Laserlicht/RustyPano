// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Tile cache.
//!
//! The tiles of a panorama (all pyramid levels, with gutter) are stored as
//! JPEG files, keyed by the source file's identity and the colour settings,
//! instead of keeping the image in memory. Opening the same panorama again in
//! the same session loads the tiles from here instead of decoding the image.
//! The cache belongs to the running app and is deleted when it closes.
//!
//! * Desktop: `<cache dir>/rustypano/tiles/session-<id>/<key>/<level>_<x>_<y>.jpg`
//!   plus `meta.toml`. A panorama is built in a temporary directory and
//!   renamed when complete, so an interrupted build is never used.
//! * Browser: one Cache Storage cache per panorama and tab, with the metadata
//!   written last.
//! * If neither is available, tiles are kept in memory (still JPEG-compressed).
//!
//! The size is limited; the least recently used panoramas are removed.

use std::collections::{BTreeMap, HashMap};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::tiles::TileId;

/// Bump when the tile layout or encoding changes: old caches are rebuilt.
pub const FORMAT_VERSION: u32 = 1;
/// JPEG quality of cached tiles (4:4:4, no chroma subsampling).
const QUALITY: u8 = 92;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CacheMeta {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub gpano: BTreeMap<String, String>,
    pub icc_used: Option<String>,
    /// Total size of the tiles in bytes.
    pub bytes: u64,
}

fn tile_name(t: TileId) -> String {
    format!("{}_{}_{}.jpg", t.level, t.tx, t.ty)
}

pub fn encode_tile(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(rgb.len() / 8);
    let mut enc = jpeg_encoder::Encoder::new(&mut out, QUALITY);
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_4_4);
    enc.encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb).map_err(|e| e.to_string())?;
    Ok(out)
}

/// RGB8 pixels of a cached tile.
pub fn decode_tile(jpeg: &[u8]) -> Result<Vec<u8>, String> {
    image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)
        .map(|i| i.into_rgb8().into_raw())
        .map_err(|e| e.to_string())
}

/// FNV-1a, stable across platforms and Rust versions.
pub fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Fast content hash for files held in memory.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ data.len() as u64;
    let (chunks, rest) = data.as_chunks::<8>();
    for c in chunks {
        let v = u64::from_le_bytes(*c);
        h = (h ^ v).wrapping_mul(0xff51_afd7_ed55_8ccd).rotate_left(29);
    }
    for &b in rest {
        h = (h ^ b as u64).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    }
    h
}

#[derive(Clone)]
pub enum TileStore {
    #[cfg(not(target_arch = "wasm32"))]
    Disk(PathBuf),
    #[cfg(target_arch = "wasm32")]
    Browser(String),
    Memory(Arc<Mutex<HashMap<TileId, Vec<u8>>>>),
}

impl TileStore {
    fn memory() -> Self {
        TileStore::Memory(Arc::new(Mutex::new(HashMap::new())))
    }

    /// Whether tiles survive closing the app.
    pub fn is_persistent(&self) -> bool {
        !matches!(self, TileStore::Memory(_))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn put_blocking(&self, t: TileId, data: Vec<u8>) -> Result<(), String> {
        match self {
            TileStore::Disk(dir) => std::fs::write(dir.join(tile_name(t)), data).map_err(|e| e.to_string()),
            TileStore::Memory(m) => {
                m.lock().unwrap().insert(t, data);
                Ok(())
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_blocking(&self, t: TileId) -> Result<Vec<u8>, String> {
        match self {
            TileStore::Disk(dir) => std::fs::read(dir.join(tile_name(t))).map_err(|e| e.to_string()),
            TileStore::Memory(m) => m.lock().unwrap().get(&t).cloned().ok_or_else(|| "missing tile".into()),
        }
    }

    // Natively the encoder threads call `put_blocking` directly.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub async fn put(&self, t: TileId, data: Vec<u8>) -> Result<(), String> {
        #[cfg(not(target_arch = "wasm32"))]
        return self.put_blocking(t, data);
        #[cfg(target_arch = "wasm32")]
        match self {
            TileStore::Browser(name) => crate::web::cache_put(name, &tile_name(t), &data).await,
            TileStore::Memory(m) => {
                m.lock().unwrap().insert(t, data);
                Ok(())
            }
        }
    }

    pub async fn get(&self, t: TileId) -> Result<Vec<u8>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        return self.get_blocking(t);
        #[cfg(target_arch = "wasm32")]
        match self {
            TileStore::Browser(name) => {
                crate::web::cache_get(name, &tile_name(t)).await?.ok_or_else(|| "missing tile".to_string())
            }
            TileStore::Memory(m) => m.lock().unwrap().get(&t).cloned().ok_or_else(|| "missing tile".into()),
        }
    }
}

/// A cache being filled.
pub struct Building {
    pub store: TileStore,
    key: String,
    #[cfg(not(target_arch = "wasm32"))]
    final_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Sessions
//
// Every running app (and every browser tab) has its own session; its tiles are
// deleted when it closes. Sessions signal that they are alive once a minute,
// so leftovers of crashed sessions can be removed by the next start without
// touching other instances that are still running.

static SESSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Sessions without a sign of life for this long are considered dead.
const STALE_MS: f64 = 10.0 * 60.0 * 1000.0;

fn now_ms() -> f64 {
    web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map_or(0.0, |d| d.as_millis() as f64)
}

fn session_id() -> &'static str {
    SESSION.get_or_init(|| {
        let r = store_random();
        format!("{:08x}{:08x}", (now_ms() as u64) & 0xffff_ffff, r)
    })
}

fn store_random() -> u32 {
    #[cfg(target_arch = "wasm32")]
    return (js_sys::Math::random() * u32::MAX as f64) as u32;
    #[cfg(not(target_arch = "wasm32"))]
    return (hash_str(&format!("{:?}{}", std::time::Instant::now(), std::process::id())) >> 16) as u32;
}

/// Start this app's session: remove caches of closed or crashed sessions and
/// keep signalling that this one is alive.
pub fn start_session() {
    let id = session_id().to_string();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(root) = root() else { return };
        let mine = root.join(format!("session-{id}"));
        if std::fs::create_dir_all(&mine).is_err() {
            return;
        }
        let alive = mine.join("alive");
        let _ = std::fs::write(&alive, b"");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for e in entries.flatten() {
                let path = e.path();
                if path == mine {
                    continue;
                }
                // Session directories of dead sessions, and caches of older
                // versions (which kept tiles between runs).
                let last = std::fs::metadata(path.join("alive")).and_then(|m| m.modified()).ok();
                let age = last.and_then(|t| t.elapsed().ok()).map_or(f64::MAX, |d| d.as_millis() as f64);
                if age > STALE_MS {
                    let _ = std::fs::remove_dir_all(&path);
                }
            }
        }
        std::thread::Builder::new()
            .name("cache-heartbeat".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(60));
                    match std::fs::File::options().append(true).open(&alive) {
                        Ok(f) => {
                            let _ = f.set_modified(std::time::SystemTime::now());
                        }
                        Err(_) => return, // session ended
                    }
                }
            })
            .ok();
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_heartbeat(&id);
        crate::web::every_minute(Box::new(move || web_heartbeat(&id)));
        crate::web::on_page_hide(Box::new(|| wasm_bindgen_futures::spawn_local(end_session())));
        wasm_bindgen_futures::spawn_local(web_remove_dead_sessions());
    }
}

/// Delete this session's tiles (when the app closes).
pub async fn end_session() {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dir) = session_dir() {
        let _ = std::fs::remove_dir_all(dir);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut beats = web_sessions();
        beats.remove(session_id());
        web_sessions_save(&beats);
        clear_all().await;
    }
}

// ---------------------------------------------------------------------------
// Desktop

#[cfg(not(target_arch = "wasm32"))]
fn root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RUSTYPANO_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    dirs::cache_dir().map(|d| d.join("rustypano").join("tiles"))
}

#[cfg(not(target_arch = "wasm32"))]
fn session_dir() -> Option<PathBuf> {
    Some(root()?.join(format!("session-{}", session_id())))
}

/// Where cached tiles are stored (for display in the settings).
pub fn location() -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    return root().map(|p| p.display().to_string());
    #[cfg(target_arch = "wasm32")]
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn read_meta(dir: &std::path::Path) -> Option<CacheMeta> {
    let m: CacheMeta = toml::from_str(&std::fs::read_to_string(dir.join("meta.toml")).ok()?).ok()?;
    (m.version == FORMAT_VERSION).then_some(m)
}

/// A complete cache for `key` in this session, if there is one.
pub async fn lookup(key: &str) -> Option<(TileStore, CacheMeta)> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = session_dir()?.join(key);
        let meta = read_meta(&dir)?;
        // Mark as recently used.
        if let Ok(f) = std::fs::File::options().append(true).open(dir.join("meta.toml")) {
            let _ = f.set_modified(std::time::SystemTime::now());
        }
        Some((TileStore::Disk(dir), meta))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let name = web_cache_name(key);
        let data = crate::web::cache_get(&name, "meta.toml").await.ok()??;
        let meta: CacheMeta = toml::from_str(std::str::from_utf8(&data).ok()?).ok()?;
        if meta.version != FORMAT_VERSION {
            return None;
        }
        web_index_touch(key, meta.bytes);
        Some((TileStore::Browser(name), meta))
    }
}

/// Start filling a new cache for `key`.
pub async fn begin(key: &str) -> Building {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nonce = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let tmp = session_dir().map(|r| r.join(format!("{key}.tmp-{nonce}")));
        if let Some(tmp) = tmp.filter(|t| std::fs::create_dir_all(t).is_ok()) {
            let final_dir = tmp.parent().map(|p| p.join(key));
            return Building { store: TileStore::Disk(tmp), key: key.to_string(), final_dir };
        }
        log::warn!("No tile cache directory available, keeping tiles in memory");
        Building { store: TileStore::memory(), key: key.to_string(), final_dir: None }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let name = web_cache_name(key);
        // Drop a stale, incomplete cache of the same panorama first.
        crate::web::cache_delete(&name).await;
        if crate::web::cache_put(&name, "probe", b"").await.is_ok() {
            return Building { store: TileStore::Browser(name), key: key.to_string() };
        }
        log::warn!("Browser cache storage not available, keeping tiles in memory");
        Building { store: TileStore::memory(), key: key.to_string() }
    }
}

/// Mark the cache complete and enforce the size limit (in MB).
pub async fn finish(b: Building, meta: &CacheMeta, limit_mb: u32) -> TileStore {
    let text = toml::to_string(meta).unwrap_or_default();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (TileStore::Disk(tmp), Some(final_dir)) = (&b.store, &b.final_dir) else { return b.store };
        if std::fs::write(tmp.join("meta.toml"), text).is_err() {
            return b.store;
        }
        let _ = std::fs::remove_dir_all(final_dir);
        if std::fs::rename(tmp, final_dir).is_err() {
            return b.store;
        }
        enforce_limit(limit_mb as u64 * 1024 * 1024, &b.key);
        TileStore::Disk(final_dir.clone())
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let TileStore::Browser(name) = &b.store
            && crate::web::cache_put(name, "meta.toml", text.as_bytes()).await.is_ok()
        {
            web_index_touch(&b.key, meta.bytes);
            web_enforce_limit(limit_mb as u64 * 1024 * 1024, &b.key).await;
        }
        b.store
    }
}

/// Remove the least recently used panoramas of this session until the total
/// fits `limit`.
#[cfg(not(target_arch = "wasm32"))]
fn enforce_limit(limit: u64, keep: &str) {
    let Some(dir) = session_dir() else { return };
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    let mut caches = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if !path.is_dir() || name.contains(".tmp-") {
            continue;
        }
        let used = std::fs::metadata(path.join("meta.toml")).and_then(|m| m.modified()).ok();
        let bytes = read_meta(&path).map_or(0, |m| m.bytes);
        caches.push((used, bytes, name, path));
    }
    let mut total: u64 = caches.iter().map(|c| c.1).sum();
    caches.sort_by_key(|c| c.0);
    for (_, bytes, name, path) in caches {
        if total <= limit {
            break;
        }
        if name != keep && std::fs::remove_dir_all(&path).is_ok() {
            total -= bytes;
        }
    }
}

/// Delete all tiles of this session.
pub async fn clear_all() {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dir) = session_dir()
        && let Ok(entries) = std::fs::read_dir(&dir)
    {
        for e in entries.flatten() {
            if e.path().is_dir() {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let prefix = web_session_prefix();
        for name in crate::web::cache_names().await {
            if name.starts_with(&prefix) {
                crate::web::cache_delete(&name).await;
            }
        }
        WEB_INDEX.with(|i| i.borrow_mut().clear());
    }
}

// ---------------------------------------------------------------------------
// Browser

#[cfg(target_arch = "wasm32")]
const WEB_PREFIX: &str = "rustypano-tiles-";
#[cfg(target_arch = "wasm32")]
const WEB_SESSIONS: &str = "rustypano.sessions";

#[cfg(target_arch = "wasm32")]
fn web_session_prefix() -> String {
    format!("{WEB_PREFIX}{}-", session_id())
}

#[cfg(target_arch = "wasm32")]
fn web_cache_name(key: &str) -> String {
    format!("{}{key}", web_session_prefix())
}

/// Last sign of life (ms since the epoch) of every open tab's session.
#[cfg(target_arch = "wasm32")]
fn web_sessions() -> BTreeMap<String, f64> {
    crate::web::storage_get(WEB_SESSIONS).and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn web_sessions_save(m: &BTreeMap<String, f64>) {
    crate::web::storage_set(WEB_SESSIONS, &toml::to_string(m).unwrap_or_default());
}

#[cfg(target_arch = "wasm32")]
fn web_heartbeat(id: &str) {
    let mut m = web_sessions();
    let now = now_ms();
    m.retain(|_, t| now - *t < STALE_MS);
    m.insert(id.to_string(), now);
    web_sessions_save(&m);
}

/// Delete caches of tabs that were closed (or crashed) and of older versions.
#[cfg(target_arch = "wasm32")]
async fn web_remove_dead_sessions() {
    let alive = web_sessions();
    let now = now_ms();
    for name in crate::web::cache_names().await {
        let Some(rest) = name.strip_prefix(WEB_PREFIX) else { continue };
        let session = rest.split('-').next().unwrap_or("");
        let live = alive.get(session).is_some_and(|t| now - *t < STALE_MS);
        if !live {
            crate::web::cache_delete(&name).await;
        }
    }
    crate::web::storage_set("rustypano.tilecache", ""); // index of older versions
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Size and last use of this session's panoramas, for the size limit.
    static WEB_INDEX: std::cell::RefCell<Vec<(String, u64, f64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "wasm32")]
fn web_index_touch(key: &str, bytes: u64) {
    WEB_INDEX.with(|i| {
        let mut i = i.borrow_mut();
        i.retain(|e| e.0 != key);
        i.push((key.to_string(), bytes, now_ms()));
    });
}

#[cfg(target_arch = "wasm32")]
async fn web_enforce_limit(limit: u64, keep: &str) {
    let mut entries = WEB_INDEX.with(|i| std::mem::take(&mut *i.borrow_mut()));
    entries.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let mut total: u64 = entries.iter().map(|e| e.1).sum();
    let mut kept = Vec::new();
    for e in entries {
        if total > limit && e.0 != keep {
            crate::web::cache_delete(&web_cache_name(&e.0)).await;
            total -= e.1;
        } else {
            kept.push(e);
        }
    }
    WEB_INDEX.with(|i| *i.borrow_mut() = kept);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_jpeg_roundtrip() {
        let (w, h) = (96u32, 64u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let jpeg = encode_tile(&rgb, w, h).unwrap();
        assert!(jpeg.len() < rgb.len());
        let back = decode_tile(&jpeg).unwrap();
        assert_eq!(back.len(), rgb.len());
    }

    /// `RUSTYPANO_BENCH_TILES=<jpeg> cargo test --release tile_encode_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn tile_encode_speed() {
        let Ok(path) = std::env::var("RUSTYPANO_BENCH_TILES") else { return };
        let mut reader = image::ImageReader::open(&path).unwrap();
        reader.no_limits();
        let img = reader.decode().unwrap().into_rgb8();
        let (tw, th) = (736u32, 1088u32);
        let tiles: Vec<Vec<u8>> = (0..12)
            .map(|i| image::imageops::crop_imm(&img, 1000 + i * tw, 2000, tw, th).to_image().into_raw())
            .collect();
        let px = (tiles.len() as u32 * tw * th) as f64 / 1e6;
        for (name, sampling) in [("4:4:4", jpeg_encoder::SamplingFactor::R_4_4_4), ("4:2:0", jpeg_encoder::SamplingFactor::R_4_2_0)] {
            for q in [92u8, 85] {
                let t = std::time::Instant::now();
                let mut bytes = 0;
                for rgb in &tiles {
                    let mut out = Vec::new();
                    let mut enc = jpeg_encoder::Encoder::new(&mut out, q);
                    enc.set_sampling_factor(sampling);
                    enc.encode(rgb, tw as u16, th as u16, jpeg_encoder::ColorType::Rgb).unwrap();
                    bytes += out.len();
                }
                let s = t.elapsed().as_secs_f64();
                println!("encode {name} q{q}: {:.0} MP/s single-threaded, {:.2} bytes/px", px / s, bytes as f64 / (px * 1e6));
            }
        }
        let jpeg = encode_tile(&tiles[0], tw, th).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..12 {
            decode_tile(&jpeg).unwrap();
        }
        println!("decode tile: {:.0} MP/s", px / t.elapsed().as_secs_f64());
    }

    #[test]
    fn hashes_are_stable() {
        assert_eq!(hash_str(""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(hash_bytes(b"abcdefgh1"), hash_bytes(b"abcdefgh2"));
    }

    #[test]
    fn session_cache_lifecycle() {
        // Use a private cache root, never the user's real cache.
        let root = std::env::temp_dir().join(format!("rustypano-cache-test-{}", std::process::id()));
        unsafe { std::env::set_var("RUSTYPANO_CACHE_DIR", &root) };
        // Leftover of a dead session and of an older version: removed at start.
        let dead = root.join("session-dead");
        std::fs::create_dir_all(&dead).unwrap();
        std::fs::create_dir_all(root.join("0123456789abcdef")).unwrap();
        start_session();
        assert!(!dead.exists() && !root.join("0123456789abcdef").exists());

        let t = TileId { level: 1, tx: 2, ty: 3 };
        let b = pollster::block_on(begin("k1"));
        pollster::block_on(b.store.put(t, vec![1, 2, 3])).unwrap();
        // Incomplete builds are not found.
        assert!(pollster::block_on(lookup("k1")).is_none());
        let meta = CacheMeta { version: FORMAT_VERSION, width: 10, height: 5, bytes: 3, ..Default::default() };
        let store = pollster::block_on(finish(b, &meta, u32::MAX));
        assert!(store.is_persistent());
        let (found, m) = pollster::block_on(lookup("k1")).expect("cache found");
        assert_eq!((m.width, m.height), (10, 5));
        assert_eq!(pollster::block_on(found.get(t)).unwrap(), vec![1, 2, 3]);

        // Closing the app deletes the session's tiles.
        pollster::block_on(end_session());
        assert!(pollster::block_on(lookup("k1")).is_none());
        assert!(!session_dir().unwrap().exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
