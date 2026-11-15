// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Building the tile pyramid from an image read in horizontal stripes.
//!
//! [`open_rows`] turns an image file into a [`RowReader`] that hands out RGB8
//! rows from top to bottom. PNG, TIFF (strips and tiles), JPEG XL and BMP are
//! read stripe by stripe, so only a few rows are in memory at a time; JPEG and
//! WebP decoders can only produce the whole image, which is then handed out
//! row by row and freed afterwards.
//!
//! [`PyramidBuilder`] consumes those rows. For every pyramid level it keeps a
//! band of about one tile row, emits finished tiles (with their gutter) as soon
//! as all their rows have arrived, and halves rows gamma-correctly into the
//! next level. The full image never has to be in memory.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Seek};

use image::{ImageDecoder, ImageDecoderRect, ImageReader};

use crate::tiles::{GUTTER, Lut, TileGrid, TileId};

pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

pub type Input = BufReader<Box<dyn ReadSeek>>;

/// Hands out RGB8 rows from top to bottom.
pub trait RowReader {
    /// Append up to `max` rows to `out`; returns how many (0 at the end).
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String>;
}

pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub icc: Option<Vec<u8>>,
    pub rows: Box<dyn RowReader>,
    /// Whether rows are decoded stripe by stripe (false: the whole image was
    /// decoded up front).
    pub streaming: bool,
}

fn err(e: impl std::fmt::Display) -> String {
    tr!("Cannot decode this image: {error}", error = e)
}

// ---------------------------------------------------------------------------
// Sample conversion

/// Append RGB8 pixels converted from interleaved 8-bit samples.
fn push_u8(src: &[u8], channels: usize, out: &mut Vec<u8>) {
    match channels {
        1 => src.iter().for_each(|&g| out.extend_from_slice(&[g, g, g])),
        2 => src.as_chunks::<2>().0.iter().for_each(|p| out.extend_from_slice(&[p[0], p[0], p[0]])),
        3 => out.extend_from_slice(src),
        _ => src.chunks_exact(channels).for_each(|p| out.extend_from_slice(&p[..3])),
    }
}

/// Append RGB8 pixels converted from interleaved 16-bit samples.
fn push_u16(src: &[u16], channels: usize, out: &mut Vec<u8>) {
    let b = |v: u16| (v >> 8) as u8;
    match channels {
        1 | 2 => src.chunks_exact(channels).for_each(|p| {
            let g = b(p[0]);
            out.extend_from_slice(&[g, g, g]);
        }),
        _ => src.chunks_exact(channels).for_each(|p| out.extend_from_slice(&[b(p[0]), b(p[1]), b(p[2])])),
    }
}

/// Append RGB8 pixels from raw bytes of an `image` colour type.
fn push_image_bytes(src: &[u8], ct: image::ColorType, out: &mut Vec<u8>) -> Result<(), String> {
    use image::ColorType as C;
    let ch = ct.channel_count() as usize;
    match ct {
        C::L8 | C::La8 | C::Rgb8 | C::Rgba8 => push_u8(src, ch, out),
        C::L16 | C::La16 | C::Rgb16 | C::Rgba16 => {
            let v: Vec<u16> = src.as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes([b[0], b[1]])).collect();
            push_u16(&v, ch, out);
        }
        C::Rgb32F | C::Rgba32F => {
            let v: Vec<u8> = src
                .as_chunks::<4>().0.iter()
                .map(|b| (f32::from_ne_bytes([b[0], b[1], b[2], b[3]]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect();
            push_u8(&v, ch, out);
        }
        other => return Err(err(format!("{other:?}"))),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Row readers

/// Fallback: the decoder produced the whole image.
struct FullRows {
    img: Option<image::RgbImage>,
    y: u32,
}

impl RowReader for FullRows {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let Some(img) = &self.img else { return Ok(0) };
        let n = (max as u32).min(img.height() - self.y);
        let row = img.width() as usize * 3;
        let start = self.y as usize * row;
        out.extend_from_slice(&img.as_raw()[start..start + n as usize * row]);
        self.y += n;
        if self.y == img.height() {
            // Release the decoded image as soon as it has been handed out.
            self.img = None;
        }
        Ok(n as usize)
    }
}

struct PngRows {
    reader: png::Reader<Input>,
    channels: usize,
    sixteen: bool,
}

impl RowReader for PngRows {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let mut n = 0;
        while n < max {
            let Some(row) = self.reader.next_row().map_err(err)? else { break };
            let data = row.data();
            if self.sixteen {
                let v: Vec<u16> = data.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes([b[0], b[1]])).collect();
                push_u16(&v, self.channels, out);
            } else {
                push_u8(data, self.channels, out);
            }
            n += 1;
        }
        Ok(n)
    }
}

struct TiffRows {
    dec: tiff::decoder::Decoder<Input>,
    width: u32,
    height: u32,
    channels: usize,
    tiled: bool,
    chunk_w: u32,
    chunk_h: u32,
    next_chunk_row: u32,
    /// Converted RGB8 rows not handed out yet.
    buffered: VecDeque<u8>,
    rows_buffered: usize,
}

impl TiffRows {
    fn push_samples(&self, r: tiff::decoder::DecodingResult, out: &mut Vec<u8>) -> Result<(), String> {
        use tiff::decoder::DecodingResult as D;
        match r {
            D::U8(v) => push_u8(&v, self.channels, out),
            D::U16(v) => push_u16(&v, self.channels, out),
            _ => return Err(err("unsupported TIFF sample format")),
        }
        Ok(())
    }

    /// Decode the next row of strips or tiles into the buffer.
    fn fill(&mut self) -> Result<bool, String> {
        let y0 = self.next_chunk_row * self.chunk_h;
        if y0 >= self.height {
            return Ok(false);
        }
        let rows = self.chunk_h.min(self.height - y0) as usize;
        let mut rgb = Vec::with_capacity(rows * self.width as usize * 3);
        if !self.tiled {
            let r = self.dec.read_chunk(self.next_chunk_row).map_err(err)?;
            self.push_samples(r, &mut rgb)?;
        } else {
            // Assemble a row of tiles into full-width rows.
            let across = self.width.div_ceil(self.chunk_w);
            let mut tiles = Vec::with_capacity(across as usize);
            for tx in 0..across {
                let idx = self.next_chunk_row * across + tx;
                let (tw, _) = self.dec.chunk_data_dimensions(idx);
                let mut t = Vec::new();
                let chunk = self.dec.read_chunk(idx).map_err(err)?;
                self.push_samples(chunk, &mut t)?;
                tiles.push((tw as usize, t));
            }
            for y in 0..rows {
                for (tw, t) in &tiles {
                    rgb.extend_from_slice(&t[y * tw * 3..(y + 1) * tw * 3]);
                }
            }
        }
        self.buffered.extend(rgb);
        self.rows_buffered += rows;
        self.next_chunk_row += 1;
        Ok(true)
    }
}

impl RowReader for TiffRows {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        while self.rows_buffered < max && self.fill()? {}
        let n = max.min(self.rows_buffered);
        let bytes = n * self.width as usize * 3;
        out.extend(self.buffered.drain(..bytes));
        self.rows_buffered -= n;
        Ok(n)
    }
}

/// Decoders that can render a rectangle (JPEG XL, BMP).
struct RectRows<D: ImageDecoderRect> {
    dec: D,
    width: u32,
    height: u32,
    y: u32,
    ct: image::ColorType,
}

impl<D: ImageDecoderRect> RowReader for RectRows<D> {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let n = (max as u32).min(self.height - self.y);
        if n == 0 {
            return Ok(0);
        }
        let bpp = self.ct.bytes_per_pixel() as usize;
        let pitch = self.width as usize * bpp;
        let mut buf = vec![0u8; pitch * n as usize];
        self.dec.read_rect(0, self.y, self.width, n, &mut buf, pitch).map_err(err)?;
        push_image_bytes(&buf, self.ct, out)?;
        self.y += n;
        Ok(n as usize)
    }
}

fn is_jxl(head: &[u8]) -> bool {
    head.starts_with(&[0xFF, 0x0A]) || head.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
}

fn sniff(input: &mut Input) -> Result<Vec<u8>, String> {
    let head = input.fill_buf().map_err(|e| e.to_string())?;
    Ok(head[..head.len().min(16)].to_vec())
}

fn full_decode(input: Input, max_pixels: u64) -> Result<Decoded, String> {
    let reader = ImageReader::new(input).with_guessed_format().map_err(|e| e.to_string())?;
    let mut reader = reader;
    reader.no_limits();
    let mut decoder = reader.into_decoder().map_err(err)?;
    let (w, h) = decoder.dimensions();
    check_pixels(w, h, max_pixels)?;
    let icc = decoder.icc_profile().ok().flatten();
    let img = image::DynamicImage::from_decoder(decoder).map_err(err)?.into_rgb8();
    Ok(Decoded { width: w, height: h, icc, rows: Box::new(FullRows { img: Some(img), y: 0 }), streaming: false })
}

fn check_pixels(w: u32, h: u32, max: u64) -> Result<(), String> {
    if w as u64 * h as u64 > max {
        return Err(tr!("This panorama ({w}×{h}) is too large for the browser version.", w = w, h = h));
    }
    Ok(())
}

/// Largest images the browser build accepts (wasm has a 4 GB address space):
/// formats decoded at once need the whole image in memory, streamed ones only
/// a band of rows.
#[cfg(target_arch = "wasm32")]
const MAX_FULL_PIXELS: u64 = 300_000_000;
#[cfg(target_arch = "wasm32")]
const MAX_STREAM_PIXELS: u64 = 2_000_000_000;
#[cfg(not(target_arch = "wasm32"))]
const MAX_FULL_PIXELS: u64 = u64::MAX;
#[cfg(not(target_arch = "wasm32"))]
const MAX_STREAM_PIXELS: u64 = u64::MAX;

/// Open an image for reading rows. `reopen` provides a fresh reader for the
/// fallback when a streaming decoder cannot handle the file.
pub fn open_rows(mut reopen: impl FnMut() -> Result<Input, String>) -> Result<Decoded, String> {
    let mut input = reopen()?;
    let head = sniff(&mut input)?;

    if is_jxl(&head) {
        let mut dec = jxl_oxide::integration::JxlDecoder::new(input).map_err(err)?;
        let (w, h) = dec.dimensions();
        check_pixels(w, h, MAX_STREAM_PIXELS)?;
        let icc = dec.icc_profile().ok().flatten();
        let ct = dec.color_type();
        return Ok(Decoded { width: w, height: h, icc, rows: Box::new(RectRows { dec, width: w, height: h, y: 0, ct }), streaming: true });
    }

    if head.starts_with(b"\x89PNG") {
        let mut d = png::Decoder::new(input);
        d.set_transformations(png::Transformations::EXPAND);
        if let Ok(reader) = d.read_info() {
            let info = reader.info();
            let (w, h) = (info.width, info.height);
            if !info.interlaced {
                check_pixels(w, h, MAX_STREAM_PIXELS)?;
                let icc = info.icc_profile.as_ref().map(|c| c.to_vec());
                let (ct, depth) = reader.output_color_type();
                let rows = PngRows { channels: ct.samples(), sixteen: depth == png::BitDepth::Sixteen, reader };
                return Ok(Decoded { width: w, height: h, icc, rows: Box::new(rows), streaming: true });
            }
        }
        return full_decode(reopen()?, MAX_FULL_PIXELS);
    }

    if head.starts_with(b"II*\0") || head.starts_with(b"MM\0*") {
        if let Some(d) = open_tiff(input)? {
            return Ok(d);
        }
        return full_decode(reopen()?, MAX_FULL_PIXELS);
    }

    if head.starts_with(b"BM") {
        let mut dec = image::codecs::bmp::BmpDecoder::new(input).map_err(err)?;
        let (w, h) = dec.dimensions();
        check_pixels(w, h, MAX_STREAM_PIXELS)?;
        let icc = dec.icc_profile().ok().flatten();
        let ct = dec.color_type();
        return Ok(Decoded { width: w, height: h, icc, rows: Box::new(RectRows { dec, width: w, height: h, y: 0, ct }), streaming: true });
    }

    // JPEG, WebP and anything else: decoded at once.
    full_decode(input, MAX_FULL_PIXELS)
}

/// Streaming TIFF reader for 8/16-bit grey, grey+alpha, RGB and RGBA;
/// `None` for anything else (handled by the full decoder).
fn open_tiff(input: Input) -> Result<Option<Decoded>, String> {
    use tiff::ColorType as T;
    let Ok(mut dec) = tiff::decoder::Decoder::new(input) else { return Ok(None) };
    let (w, h) = dec.dimensions().map_err(err)?;
    let channels = match dec.colortype().map_err(err)? {
        T::Gray(8 | 16) => 1,
        T::GrayA(8 | 16) => 2,
        T::RGB(8 | 16) => 3,
        T::RGBA(8 | 16) => 4,
        _ => return Ok(None),
    };
    check_pixels(w, h, MAX_STREAM_PIXELS)?;
    let icc = dec.get_tag_u8_vec(tiff::tags::Tag::IccProfile).ok();
    let tiled = matches!(dec.get_chunk_type(), tiff::decoder::ChunkType::Tile);
    let (chunk_w, chunk_h) = dec.chunk_dimensions();
    let rows = TiffRows {
        dec,
        width: w,
        height: h,
        channels,
        tiled,
        chunk_w,
        chunk_h: chunk_h.max(1),
        next_chunk_row: 0,
        buffered: VecDeque::new(),
        rows_buffered: 0,
    };
    Ok(Some(Decoded { width: w, height: h, icc, rows: Box::new(rows), streaming: true }))
}

// ---------------------------------------------------------------------------
// Streaming pyramid builder

/// Rows of one pyramid level that are still needed.
struct Band {
    w: usize,
    h: usize,
    rows: VecDeque<Vec<u8>>,
    /// Index of `rows[0]` in the level.
    first: usize,
    /// Rows received so far.
    received: usize,
    next_tile_row: u32,
}

impl Band {
    fn row(&self, y: usize) -> &[u8] {
        &self.rows[y - self.first]
    }
}

/// Builds all tiles of all levels from rows of the full-resolution image.
pub struct PyramidBuilder {
    grid: TileGrid,
    bands: Vec<Band>,
    tiles: Vec<(TileId, Vec<u8>)>,
}

impl PyramidBuilder {
    pub fn new(grid: TileGrid) -> Self {
        let bands = grid
            .levels
            .iter()
            .map(|l| Band { w: l.w as usize, h: l.h as usize, rows: VecDeque::new(), first: 0, received: 0, next_tile_row: 0 })
            .collect();
        Self { grid, bands, tiles: Vec::new() }
    }

    /// Feed full-resolution RGB8 rows (any number, top to bottom).
    pub fn push_rows(&mut self, data: &[u8], lut: &Lut) {
        let row_len = self.bands[0].w * 3;
        for row in data.chunks_exact(row_len) {
            self.push(0, row.to_vec(), lut);
        }
    }

    /// Tiles finished since the last call: RGB8, `sx × sy` pixels incl. gutter.
    pub fn take_tiles(&mut self) -> Vec<(TileId, Vec<u8>)> {
        std::mem::take(&mut self.tiles)
    }

    /// True once every level has received all its rows.
    pub fn is_complete(&self) -> bool {
        self.bands.iter().all(|b| b.received == b.h)
    }

    fn push(&mut self, k: usize, row: Vec<u8>, lut: &Lut) {
        {
            let b = &mut self.bands[k];
            debug_assert_eq!(row.len(), b.w * 3);
            b.rows.push_back(row);
            b.received += 1;
        }
        self.emit_ready(k);

        let (y, h) = (self.bands[k].received - 1, self.bands[k].h);
        if k + 1 < self.bands.len() {
            // Halve pairs of rows into the next level; an odd last row is
            // paired with itself.
            if y % 2 == 1 || y == h - 1 {
                let b = &self.bands[k];
                let a = b.row(y - y % 2);
                let c = b.row(y);
                let half = halve_rows(a, c, b.w, lut);
                self.push(k + 1, half, lut);
            }
        }
        self.drop_unneeded(k);
    }

    /// Emit every tile row of level `k` whose rows (incl. gutter) are present.
    fn emit_ready(&mut self, k: usize) {
        let (cx, cy, sx, sy) = (self.grid.cx as usize, self.grid.cy as usize, self.grid.sx as usize, self.grid.sy as usize);
        let g = GUTTER as usize;
        let level = self.grid.levels[k];
        loop {
            let b = &self.bands[k];
            let r = b.next_tile_row as usize;
            if r >= level.ny as usize {
                return;
            }
            let last_needed = ((r + 1) * cy + g).min(b.h) - 1;
            if b.received <= last_needed {
                return;
            }
            for tx in 0..level.nx as usize {
                let mut rgb = vec![0u8; sx * sy * 3];
                let x0 = (tx * cx) as i64 - g as i64;
                for j in 0..sy {
                    let sy_ = ((r * cy) as i64 - g as i64 + j as i64).clamp(0, b.h as i64 - 1) as usize;
                    let src = b.row(sy_);
                    let dst = &mut rgb[j * sx * 3..(j + 1) * sx * 3];
                    for i in 0..sx {
                        let x = (x0 + i as i64).rem_euclid(b.w as i64) as usize;
                        dst[i * 3..i * 3 + 3].copy_from_slice(&src[x * 3..x * 3 + 3]);
                    }
                }
                self.tiles.push((TileId { level: k as u8, tx: tx as u16, ty: r as u16 }, rgb));
            }
            self.bands[k].next_tile_row += 1;
        }
    }

    /// Forget rows that no future tile and no pending halving needs.
    fn drop_unneeded(&mut self, k: usize) {
        let cy = self.grid.cy as usize;
        let g = GUTTER as usize;
        let b = &mut self.bands[k];
        let next_tile_start = (b.next_tile_row as usize * cy).saturating_sub(g);
        // Keep the latest row in case it is the first of a halving pair.
        let keep_from = next_tile_start.min(b.received.saturating_sub(1));
        while b.first < keep_from {
            b.rows.pop_front();
            b.first += 1;
        }
    }
}

/// Gamma-correct 2×2 average of two rows (odd widths round up).
fn halve_rows(a: &[u8], b: &[u8], w: usize, lut: &Lut) -> Vec<u8> {
    let nw = w.div_ceil(2);
    let mut out = vec![0u8; nw * 3];
    for x in 0..nw {
        let (p, q) = (2 * x, (2 * x + 1).min(w - 1));
        for c in 0..3 {
            out[x * 3 + c] = lut.avg4(a[p * 3 + c], a[q * 3 + c], b[p * 3 + c], b[q * 3 + c]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: full pyramid by halving whole images, tiles cut directly.
    fn reference_tile(levels: &[image::RgbImage], grid: &TileGrid, t: TileId) -> Vec<u8> {
        let img = &levels[t.level as usize];
        let (w, h) = (img.width() as i64, img.height() as i64);
        let (sx, sy) = (grid.sx as usize, grid.sy as usize);
        let x0 = (t.tx as u32 * grid.cx) as i64 - GUTTER as i64;
        let y0 = (t.ty as u32 * grid.cy) as i64 - GUTTER as i64;
        let mut out = vec![0u8; sx * sy * 3];
        for j in 0..sy {
            let y = (y0 + j as i64).clamp(0, h - 1) as u32;
            for i in 0..sx {
                let x = (x0 + i as i64).rem_euclid(w) as u32;
                out[(j * sx + i) * 3..(j * sx + i) * 3 + 3].copy_from_slice(&img.get_pixel(x, y).0);
            }
        }
        out
    }

    fn test_image(w: u32, h: u32) -> image::RgbImage {
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 7 + y * 13) as u8, (x ^ y) as u8, (x * y / 3) as u8])
        })
    }

    #[test]
    fn streaming_builder_matches_full_pyramid() {
        // Odd sizes on purpose: odd halving, partial edge tiles, several levels.
        for &(w, h) in &[(2501u32, 1253u32), (3000, 600), (1100, 555)] {
            let img = test_image(w, h);
            let grid = TileGrid::new(w, h);
            let lut = Lut::new();
            let mut levels = vec![img.clone()];
            for _ in 1..grid.levels.len() {
                let next = crate::tiles::halve(levels.last().unwrap(), &lut);
                levels.push(next);
            }
            let mut b = PyramidBuilder::new(grid.clone());
            // Feed in uneven chunks.
            let row = w as usize * 3;
            let raw = img.as_raw();
            let mut y = 0usize;
            let mut tiles = Vec::new();
            for n in [1usize, 7, 64, 3].iter().cycle() {
                if y >= h as usize {
                    break;
                }
                let m = (*n).min(h as usize - y);
                b.push_rows(&raw[y * row..(y + m) * row], &lut);
                tiles.extend(b.take_tiles());
                y += m;
            }
            assert!(b.is_complete(), "{w}x{h}");
            assert_eq!(tiles.len() as u32, grid.tile_count(), "{w}x{h}");
            for (t, rgb) in &tiles {
                assert_eq!(rgb, &reference_tile(&levels, &grid, *t), "{w}x{h} {t:?}");
            }
            // Memory stays bounded: level 0 never holds more than ~a tile row.
            assert!(b.bands[0].rows.len() <= grid.cy as usize + 2 * GUTTER as usize + 2);
        }
    }

    #[test]
    fn png_is_read_row_by_row() {
        let img = test_image(300, 200);
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let data: std::sync::Arc<[u8]> = buf.into();
        let open = || -> Result<Input, String> {
            Ok(BufReader::new(Box::new(std::io::Cursor::new(data.clone())) as Box<dyn ReadSeek>))
        };
        let mut d = open_rows(open).unwrap();
        assert!(d.streaming);
        let mut out = Vec::new();
        while d.rows.read(17, &mut out).unwrap() > 0 {}
        assert_eq!(out, img.into_raw());
    }

    #[test]
    fn tiff_is_read_strip_by_strip() {
        let img = test_image(257, 130);
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Tiff)
            .unwrap();
        let data: std::sync::Arc<[u8]> = buf.into();
        let open = || -> Result<Input, String> {
            Ok(BufReader::new(Box::new(std::io::Cursor::new(data.clone())) as Box<dyn ReadSeek>))
        };
        let mut d = open_rows(open).unwrap();
        assert!(d.streaming);
        let mut out = Vec::new();
        while d.rows.read(9, &mut out).unwrap() > 0 {}
        assert_eq!(out, img.into_raw());
    }
}
