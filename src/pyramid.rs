// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Building the tile pyramid from an image read in horizontal stripes.
//!
//! [`open_rows`] turns an image file into a [`RowReader`] that hands out RGB8
//! rows from top to bottom. Baseline JPEG, PNG, TIFF (strips and tiles),
//! JPEG XL and BMP are read stripe by stripe, so only a few rows are in memory
//! at a time. Progressive JPEG (which needs all DCT coefficients before the
//! first row) and WebP (at most 16383 px per side anyway) are decoded as a
//! whole, handed out row by row and freed afterwards.
//!
//! [`PyramidBuilder`] consumes those rows. For every pyramid level it keeps a
//! band of about one tile row, emits finished tiles (with their gutter) as soon
//! as all their rows have arrived, and halves rows gamma-correctly into the
//! next level. The full image never has to be in memory.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Seek};

use image::{ImageDecoder, ImageDecoderRect, ImageReader};

use crate::tiles::{GUTTER, Lut, TileGrid, TileId};

pub trait ReadSeek: Read + Seek + Send {}
impl<T: Read + Seek + Send> ReadSeek for T {}

pub type Input = BufReader<Box<dyn ReadSeek>>;

/// Hands out RGB8 rows from top to bottom.
pub trait RowReader: Send {
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

impl<D: ImageDecoderRect + Send> RowReader for RectRows<D> {
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

// ---------------------------------------------------------------------------
// Baseline JPEG, one iMCU row (8 or 16 rows) at a time

/// zune-jpeg's fixed-point YCbCr → RGB coefficients (14-bit precision), so
/// the result matches its full decode.
const Y_CF: i32 = 16384;
const CR_CF: i32 = 22970;
const CB_CF: i32 = 29032;
const G_CR: i32 = -11700;
const G_CB: i32 = -5638;
const YUV_RND: i32 = (1 << 13) - 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum JpegColour {
    Gray,
    YCbCr,
    Rgb,
}

/// zune-jpeg peeks by reading and seeking back. `BufReader::seek` throws its
/// buffer away every time, which made decoding ten times slower; this keeps
/// relative seeks inside the buffer and tracks the position itself.
struct PeekFriendly {
    inner: Input,
    pos: u64,
}

impl Read for PeekFriendly {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl BufRead for PeekFriendly {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }
    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
        self.pos += amt as u64;
    }
}

impl Seek for PeekFriendly {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        match to {
            std::io::SeekFrom::Current(d) => {
                self.inner.seek_relative(d)?;
                self.pos = self.pos.checked_add_signed(d).ok_or(std::io::ErrorKind::InvalidInput)?;
            }
            other => self.pos = self.inner.seek(other)?,
        }
        Ok(self.pos)
    }
    fn stream_position(&mut self) -> std::io::Result<u64> {
        Ok(self.pos)
    }
}

struct JpegRows {
    dec: zune_jpeg::JpegDecoder<PeekFriendly>,
    width: usize,
    height: usize,
    colour: JpegColour,
    planes: Vec<zune_jpeg::PlaneInfo>,
    /// Up-sampling factor of each component (1 or 2) horizontally / vertically.
    fx: Vec<usize>,
    fy: Vec<usize>,
    /// Decoded component rows still needed, from row `first[c]` on.
    rows: Vec<VecDeque<Vec<u8>>>,
    first: Vec<usize>,
    received: Vec<usize>,
    /// Buffers for one iMCU row per component.
    imcu: Vec<Vec<u8>>,
    next_out: usize,
    complete: bool,
}

impl JpegRows {
    fn pull(&mut self) -> Result<(), String> {
        let strides: Vec<usize> = self.planes.iter().map(|p| p.stride).collect();
        let mut planes: Vec<&mut [u8]> = self.imcu.iter_mut().map(Vec::as_mut_slice).collect();
        match self.dec.raw_output().decode_next_imcu_row(&mut planes, &strides).map_err(err)? {
            zune_jpeg::RawImcuRowStatus::RowReady { rows_written } => {
                for (c, p) in self.planes.iter().enumerate() {
                    for r in 0..rows_written[c] {
                        let start = r * p.stride;
                        self.rows[c].push_back(self.imcu[c][start..start + p.width].to_vec());
                    }
                    self.received[c] += rows_written[c];
                }
            }
            zune_jpeg::RawImcuRowStatus::Complete => self.complete = true,
            zune_jpeg::RawImcuRowStatus::NeedMoreInput => return Err(err("truncated JPEG")),
            _ => return Err(err("unexpected JPEG decoder state")),
        }
        Ok(())
    }

    fn ready(&self, y: usize) -> bool {
        (0..self.planes.len()).all(|c| {
            let (a, b) = needed_rows(self.planes[c].height, self.fy[c], y);
            self.received[c] > a.max(b)
        })
    }

    /// Forget component rows no later output row needs.
    fn drop_rows(&mut self) {
        if self.next_out >= self.height {
            return;
        }
        for c in 0..self.planes.len() {
            let (a, b) = needed_rows(self.planes[c].height, self.fy[c], self.next_out);
            while self.first[c] < a.min(b) {
                self.rows[c].pop_front();
                self.first[c] += 1;
            }
        }
    }
}

/// Component rows needed for output row `y`: the nearest sample row and, for
/// vertically up-sampled components, its neighbour on that side.
fn needed_rows(h: usize, fy: usize, y: usize) -> (usize, usize) {
    if fy == 1 {
        return (y.min(h - 1), y.min(h - 1));
    }
    let near = (y / 2).min(h - 1);
    let far = if y.is_multiple_of(2) { near.saturating_sub(1) } else { (near + 1).min(h - 1) };
    (near, far)
}

/// Read-only view of the decoded component rows, so that output rows can be
/// converted in parallel.
struct RowsView<'a> {
    rows: &'a [VecDeque<Vec<u8>>],
    first: &'a [usize],
    heights: Vec<usize>,
    fx: &'a [usize],
    fy: &'a [usize],
    width: usize,
    colour: JpegColour,
}

impl RowsView<'_> {
    /// Rows (nearest, neighbour) of component `c` for output row `y`.
    fn src(&self, c: usize, y: usize) -> (&[u8], &[u8]) {
        let (near, far) = needed_rows(self.heights[c], self.fy[c], y);
        let r = &self.rows[c];
        (&r[near - self.first[c]], &r[far - self.first[c]])
    }

    /// Up-sampled sample `x` of a component: zune-jpeg's "fancy" triangle
    /// filter, vertical (rounded) first, then horizontal.
    #[inline(always)]
    fn sample(near: &[u8], far: &[u8], vert: bool, horiz: bool, x: usize) -> u8 {
        let v = |i: usize| -> u16 {
            if vert { (3 * near[i] as u16 + far[i] as u16 + 2) >> 2 } else { near[i] as u16 }
        };
        if !horiz {
            return v(x) as u8;
        }
        let n = near.len();
        let i = (x / 2).min(n - 1);
        let j = if x.is_multiple_of(2) { i.saturating_sub(1) } else { (i + 1).min(n - 1) };
        ((3 * v(i) + v(j) + 2) >> 2) as u8
    }

    fn convert_row(&self, y: usize, dst: &mut [u8]) {
        let comp = |c: usize| {
            let (n, f) = self.src(c, y);
            (n, f, self.fy[c] == 2, self.fx[c] == 2)
        };
        match self.colour {
            JpegColour::Gray => {
                let (n, f, v, h) = comp(0);
                for (x, d) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let g = Self::sample(n, f, v, h, x);
                    d.copy_from_slice(&[g, g, g]);
                }
            }
            JpegColour::Rgb | JpegColour::YCbCr => {
                let (c0, c1, c2) = (comp(0), comp(1), comp(2));
                let ycc = self.colour == JpegColour::YCbCr;
                let clamp = |v: i32| v.clamp(0, 255) as u8;
                for (x, d) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate().take(self.width) {
                    let a = Self::sample(c0.0, c0.1, c0.2, c0.3, x);
                    let b = Self::sample(c1.0, c1.1, c1.2, c1.3, x);
                    let c = Self::sample(c2.0, c2.1, c2.2, c2.3, x);
                    if ycc {
                        let y0 = a as i32 * Y_CF + YUV_RND;
                        let (cb, cr) = (b as i32 - 128, c as i32 - 128);
                        d[0] = clamp((y0 + cr * CR_CF) >> 14);
                        d[1] = clamp((y0 + cr * G_CR + cb * G_CB) >> 14);
                        d[2] = clamp((y0 + cb * CB_CF) >> 14);
                    } else {
                        d.copy_from_slice(&[a, b, c]);
                    }
                }
            }
        }
    }
}

impl RowReader for JpegRows {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let start = self.next_out;
        let mut end = (start + max).min(self.height);
        if start >= end {
            return Ok(0);
        }
        // Decode iMCU rows until the whole block of output rows is ready.
        while !self.ready(end - 1) {
            if self.complete {
                while end > start && !self.ready(end - 1) {
                    end -= 1;
                }
                if end == start {
                    return Err(err("JPEG data ended early"));
                }
                break;
            }
            self.pull()?;
        }
        // Up-sample and convert the block, in parallel where possible.
        let row = self.width * 3;
        let base = out.len();
        out.resize(base + (end - start) * row, 0);
        let view = RowsView {
            rows: &self.rows,
            first: &self.first,
            heights: self.planes.iter().map(|p| p.height).collect(),
            fx: &self.fx,
            fy: &self.fy,
            width: self.width,
            colour: self.colour,
        };
        let per = (end - start).div_ceil(crate::loader::threads()).max(1) * row;
        crate::loader::par_chunks(&mut out[base..], per, |chunk, off| {
            for (i, dst) in chunk.chunks_exact_mut(row).enumerate() {
                view.convert_row(start + off / row + i, dst);
            }
        });
        self.next_out = end;
        self.drop_rows();
        Ok(end - start)
    }
}

/// Streaming reader for baseline/sequential JPEGs in grey, YCbCr or RGB with
/// 1×/2× chroma subsampling; `None` for anything else (progressive, CMYK, …),
/// which is decoded as a whole instead.
fn open_jpeg(mut input: Input) -> Result<Option<Decoded>, String> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let opts = DecoderOptions::default().set_max_width(usize::from(u16::MAX)).set_max_height(usize::from(u16::MAX));
    let pos = input.stream_position().map_err(|e| e.to_string())?;
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(PeekFriendly { inner: input, pos }, opts);
    if dec.decode_headers().is_err() {
        return Ok(None);
    }
    let Some(info) = dec.info() else { return Ok(None) };
    let sequential = info.sof.is_sequential_dct();
    let colour = match (dec.input_colorspace(), info.components) {
        (Some(ColorSpace::Luma), 1) => JpegColour::Gray,
        (Some(ColorSpace::YCbCr), 3) => JpegColour::YCbCr,
        (Some(ColorSpace::RGB), 3) => JpegColour::Rgb,
        _ => return Ok(None),
    };
    let (w, h) = (info.width as usize, info.height as usize);
    if !sequential || w == 0 || h == 0 {
        return Ok(None);
    }
    let Some(layout) = dec.raw_output().layout() else { return Ok(None) };
    let planes: Vec<zune_jpeg::PlaneInfo> = layout[..info.components as usize].to_vec();
    let hmax = planes.iter().map(|p| p.horizontal_sampling_factor).max().unwrap_or(1);
    let vmax = planes.iter().map(|p| p.vertical_sampling_factor).max().unwrap_or(1);
    let factor = |max: usize, f: usize| (f > 0 && max.is_multiple_of(f)).then(|| max / f).filter(|r| *r <= 2);
    let fx: Option<Vec<usize>> = planes.iter().map(|p| factor(hmax, p.horizontal_sampling_factor)).collect();
    let fy: Option<Vec<usize>> = planes.iter().map(|p| factor(vmax, p.vertical_sampling_factor)).collect();
    let (Some(fx), Some(fy)) = (fx, fy) else { return Ok(None) };
    check_pixels(w as u32, h as u32, MAX_STREAM_PIXELS)?;
    let icc = dec.icc_profile();
    let n = planes.len();
    let imcu = planes.iter().map(|p| vec![0u8; p.stride * p.vertical_sampling_factor * 8]).collect();
    let rows = JpegRows {
        dec,
        width: w,
        height: h,
        colour,
        planes,
        fx,
        fy,
        rows: vec![VecDeque::new(); n],
        first: vec![0; n],
        received: vec![0; n],
        imcu,
        next_out: 0,
        complete: false,
    };
    Ok(Some(Decoded { width: w as u32, height: h as u32, icc, rows: Box::new(rows), streaming: true }))
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

    if head.starts_with(&[0xFF, 0xD8]) {
        if let Some(d) = open_jpeg(input)? {
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

    // WebP and anything else: decoded at once.
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
    /// Seconds spent cutting tiles and halving rows (for the load log).
    pub time_tiles: f64,
    pub time_halve: f64,
}

impl PyramidBuilder {
    pub fn new(grid: TileGrid) -> Self {
        let bands = grid
            .levels
            .iter()
            .map(|l| Band { w: l.w as usize, h: l.h as usize, rows: VecDeque::new(), first: 0, received: 0, next_tile_row: 0 })
            .collect();
        Self { grid, bands, tiles: Vec::new(), time_tiles: 0.0, time_halve: 0.0 }
    }

    /// Feed full-resolution RGB8 rows (any number, top to bottom).
    pub fn push_rows(&mut self, data: &[u8], lut: &Lut) {
        let row_len = self.bands[0].w * 3;
        let rows: Vec<Vec<u8>> = data.chunks_exact(row_len).map(<[u8]>::to_vec).collect();
        self.push_batch(0, rows, lut);
    }

    /// Tiles finished since the last call: RGB8, `sx × sy` pixels incl. gutter.
    pub fn take_tiles(&mut self) -> Vec<(TileId, Vec<u8>)> {
        std::mem::take(&mut self.tiles)
    }

    /// True once every level has received all its rows.
    pub fn is_complete(&self) -> bool {
        self.bands.iter().all(|b| b.received == b.h)
    }

    /// Add rows to level `k`, emit finished tiles, and pass the rows halved
    /// to the next level.
    fn push_batch(&mut self, k: usize, rows: Vec<Vec<u8>>, lut: &Lut) {
        if rows.is_empty() {
            return;
        }
        let start = self.bands[k].received;
        {
            let b = &mut self.bands[k];
            debug_assert!(rows.iter().all(|r| r.len() == b.w * 3));
            b.received += rows.len();
            b.rows.extend(rows);
        }
        let t = web_time::Instant::now();
        self.emit_ready(k);
        self.time_tiles += t.elapsed().as_secs_f64();

        if k + 1 < self.bands.len() {
            // Row pairs completed by this batch (an odd last row is paired
            // with itself); the first row of a pair may be from the previous
            // batch, which is why the latest row is always kept.
            let b = &self.bands[k];
            let pairs: Vec<(usize, usize)> =
                (start..b.received).filter(|&y| y % 2 == 1 || y == b.h - 1).map(|y| (y - y % 2, y)).collect();
            let t = web_time::Instant::now();
            let nw = b.w.div_ceil(2);
            let mut halved: Vec<Vec<u8>> = vec![vec![0u8; nw * 3]; pairs.len()];
            crate::loader::par_each(&mut halved, 16, |i, dst| {
                let (a, c) = pairs[i];
                halve_rows_into(b.row(a), b.row(c), b.w, lut, dst);
            });
            self.time_halve += t.elapsed().as_secs_f64();
            self.push_batch(k + 1, halved, lut);
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
            let mut tiles: Vec<Vec<u8>> = vec![vec![0u8; sx * sy * 3]; level.nx as usize];
            crate::loader::par_each(&mut tiles, 1, |tx, rgb| {
                let x0 = ((tx * cx) as i64 - g as i64).rem_euclid(b.w as i64) as usize;
                for (j, dst) in rgb.chunks_exact_mut(sx * 3).enumerate() {
                    let y = ((r * cy) as i64 - g as i64 + j as i64).clamp(0, b.h as i64 - 1) as usize;
                    let src = b.row(y);
                    // Copy whole runs of pixels, wrapping around at the seam.
                    let (mut x, mut filled) = (x0, 0);
                    while filled < sx {
                        let n = (sx - filled).min(b.w - x);
                        dst[filled * 3..(filled + n) * 3].copy_from_slice(&src[x * 3..(x + n) * 3]);
                        filled += n;
                        x = 0;
                    }
                }
            });
            self.tiles.extend(
                tiles.into_iter().enumerate().map(|(tx, rgb)| (TileId { level: k as u8, tx: tx as u16, ty: r as u16 }, rgb)),
            );
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
fn halve_rows_into(a: &[u8], b: &[u8], w: usize, lut: &Lut, out: &mut [u8]) {
    for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let (p, q) = (2 * x, (2 * x + 1).min(w - 1));
        for c in 0..3 {
            o[c] = lut.avg4(a[p * 3 + c], a[q * 3 + c], b[p * 3 + c], b[q * 3 + c]);
        }
    }
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

    fn open_bytes(buf: Vec<u8>) -> Decoded {
        let data: std::sync::Arc<[u8]> = buf.into();
        open_rows(move || Ok(BufReader::new(Box::new(std::io::Cursor::new(data.clone())) as Box<dyn ReadSeek>))).unwrap()
    }

    fn read_all(d: &mut Decoded, chunk: usize) -> Vec<u8> {
        let mut out = Vec::new();
        while d.rows.read(chunk, &mut out).unwrap() > 0 {}
        out
    }

    #[test]
    fn baseline_jpeg_streams_and_matches_full_decode() {
        use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
        // Odd sizes; colourful content so chroma up-sampling matters.
        let (w, h) = (517u16, 263u16);
        let img = test_image(w as u32, h as u32);
        let gray: Vec<u8> = img.pixels().map(|p| p.0[0]).collect();
        let cases = [
            ("4:2:0", Some(SamplingFactor::R_4_2_0)),
            ("4:2:2", Some(SamplingFactor::R_4_2_2)),
            ("4:4:4", Some(SamplingFactor::R_4_4_4)),
            ("4:4:0", Some(SamplingFactor::R_4_4_0)),
            ("grey", None),
        ];
        for (name, sampling) in cases {
            let mut jpeg = Vec::new();
            let mut enc = Encoder::new(&mut jpeg, 90);
            match sampling {
                Some(s) => {
                    enc.set_sampling_factor(s);
                    enc.encode(img.as_raw(), w, h, ColorType::Rgb).unwrap();
                }
                None => enc.encode(&gray, w, h, ColorType::Luma).unwrap(),
            }
            // Reference: zune-jpeg's own full decode to RGB.
            let opts = zune_jpeg::zune_core::options::DecoderOptions::default()
                .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGB);
            let reference = zune_jpeg::JpegDecoder::new_with_options(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg), opts)
                .decode()
                .unwrap();
            let mut d = open_bytes(jpeg);
            assert!(d.streaming, "{name}: not streamed");
            let got = read_all(&mut d, 7);
            assert_eq!(got.len(), reference.len(), "{name}");
            let max = got.iter().zip(&reference).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            let differing = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
            // The raw rows are already rounded to 8 bits before up-sampling,
            // the full decoder up-samples the 16-bit IDCT output: a few
            // chroma-interpolated values differ by one.
            assert!(max <= 1 && differing * 500 < got.len(), "{name}: max difference {max}, {differing} values differ");
        }
    }

    /// `cargo test --release jpeg_row_speed -- --ignored --nocapture` with
    /// `RUSTYPANO_BENCH_JPEG=<file>`.
    #[test]
    #[ignore]
    fn jpeg_row_speed() {
        let Ok(path) = std::env::var("RUSTYPANO_BENCH_JPEG") else { return };
        let t = std::time::Instant::now();
        let mut d = open_rows(|| Ok(BufReader::new(Box::new(std::fs::File::open(&path).unwrap()) as Box<dyn ReadSeek>))).unwrap();
        let mut out = Vec::new();
        let mut rows = 0;
        loop {
            out.clear();
            let n = d.rows.read(139, &mut out).unwrap();
            if n == 0 {
                break;
            }
            rows += n;
        }
        println!("{rows} rows ({}x{}, streaming {}) in {:.2}s", d.width, d.height, d.streaming, t.elapsed().as_secs_f64());
        // Compare a band of rows with both full decoders.
        let (y0, n) = (d.height as usize / 2, 64usize);
        let mut d = open_rows(|| Ok(BufReader::new(Box::new(std::fs::File::open(&path).unwrap()) as Box<dyn ReadSeek>))).unwrap();
        let mut band = Vec::new();
        let mut skip = Vec::new();
        let mut y = 0;
        while y < y0 {
            skip.clear();
            y += d.rows.read((y0 - y).min(256), &mut skip).unwrap();
        }
        d.rows.read(n, &mut band).unwrap();
        let row = d.width as usize * 3;
        let data = std::fs::read(&path).unwrap();
        let opts = zune_jpeg::zune_core::options::DecoderOptions::default()
            .set_max_width(1 << 16)
            .set_max_height(1 << 16)
            .jpeg_set_out_colorspace(zune_jpeg::zune_core::colorspace::ColorSpace::RGB);
        let rc = zune_jpeg::JpegDecoder::new_with_options(zune_jpeg::zune_core::bytestream::ZCursor::new(&data), opts).decode().unwrap();
        let mut reader = image::ImageReader::new(std::io::Cursor::new(&data)).with_guessed_format().unwrap();
        reader.no_limits();
        let img = reader.decode().unwrap().into_rgb8();
        for (name, full) in [("zune-jpeg rc", &rc[..]), ("image (zune-jpeg 0.5.15)", img.as_raw())] {
            let reference = &full[y0 * row..(y0 + n) * row];
            let max = band.iter().zip(reference).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            let diff = band.iter().zip(reference).filter(|(a, b)| a != b).count();
            println!("vs {name}: max difference {max}, {diff} of {} values differ", band.len());
        }
    }

    #[test]
    fn progressive_jpeg_falls_back_to_full_decode() {
        let img = test_image(120, 60);
        let mut jpeg = Vec::new();
        let mut enc = jpeg_encoder::Encoder::new(&mut jpeg, 90);
        enc.set_progressive(true);
        enc.encode(img.as_raw(), 120, 60, jpeg_encoder::ColorType::Rgb).unwrap();
        let mut d = open_bytes(jpeg);
        assert!(!d.streaming);
        assert_eq!(read_all(&mut d, 16).len(), 120 * 60 * 3);
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
