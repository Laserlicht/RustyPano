// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Photoshop documents (PSD, and PSB for large documents): the merged image
//! stored after the layers, read row by row. Layers are not composited.
//!
//! The merged image is stored channel after channel, so every channel gets a
//! reader of its own that moves through its rows in step with the others.

use std::io::{Read, Seek, SeekFrom};

use crate::pyramid::{Input, RowReader};

fn err(e: impl std::fmt::Display) -> String {
    tr!("Cannot decode this image: {error}", error = e)
}

pub fn is_psd(head: &[u8]) -> bool {
    head.starts_with(b"8BPS\0\x01") || head.starts_with(b"8BPS\0\x02")
}

const MODE_GRAY: u16 = 1;
const MODE_RGB: u16 = 3;

/// Header and the parts of the file the viewer needs.
#[derive(Debug)]
pub struct PsdInfo {
    pub width: u32,
    pub height: u32,
    channels: usize,
    depth: u16,
    mode: u16,
    /// PSB: 4-byte row lengths and section sizes.
    large: bool,
    pub icc: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    /// Offset of the merged image data (its compression field).
    image_data: u64,
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_be_bytes(b))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_be_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_be_bytes(b))
}

/// Parse the header and image resources (ICC profile, XMP).
pub fn read_info(r: &mut (impl Read + Seek)) -> Result<PsdInfo, String> {
    let io = |e: std::io::Error| err(e);
    let mut sig = [0; 4];
    r.read_exact(&mut sig).map_err(io)?;
    let version = read_u16(r).map_err(io)?;
    if &sig != b"8BPS" || !(1..=2).contains(&version) {
        return Err(err("not a Photoshop document"));
    }
    r.seek(SeekFrom::Current(6)).map_err(io)?;
    let channels = read_u16(r).map_err(io)? as usize;
    let height = read_u32(r).map_err(io)?;
    let width = read_u32(r).map_err(io)?;
    let depth = read_u16(r).map_err(io)?;
    let mode = read_u16(r).map_err(io)?;
    let large = version == 2;

    // Colour mode data (palettes, duotone curves).
    let len = read_u32(r).map_err(io)?;
    r.seek(SeekFrom::Current(len as i64)).map_err(io)?;

    // Image resources.
    let len = read_u32(r).map_err(io)? as u64;
    let start = r.stream_position().map_err(io)?;
    let (mut icc, mut xmp) = (None, None);
    while r.stream_position().map_err(io)? + 12 <= start + len {
        let mut sig = [0; 4];
        r.read_exact(&mut sig).map_err(io)?;
        let id = read_u16(r).map_err(io)?;
        // Pascal string name, padded to an even length.
        let mut n = [0u8; 1];
        r.read_exact(&mut n).map_err(io)?;
        let name = (n[0] as u64 + 1).next_multiple_of(2) as i64 - 1;
        r.seek(SeekFrom::Current(name)).map_err(io)?;
        let size = read_u32(r).map_err(io)? as u64;
        if size > len {
            break;
        }
        if id == 1039 || id == 1060 {
            let mut data = vec![0; size as usize];
            r.read_exact(&mut data).map_err(io)?;
            *(if id == 1039 { &mut icc } else { &mut xmp }) = Some(data);
            r.seek(SeekFrom::Current((size % 2) as i64)).map_err(io)?;
        } else {
            r.seek(SeekFrom::Current(size.next_multiple_of(2) as i64)).map_err(io)?;
        }
    }
    r.seek(SeekFrom::Start(start + len)).map_err(io)?;

    // Layer and mask information.
    let len = if large { read_u64(r).map_err(io)? } else { read_u32(r).map_err(io)? as u64 };
    let image_data = r.stream_position().map_err(io)? + len;
    Ok(PsdInfo { width, height, channels, depth, mode, large, icc, xmp, image_data })
}

/// XMP packet of a PSD/PSB file whose start is in `data`.
pub fn xmp(data: &[u8]) -> Option<String> {
    let info = read_info(&mut std::io::Cursor::new(data)).ok()?;
    info.xmp.map(|x| String::from_utf8_lossy(&x).into_owned())
}

/// Reads the merged image; `reopen` gives a fresh reader per channel.
pub fn open(mut reopen: impl FnMut() -> Result<Input, String>) -> Result<(PsdInfo, Box<dyn RowReader>), String> {
    let io = |e: std::io::Error| err(e);
    let mut r = reopen()?;
    let info = read_info(&mut r)?;
    let used = match info.mode {
        MODE_RGB if info.channels >= 3 => 3,
        MODE_GRAY if info.channels >= 1 => 1,
        m => {
            let name = match m {
                0 => "bitmap",
                2 => "indexed",
                4 => "CMYK",
                7 => "multichannel",
                8 => "duotone",
                9 => "Lab",
                _ => "unknown",
            };
            return Err(err(format!("Photoshop colour mode {name} (only RGB and grayscale are supported)")));
        }
    };
    if info.depth != 8 && info.depth != 16 {
        return Err(err(format!("Photoshop documents with {} bits per channel", info.depth)));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 {
        return Err(err("empty image"));
    }
    let row_bytes = w * info.depth as usize / 8;

    r.seek(SeekFrom::Start(info.image_data)).map_err(io)?;
    let compression = read_u16(&mut r).map_err(io)?;
    let data = info.image_data + 2;
    let mut channels = Vec::with_capacity(used);
    match compression {
        0 => {
            for c in 0..used {
                let mut cr = reopen()?;
                cr.seek(SeekFrom::Start(data + (c * h * row_bytes) as u64)).map_err(io)?;
                channels.push(Channel { r: cr, lengths: Vec::new(), next: 0, packed: Vec::new() });
            }
        }
        1 => {
            // PackBits: first the compressed length of every row of every
            // channel, then the rows.
            let size = if info.large { 4 } else { 2 };
            let mut table = vec![0u8; info.channels * h * size];
            r.read_exact(&mut table).map_err(io)?;
            let lengths: Vec<u32> = if info.large {
                table.as_chunks::<4>().0.iter().map(|b| u32::from_be_bytes(*b)).collect()
            } else {
                table.as_chunks::<2>().0.iter().map(|b| u16::from_be_bytes(*b) as u32).collect()
            };
            let mut offset = data + table.len() as u64;
            for c in 0..used {
                let mut cr = reopen()?;
                cr.seek(SeekFrom::Start(offset)).map_err(io)?;
                let ls = lengths[c * h..(c + 1) * h].to_vec();
                offset += ls.iter().map(|&l| l as u64).sum::<u64>();
                channels.push(Channel { r: cr, lengths: ls, next: 0, packed: Vec::new() });
            }
        }
        2 | 3 => return Err(err("ZIP-compressed Photoshop image data")),
        c => return Err(err(format!("Photoshop compression {c}"))),
    }
    let rows = PsdRows { channels, width: w, height: h, row_bytes, sixteen: info.depth == 16, y: 0 };
    Ok((info, Box::new(rows)))
}

struct Channel {
    r: Input,
    /// Compressed row lengths (empty: uncompressed).
    lengths: Vec<u32>,
    next: usize,
    packed: Vec<u8>,
}

impl Channel {
    /// Read `n` rows of `row_bytes` into `out`.
    fn read_rows(&mut self, n: usize, row_bytes: usize, out: &mut [u8]) -> Result<(), String> {
        if self.lengths.is_empty() {
            return self.r.read_exact(&mut out[..n * row_bytes]).map_err(err);
        }
        for dst in out.chunks_exact_mut(row_bytes).take(n) {
            let len = self.lengths[self.next] as usize;
            self.next += 1;
            self.packed.resize(len, 0);
            self.r.read_exact(&mut self.packed).map_err(err)?;
            unpack_bits(&self.packed, dst)?;
        }
        Ok(())
    }
}

/// Decode a PackBits row; the row must be filled exactly.
fn unpack_bits(mut src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    let mut o = 0;
    while o < dst.len() {
        let (&n, rest) = src.split_first().ok_or_else(|| err("RLE row too short"))?;
        src = rest;
        let n = n as i8;
        if n >= 0 {
            let k = n as usize + 1;
            if src.len() < k || o + k > dst.len() {
                return Err(err("corrupt RLE row"));
            }
            dst[o..o + k].copy_from_slice(&src[..k]);
            src = &src[k..];
            o += k;
        } else if n != -128 {
            let k = (1 - n as isize) as usize;
            let (&v, rest) = src.split_first().ok_or_else(|| err("RLE row too short"))?;
            src = rest;
            if o + k > dst.len() {
                return Err(err("corrupt RLE row"));
            }
            dst[o..o + k].fill(v);
            o += k;
        }
    }
    Ok(())
}

struct PsdRows {
    channels: Vec<Channel>,
    width: usize,
    height: usize,
    row_bytes: usize,
    sixteen: bool,
    y: usize,
}

impl RowReader for PsdRows {
    fn read(&mut self, max: usize, out: &mut Vec<u8>) -> Result<usize, String> {
        let n = max.min(self.height - self.y);
        if n == 0 {
            return Ok(0);
        }
        // Rows of every channel, decoded in parallel.
        let row_bytes = self.row_bytes;
        let mut planes: Vec<(&mut Channel, Vec<u8>, Result<(), String>)> =
            self.channels.iter_mut().map(|c| (c, vec![0u8; n * row_bytes], Ok(()))).collect();
        crate::loader::par_each(&mut planes, 1, |_, (c, buf, res)| *res = c.read_rows(n, row_bytes, buf));
        for (_, _, res) in &mut planes {
            std::mem::replace(res, Ok(()))?;
        }
        // Interleave to RGB8 (16-bit samples are big-endian).
        let w = self.width;
        let base = out.len();
        out.resize(base + n * w * 3, 0);
        let sample = |p: &[u8], i: usize| if self.sixteen { p[2 * i] } else { p[i] };
        let src: Vec<&[u8]> = planes.iter().map(|(_, b, _)| b.as_slice()).collect();
        for (r, dst) in out[base..].chunks_exact_mut(w * 3).enumerate() {
            let rows: Vec<&[u8]> = src.iter().map(|p| &p[r * row_bytes..(r + 1) * row_bytes]).collect();
            for (x, d) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                *d = match rows.len() {
                    1 => [sample(rows[0], x); 3],
                    _ => [sample(rows[0], x), sample(rows[1], x), sample(rows[2], x)],
                };
            }
        }
        self.y += n;
        Ok(n)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    /// PackBits with repeat runs for runs of 3 or more, literals otherwise.
    fn pack_bits(src: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < src.len() {
            let run = src[i..].iter().take(128).take_while(|&&v| v == src[i]).count();
            if run >= 3 {
                out.extend_from_slice(&[(1 - run as isize) as i8 as u8, src[i]]);
                i += run;
            } else {
                let mut j = i;
                while j < src.len() && j - i < 128 && !(j + 2 < src.len() && src[j] == src[j + 1] && src[j] == src[j + 2]) {
                    j += 1;
                }
                out.push((j - i - 1) as u8);
                out.extend_from_slice(&src[i..j]);
                i = j;
            }
        }
        out
    }

    /// A minimal Photoshop document with the given planar channel data
    /// (`depth` 8 or 16), an extra alpha channel, ICC and XMP resources.
    pub(crate) fn write_psd(w: u32, h: u32, planes: &[Vec<u8>], depth: u16, large: bool, rle: bool) -> Vec<u8> {
        let mut f = Vec::new();
        let mode = if planes.len() == 1 { MODE_GRAY } else { MODE_RGB };
        let row = w as usize * depth as usize / 8;
        let mut planes = planes.to_vec();
        planes.push(vec![0xAB; row * h as usize]); // alpha, ignored
        f.extend_from_slice(b"8BPS");
        f.extend_from_slice(&(if large { 2u16 } else { 1 }).to_be_bytes());
        f.extend_from_slice(&[0; 6]);
        f.extend_from_slice(&(planes.len() as u16).to_be_bytes());
        f.extend_from_slice(&h.to_be_bytes());
        f.extend_from_slice(&w.to_be_bytes());
        f.extend_from_slice(&depth.to_be_bytes());
        f.extend_from_slice(&mode.to_be_bytes());
        f.extend_from_slice(&0u32.to_be_bytes());
        let mut res = Vec::new();
        for (id, name, data) in [(1005u16, &b"x"[..], &b"skip"[..]), (1039, b"", b"ICC!"), (1060, b"xmp", b"<x:xmpmeta GPano:PoseHeadingDegrees=\"12\"/>")] {
            res.extend_from_slice(b"8BIM");
            res.extend_from_slice(&id.to_be_bytes());
            res.push(name.len() as u8);
            res.extend_from_slice(name);
            if name.len() % 2 == 0 {
                res.push(0);
            }
            res.extend_from_slice(&(data.len() as u32).to_be_bytes());
            res.extend_from_slice(data);
            if data.len() % 2 == 1 {
                res.push(0);
            }
        }
        f.extend_from_slice(&(res.len() as u32).to_be_bytes());
        f.extend_from_slice(&res);
        let layers = [1u8, 2, 3, 4];
        if large {
            f.extend_from_slice(&(layers.len() as u64).to_be_bytes());
        } else {
            f.extend_from_slice(&(layers.len() as u32).to_be_bytes());
        }
        f.extend_from_slice(&layers);
        f.extend_from_slice(&(rle as u16).to_be_bytes());
        if rle {
            let packed: Vec<Vec<u8>> = planes.iter().flat_map(|p| p.chunks_exact(row).map(pack_bits)).collect();
            for p in &packed {
                if large {
                    f.extend_from_slice(&(p.len() as u32).to_be_bytes());
                } else {
                    f.extend_from_slice(&(p.len() as u16).to_be_bytes());
                }
            }
            packed.iter().for_each(|p| f.extend_from_slice(p));
        } else {
            planes.iter().for_each(|p| f.extend_from_slice(p));
        }
        f
    }

    fn read(file: Vec<u8>) -> Result<(PsdInfo, Vec<u8>), String> {
        let data: std::sync::Arc<[u8]> = file.into();
        let reopen = || -> Result<Input, String> {
            Ok(BufReader::new(Box::new(Cursor::new(data.clone())) as Box<dyn crate::pyramid::ReadSeek>))
        };
        let (info, mut rows) = open(reopen)?;
        let mut out = Vec::new();
        while rows.read(7, &mut out)? > 0 {}
        Ok((info, out))
    }

    #[test]
    fn reads_rgb_and_gray_in_all_variants() {
        let (w, h) = (53u32, 21u32);
        let px = |x: u32, y: u32, c: u32| -> u8 { if x % 9 < 4 { (40 * c + y) as u8 } else { (x * 5 + y * 3 + c * 70) as u8 } };
        for large in [false, true] {
            for rle in [false, true] {
                for depth in [8u16, 16] {
                    for channels in [3u32, 1] {
                        let planes: Vec<Vec<u8>> = (0..channels)
                            .map(|c| {
                                (0..h)
                                    .flat_map(|y| (0..w).flat_map(move |x| {
                                        let v = px(x, y, c);
                                        if depth == 16 { vec![v, 0x5A] } else { vec![v] }
                                    }))
                                    .collect()
                            })
                            .collect();
                        let (info, out) = read(write_psd(w, h, &planes, depth, large, rle)).unwrap();
                        assert_eq!((info.width, info.height), (w, h));
                        assert_eq!(info.icc.as_deref(), Some(&b"ICC!"[..]));
                        let expect: Vec<u8> = (0..h)
                            .flat_map(|y| (0..w).flat_map(move |x| {
                                let g = px(x, y, 0);
                                if channels == 1 { [g, g, g] } else { [g, px(x, y, 1), px(x, y, 2)] }
                            }))
                            .collect();
                        assert!(out == expect, "large {large}, rle {rle}, depth {depth}, channels {channels}");
                    }
                }
            }
        }
    }

    #[test]
    fn xmp_and_errors() {
        let f = write_psd(4, 2, &[vec![1; 8]], 8, false, true);
        assert!(xmp(&f).unwrap().contains("PoseHeadingDegrees"));
        assert!(is_psd(&f));
        let mut cmyk = f.clone();
        cmyk[24..26].copy_from_slice(&4u16.to_be_bytes());
        assert!(read(cmyk).unwrap_err().contains("CMYK"));
        let mut broken = f;
        let n = broken.len();
        broken.truncate(n - 6); // into the gray channel (the alpha channel is not read)
        assert!(read(broken).is_err());
    }
}
