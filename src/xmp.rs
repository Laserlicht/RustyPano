// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Minimal XMP handling for Google Photo Sphere (GPano) tags.
//! Reads from JPEG (APP1), TIFF (tag 700) and PNG (iTXt); writes to JPEG.

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

const XMP_SIG: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // used for writing (desktop only)
const GPANO_NS: &str = "http://ns.google.com/photos/1.0/panorama/";

/// Tags the viewer understands, in display order.
pub const GPANO_TAGS: &[&str] = &[
    "ProjectionType",
    "UsePanoramaViewer",
    "PoseHeadingDegrees",
    "InitialViewHeadingDegrees",
    "InitialViewPitchDegrees",
    "InitialHorizontalFOVDegrees",
    "FullPanoWidthPixels",
    "FullPanoHeightPixels",
    "CroppedAreaImageWidthPixels",
    "CroppedAreaImageHeightPixels",
    "CroppedAreaLeftPixels",
    "CroppedAreaTopPixels",
];

#[derive(Clone, Debug, Default)]
pub struct GPano {
    pub tags: BTreeMap<String, String>,
}

impl GPano {
    pub fn get(&self, name: &str) -> Option<f64> {
        self.tags.get(name).and_then(|v| v.trim().parse::<f64>().ok())
    }
    pub fn set(&mut self, name: &str, value: impl ToString) {
        self.tags.insert(name.to_string(), value.to_string());
    }
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// GPano tags of an image file's contents; `ext` is the lower-case file extension.
    pub fn from_bytes(data: &[u8], ext: &str) -> GPano {
        Self::parse(&read_xmp(data, ext).unwrap_or_default())
    }

    /// GPano tags of a local file, reading only the parts that can hold XMP
    /// (panoramas can be many gigabytes).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(path: &Path) -> GPano {
        use std::io::{Read, Seek, SeekFrom};
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let Ok(mut f) = std::fs::File::open(path) else { return GPano::default() };
        if ext == "tif" || ext == "tiff" {
            let Ok(mut d) = tiff::decoder::Decoder::new(std::io::BufReader::new(f)) else { return GPano::default() };
            let v = d.get_tag_u8_vec(tiff::tags::Tag::Unknown(700)).unwrap_or_default();
            return Self::parse(&String::from_utf8_lossy(&v));
        }
        // JPEG and PNG keep XMP near the start; WebP and JPEG XL often at the end.
        const PART: u64 = 16 << 20;
        let len = f.metadata().map_or(0, |m| m.len());
        let mut data = Vec::new();
        let _ = (&mut f).take(PART).read_to_end(&mut data);
        let read_rest = if len > 2 * PART {
            matches!(ext.as_str(), "webp" | "jxl") && f.seek(SeekFrom::End(-(PART as i64))).is_ok()
        } else {
            len > PART
        };
        if read_rest {
            let _ = f.read_to_end(&mut data);
        }
        Self::from_bytes(&data, &ext)
    }

    pub fn parse(xmp: &str) -> GPano {
        let mut g = GPano::default();
        // Attribute form: GPano:Name="value"   Element form: <GPano:Name>value</GPano:Name>
        let mut rest = xmp;
        while let Some(i) = rest.find("GPano:") {
            let after = &rest[i + 6..];
            let name_len = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            let name = &after[..name_len];
            let tail = &after[name_len..];
            let prev = rest[..i].chars().last();
            if let Some(t) = tail.strip_prefix('=') {
                let t = t.trim_start();
                if let Some(q) = t.chars().next().filter(|c| *c == '"' || *c == '\'')
                    && let Some(end) = t[1..].find(q) {
                        g.tags.insert(name.to_string(), t[1..1 + end].to_string());
                    }
            } else if prev == Some('<') && tail.starts_with('>')
                && let Some(end) = tail[1..].find('<') {
                    g.tags.insert(name.to_string(), tail[1..1 + end].trim().to_string());
                }
            rest = &after[name_len..];
        }
        g
    }
}

fn read_xmp(data: &[u8], ext: &str) -> Option<String> {
    match ext {
        "jpg" | "jpeg" | "jpe" => read_jpeg_xmp(data),
        "tif" | "tiff" => {
            let mut d = tiff::decoder::Decoder::new(std::io::Cursor::new(data)).ok()?;
            let v = d.get_tag_u8_vec(tiff::tags::Tag::Unknown(700)).ok()?;
            Some(String::from_utf8_lossy(&v).into_owned())
        }
        "png" => read_png_xmp(data),
        // WebP (RIFF "XMP " chunk) and JPEG XL ("xml " box) store the packet
        // uncompressed in most files: scan for it.
        "webp" | "jxl" => scan_xmp(data),
        _ => None,
    }
}

fn scan_xmp(data: &[u8]) -> Option<String> {
    let start = data.windows(10).position(|w| w == b"<x:xmpmeta")?;
    let end_tag = b"</x:xmpmeta>";
    let len = data[start..].windows(end_tag.len()).position(|w| w == end_tag)? + end_tag.len();
    Some(String::from_utf8_lossy(&data[start..start + len]).into_owned())
}

fn read_jpeg_xmp(data: &[u8]) -> Option<String> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut pos = 2;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            return None;
        }
        let marker = data[pos + 1];
        if marker == 0xD9 || marker == 0xDA {
            return None;
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 || marker == 0xFF {
            pos += if marker == 0xFF { 1 } else { 2 };
            continue;
        }
        let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        if len < 2 || pos + 2 + len > data.len() {
            return None;
        }
        let payload = &data[pos + 4..pos + 2 + len];
        if marker == 0xE1 && payload.starts_with(XMP_SIG) {
            return Some(String::from_utf8_lossy(&payload[XMP_SIG.len()..]).into_owned());
        }
        pos += 2 + len;
    }
    None
}

fn read_png_xmp(data: &[u8]) -> Option<String> {
    let mut pos = 8;
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        let ty = &data[pos + 4..pos + 8];
        if ty == b"IDAT" || ty == b"IEND" {
            return None;
        }
        let body = data.get(pos + 8..pos + 8 + len)?;
        if ty == b"iTXt" && body.starts_with(b"XML:com.adobe.xmp\0") {
            // keyword\0 compression_flag compression_method lang\0 translated\0 text
            let mut p = 18 + 2;
            for _ in 0..2 {
                p += body[p..].iter().position(|&b| b == 0)? + 1;
            }
            return Some(String::from_utf8_lossy(&body[p..]).into_owned());
        }
        pos += 12 + len;
    }
    None
}

/// Set one GPano property inside an XMP packet, creating it if missing.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn set_in_xmp(xmp: &mut String, name: &str, value: &str) {
    let attr = format!("GPano:{name}=\"");
    if let Some(i) = xmp.find(&attr) {
        let start = i + attr.len();
        if let Some(end) = xmp[start..].find('"') {
            xmp.replace_range(start..start + end, value);
            return;
        }
    }
    let open = format!("<GPano:{name}>");
    let close = format!("</GPano:{name}>");
    if let (Some(i), Some(j)) = (xmp.find(&open), xmp.find(&close))
        && j > i {
            xmp.replace_range(i + open.len()..j, value);
            return;
        }
    if let Some(i) = xmp.find("<rdf:Description") {
        let at = i + "<rdf:Description".len();
        xmp.insert_str(at, &format!("\n    GPano:{name}=\"{value}\""));
    }
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn ensure_packet(xmp: &str) -> String {
    let mut s = if xmp.contains("<rdf:Description") {
        xmp.to_string()
    } else {
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n\
             <x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n\
             <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\
             <rdf:Description rdf:about=\"\"/>\n\
             </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end=\"w\"?>".to_string()
    };
    if !s.contains(GPANO_NS)
        && let Some(i) = s.find("<rdf:Description") {
            s.insert_str(i + "<rdf:Description".len(), &format!(" xmlns:GPano=\"{GPANO_NS}\""));
        }
    s
}

/// Rewrite the JPEG at `path` with the given GPano tags merged into its XMP.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_gpano_jpeg(path: &Path, tags: &GPano) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(tr!("Tags can only be written into JPEG files").into());
    }
    // Walk segments up to SOS.
    let mut pos = 2usize;
    let mut xmp_seg: Option<(usize, usize)> = None; // (start of marker, end)
    let mut insert_at = 2usize;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            return Err(tr!("Corrupt JPEG header").into());
        }
        let marker = data[pos + 1];
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        let end = pos + 2 + len;
        if end > data.len() {
            return Err(tr!("Corrupt JPEG header").into());
        }
        let payload = &data[pos + 4..end];
        if marker == 0xE1 && payload.starts_with(XMP_SIG) {
            xmp_seg = Some((pos, end));
        }
        if marker == 0xE0 || (marker == 0xE1 && !payload.starts_with(XMP_SIG)) {
            insert_at = end;
        }
        pos = end;
    }

    let old = xmp_seg
        .map(|(s, e)| String::from_utf8_lossy(&data[s + 4 + XMP_SIG.len()..e]).into_owned())
        .unwrap_or_default();
    let mut xmp = ensure_packet(&old);
    for (k, v) in &tags.tags {
        set_in_xmp(&mut xmp, k, v);
    }
    let mut seg = Vec::new();
    let body_len = 2 + XMP_SIG.len() + xmp.len();
    if body_len > 0xFFFF {
        return Err(tr!("XMP packet too large for a single JPEG segment").into());
    }
    seg.extend_from_slice(&[0xFF, 0xE1]);
    seg.extend_from_slice(&(body_len as u16).to_be_bytes());
    seg.extend_from_slice(XMP_SIG);
    seg.extend_from_slice(xmp.as_bytes());

    let mut out = Vec::with_capacity(data.len() + seg.len());
    match xmp_seg {
        Some((s, e)) => {
            out.extend_from_slice(&data[..s]);
            out.extend_from_slice(&seg);
            out.extend_from_slice(&data[e..]);
        }
        None => {
            out.extend_from_slice(&data[..insert_at]);
            out.extend_from_slice(&seg);
            out.extend_from_slice(&data[insert_at..]);
        }
    }
    let tmp = path.with_extension("rustypano-tmp");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&out).map_err(|e| e.to_string())?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_both_forms() {
        let x = r#"<rdf:Description GPano:PoseHeadingDegrees="12.5" xmlns:GPano="x">
            <GPano:CroppedAreaTopPixels>100</GPano:CroppedAreaTopPixels></rdf:Description>"#;
        let g = GPano::parse(x);
        assert_eq!(g.get("PoseHeadingDegrees"), Some(12.5));
        assert_eq!(g.get("CroppedAreaTopPixels"), Some(100.0));
    }

    #[test]
    fn jpeg_roundtrip() {
        let dir = std::env::temp_dir().join("rustypano_xmp_test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.jpg");
        image::RgbImage::from_pixel(64, 32, image::Rgb([10, 20, 30])).save(&f).unwrap();
        assert!(GPano::read(&f).is_empty());
        let mut g = GPano::default();
        g.set("InitialViewHeadingDegrees", "90");
        g.set("CroppedAreaTopPixels", 4);
        write_gpano_jpeg(&f, &g).unwrap();
        let r = GPano::read(&f);
        assert_eq!(r.get("InitialViewHeadingDegrees"), Some(90.0));
        // second write updates in place, image still decodes
        g.set("InitialViewHeadingDegrees", "45");
        write_gpano_jpeg(&f, &g).unwrap();
        assert_eq!(GPano::read(&f).get("InitialViewHeadingDegrees"), Some(45.0));
        assert_eq!(GPano::read(&f).get("CroppedAreaTopPixels"), Some(4.0));
        let img = image::open(&f).unwrap();
        assert_eq!((img.width(), img.height()), (64, 32));
        let p = crate::fsv::PanoParams::from_image(crate::source::Source::Path(f.clone()), &GPano::read(&f));
        assert_eq!(p.yaw, Some(225.0)); // 180 - pose(0) + 45
    }

    #[test]
    fn set_creates_and_replaces() {
        let mut s = ensure_packet("");
        set_in_xmp(&mut s, "PoseHeadingDegrees", "10");
        set_in_xmp(&mut s, "PoseHeadingDegrees", "20");
        let g = GPano::parse(&s);
        assert_eq!(g.get("PoseHeadingDegrees"), Some(20.0));
        assert!(s.contains(GPANO_NS));
    }
}
