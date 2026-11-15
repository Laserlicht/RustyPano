// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Parser for `.fsv` panorama description files, plus merging of
//! Google Photo Sphere (GPano) XMP tags into the same parameter set.
//!
//! Format (case-insensitive keys, `//` comments):
//! ```text
//! IMAGENAME=pano.jpg
//! WINDOWTITLE=My panorama
//! YAW=180  PITCH=0  HFOV=70  HORIZONPOSITION=50
//! MINYAW= MAXYAW= MINPITCH= MAXPITCH= AUTOSPIN=0.2
//! BEGIN HOTSPOT
//!   X=50  Y=50            (percent of image width / height)
//!   DESCRIPTION=Go inside
//!   TARGET=inside.fsv
//!   INITIALYAW=  INITIALPITCH=  INITIALHFOV=
//! END HOTSPOT
//! ```

use std::path::{Path, PathBuf};

use crate::xmp::GPano;

#[derive(Clone, Debug, Default)]
pub struct HotspotDef {
    /// Horizontal position, percent of image width (0..100).
    pub x: f64,
    /// Vertical position, percent of image height (0..100).
    pub y: f64,
    pub description: String,
    pub target: PathBuf,
    pub initial_yaw: Option<f64>,
    pub initial_pitch: Option<f64>,
    pub initial_hfov: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct PanoParams {
    pub image: PathBuf,
    pub window_title: Option<String>,
    pub yaw: Option<f64>,
    pub pitch: Option<f64>,
    pub hfov: Option<f64>,
    /// Row of the horizon, percent of image height from the top.
    pub horizon_position: Option<f64>,
    pub min_yaw: Option<f64>,
    pub max_yaw: Option<f64>,
    pub min_pitch: Option<f64>,
    pub max_pitch: Option<f64>,
    /// Degrees per frame (at 60 fps) of automatic rotation.
    pub autospin: Option<f64>,
    pub hotspots: Vec<HotspotDef>,
}

fn strip_key<'a>(line: &'a str, upper: &str, key: &str) -> Option<&'a str> {
    if upper.starts_with(key) {
        Some(line[key.len()..].trim())
    } else {
        None
    }
}

fn parse_num(v: &str, what: &str, errors: &mut Vec<String>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    match v.replace(',', ".").parse::<f64>() {
        Ok(n) => Some(n),
        Err(_) => {
            errors.push(format!("{what} is not a number: {v}"));
            None
        }
    }
}

fn resolve(base: &Path, name: &str) -> PathBuf {
    let name = name.replace('\\', "/");
    let p = PathBuf::from(&name);
    if p.is_absolute() { p } else { base.join(p) }
}

impl PanoParams {
    pub fn read_fsv(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let text = String::from_utf8(bytes.clone())
            .unwrap_or_else(|_| bytes.iter().map(|&b| b as char).collect()); // Latin-1 fallback
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut p = PanoParams::default();
        let mut errors = Vec::new();
        let mut cur: Option<HotspotDef> = None;
        let mut has_x = false;
        let mut has_y = false;

        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let up = line.to_uppercase();
            if up == "BEGIN HOTSPOT" {
                if cur.is_some() {
                    return Err("Unclosed hotspot block: every BEGIN HOTSPOT needs an END HOTSPOT".into());
                }
                cur = Some(HotspotDef::default());
                has_x = false;
                has_y = false;
                continue;
            }
            if up == "END HOTSPOT" {
                if let Some(h) = cur.take() {
                    if !has_x {
                        errors.push("A hotspot has no X position".into());
                    } else if !has_y {
                        errors.push("A hotspot has no Y position".into());
                    } else if h.target.as_os_str().is_empty() {
                        errors.push("A hotspot has no TARGET".into());
                    } else if !(0.0..=100.0).contains(&h.x) {
                        errors.push("Hotspot X must lie between 0 and 100".into());
                    } else if !(0.0..=100.0).contains(&h.y) {
                        errors.push("Hotspot Y must lie between 0 and 100".into());
                    } else {
                        p.hotspots.push(h);
                    }
                }
                continue;
            }
            if let Some(h) = cur.as_mut() {
                if let Some(v) = strip_key(line, &up, "X=") {
                    if let Some(n) = parse_num(v, "Hotspot X", &mut errors) {
                        h.x = n;
                        has_x = true;
                    }
                } else if let Some(v) = strip_key(line, &up, "Y=") {
                    if let Some(n) = parse_num(v, "Hotspot Y", &mut errors) {
                        h.y = n;
                        has_y = true;
                    }
                } else if let Some(v) = strip_key(line, &up, "DESCRIPTION=") {
                    h.description = v.to_string();
                } else if let Some(v) = strip_key(line, &up, "TARGET=") {
                    h.target = resolve(&base, v);
                } else if let Some(v) = strip_key(line, &up, "INITIALYAW=") {
                    h.initial_yaw = parse_num(v, "Hotspot INITIALYAW", &mut errors);
                } else if let Some(v) = strip_key(line, &up, "INITIALPITCH=") {
                    h.initial_pitch = parse_num(v, "Hotspot INITIALPITCH", &mut errors);
                } else if let Some(v) = strip_key(line, &up, "INITIALHFOV=") {
                    h.initial_hfov = parse_num(v, "Hotspot INITIALHFOV", &mut errors);
                }
                continue;
            }
            if let Some(v) = strip_key(line, &up, "IMAGENAME=") {
                p.image = resolve(&base, v);
            } else if let Some(v) = strip_key(line, &up, "WINDOWTITLE=") {
                p.window_title = Some(v.to_string());
            } else if let Some(v) = strip_key(line, &up, "YAW=") {
                p.yaw = parse_num(v, "YAW", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "PITCH=") {
                p.pitch = parse_num(v, "PITCH", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "HFOV=") {
                p.hfov = parse_num(v, "HFOV", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "HORIZONPOSITION=") {
                p.horizon_position = parse_num(v, "HORIZONPOSITION", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "MINYAW=") {
                p.min_yaw = parse_num(v, "MINYAW", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "MAXYAW=") {
                p.max_yaw = parse_num(v, "MAXYAW", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "MINPITCH=") {
                p.min_pitch = parse_num(v, "MINPITCH", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "MAXPITCH=") {
                p.max_pitch = parse_num(v, "MAXPITCH", &mut errors);
            } else if let Some(v) = strip_key(line, &up, "AUTOSPIN=") {
                p.autospin = parse_num(v, "AUTOSPIN", &mut errors);
            }
        }
        if cur.is_some() {
            return Err("Unclosed hotspot block: every BEGIN HOTSPOT needs an END HOTSPOT".into());
        }
        p.check(&mut errors);
        if errors.is_empty() { Ok(p) } else { Err(errors.join("\n")) }
    }

    fn check(&self, errors: &mut Vec<String>) {
        if self.image.as_os_str().is_empty() {
            errors.push("IMAGENAME is not set".into());
        }
        if let Some(h) = self.horizon_position
            && !(0.0..=100.0).contains(&h) {
                errors.push("HORIZONPOSITION must lie between 0 and 100".into());
            }
        let yaw_ok = |v: Option<f64>| v.is_none_or(|v| (0.0..=360.0).contains(&v));
        let pitch_ok = |v: Option<f64>| v.is_none_or(|v| (-90.0..=90.0).contains(&v));
        if !yaw_ok(self.yaw) {
            errors.push("YAW must lie between 0 and 360".into());
        }
        if !pitch_ok(self.pitch) {
            errors.push("PITCH must lie between -90 and 90".into());
        }
        if !yaw_ok(self.min_yaw) {
            errors.push("MINYAW must lie between 0 and 360".into());
        }
        if !yaw_ok(self.max_yaw) {
            errors.push("MAXYAW must lie between 0 and 360".into());
        }
        if let (Some(a), Some(b)) = (self.min_yaw, self.max_yaw)
            && b <= a {
                errors.push("MAXYAW has to exceed MINYAW".into());
            }
        if !pitch_ok(self.min_pitch) {
            errors.push("MINPITCH must lie between -90 and 90".into());
        }
        if !pitch_ok(self.max_pitch) {
            errors.push("MAXPITCH must lie between -90 and 90".into());
        }
        if let (Some(a), Some(b)) = (self.min_pitch, self.max_pitch)
            && b <= a {
                errors.push("MAXPITCH has to exceed MINPITCH".into());
            }
    }

    /// Parameters for a bare image, filled from its GPano XMP tags.
    pub fn from_image(path: &Path, gpano: &GPano) -> Self {
        let mut p = PanoParams { image: path.to_path_buf(), ..Default::default() };
        p.apply_gpano(gpano);
        p
    }

    /// Yaw is measured from the left edge
    /// of the image, so the image centre (the pose heading) is yaw 180.
    pub fn apply_gpano(&mut self, g: &GPano) {
        if let Some(h) = g.get("InitialViewHeadingDegrees") {
            let pose = g.get("PoseHeadingDegrees").unwrap_or(0.0);
            self.yaw = Some((180.0 - pose + h).rem_euclid(360.0));
        }
        if let Some(v) = g.get("InitialViewPitchDegrees") {
            self.pitch = Some(v.clamp(-90.0, 90.0));
        }
        if let Some(v) = g.get("InitialHorizontalFOVDegrees") {
            self.hfov = Some(v);
        }
        if let (Some(full_h), Some(crop_h), Some(top)) = (
            g.get("FullPanoHeightPixels"),
            g.get("CroppedAreaImageHeightPixels"),
            g.get("CroppedAreaTopPixels"),
        )
            && crop_h > 0.0 {
                let hp = ((full_h * 0.5 - top) / crop_h * 100.0).round();
                if (0.0..=100.0).contains(&hp) {
                    self.horizon_position = Some(hp);
                }
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fsv() {
        let dir = std::env::temp_dir().join("fsv_test_rs");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.fsv");
        std::fs::write(
            &f,
            "// comment\nImageName=sub\\pano.jpg\nyaw=10\nPITCH=-5\nHFOV=60\nHorizonPosition=40\nAUTOSPIN=0.5\n\
             BEGIN HOTSPOT\nX=25\nY=50\nDESCRIPTION=Door\nTARGET=b.fsv\nINITIALYAW=90\nEND HOTSPOT\n",
        )
        .unwrap();
        let p = PanoParams::read_fsv(&f).unwrap();
        assert_eq!(p.image, dir.join("sub/pano.jpg"));
        assert_eq!(p.yaw, Some(10.0));
        assert_eq!(p.pitch, Some(-5.0));
        assert_eq!(p.horizon_position, Some(40.0));
        assert_eq!(p.hotspots.len(), 1);
        assert_eq!(p.hotspots[0].description, "Door");
        assert_eq!(p.hotspots[0].target, dir.join("b.fsv"));
        assert_eq!(p.hotspots[0].initial_yaw, Some(90.0));
    }

    #[test]
    fn rejects_unterminated_hotspot() {
        let f = std::env::temp_dir().join("fsv_test_rs_bad.fsv");
        std::fs::write(&f, "IMAGENAME=x.jpg\nBEGIN HOTSPOT\nX=1\n").unwrap();
        assert!(PanoParams::read_fsv(&f).is_err());
    }
}
