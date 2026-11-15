// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! View state, limits and the rectilinear projection used by both the GPU
//! shader and the CPU side (hotspots, picking).
//!
//! Conventions:
//! * yaw is measured in degrees from the left edge of the equirectangular
//!   image, increasing to the right, 0..360;
//! * pitch is positive above the horizon, -90..90;
//! * the horizon sits at `horizon_position` percent of the image height.

pub const MAX_HFOV: f64 = 140.0;
pub const MAX_ZOOM_PERCENT: f64 = 1600.0;

#[derive(Clone, Copy, Debug)]
pub struct PanoGeometry {
    pub width: f64,
    pub height: f64,
    /// Horizon row in image pixels from the top.
    pub horizon_row: f64,
}

impl PanoGeometry {
    pub fn new(width: u32, height: u32, horizon_percent: Option<f64>) -> Self {
        let (w, h) = (width as f64, height as f64);
        let hp = horizon_percent.unwrap_or(50.0).clamp(0.0, 100.0);
        Self { width: w, height: h, horizon_row: hp / 100.0 * h }
    }
    pub fn deg_per_px(&self) -> f64 {
        360.0 / self.width
    }
    pub fn max_pitch(&self) -> f64 {
        (self.horizon_row * self.deg_per_px()).min(90.0)
    }
    pub fn min_pitch(&self) -> f64 {
        (-(self.height - self.horizon_row) * self.deg_per_px()).max(-90.0)
    }
    pub fn horizon_percent(&self) -> f64 {
        self.horizon_row / self.height * 100.0
    }
    /// Image pixel → (yaw, pitch).
    pub fn pixel_to_angles(&self, x: f64, y: f64) -> (f64, f64) {
        (x * self.deg_per_px(), (self.horizon_row - y) * self.deg_per_px())
    }
    /// (yaw, pitch) → image pixel (may be outside the image vertically).
    pub fn angles_to_pixel(&self, yaw: f64, pitch: f64) -> (f64, f64) {
        (yaw.rem_euclid(360.0) / self.deg_per_px(), self.horizon_row - pitch / self.deg_per_px())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UserLimits {
    pub min_yaw: Option<f64>,
    pub max_yaw: Option<f64>,
    pub min_pitch: Option<f64>,
    pub max_pitch: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct View {
    pub yaw: f64,
    pub pitch: f64,
    pub hfov: f64,
    /// Viewport size in physical pixels.
    pub vw: f64,
    pub vh: f64,
    pub geo: PanoGeometry,
    pub limits: UserLimits,
}

fn d2r(d: f64) -> f64 {
    d.to_radians()
}

pub fn vfov_from_hfov(hfov: f64, w: f64, h: f64) -> f64 {
    let f = (w * 0.5) / d2r(hfov * 0.5).tan();
    2.0 * (h * 0.5 / f).atan().to_degrees()
}

pub fn hfov_from_vfov(vfov: f64, w: f64, h: f64) -> f64 {
    let f = (h * 0.5) / d2r(vfov * 0.5).tan();
    2.0 * (w * 0.5 / f).atan().to_degrees()
}

impl View {
    pub fn new(geo: PanoGeometry) -> Self {
        Self { yaw: 180.0, pitch: 0.0, hfov: 70.0, vw: 800.0, vh: 600.0, geo, limits: UserLimits::default() }
    }

    pub fn vfov(&self) -> f64 {
        vfov_from_hfov(self.hfov, self.vw, self.vh)
    }

    /// Focal length in pixels.
    pub fn focal(&self) -> f64 {
        (self.vw * 0.5) / d2r(self.hfov * 0.5).tan()
    }

    /// Screen pixels per image pixel at the centre of the view, in percent.
    pub fn zoom_percent(&self) -> f64 {
        self.focal() * std::f64::consts::TAU / self.geo.width * 100.0
    }

    pub fn min_hfov(&self) -> f64 {
        // focal = Z/100 * W / 2π   →  hfov = 2 atan(vw/2 / focal)
        let focal = MAX_ZOOM_PERCENT / 100.0 * self.geo.width / std::f64::consts::TAU;
        (2.0 * (self.vw * 0.5 / focal).atan().to_degrees()).max(0.2)
    }

    fn pitch_bounds(&self) -> (f64, f64) {
        let lo = match self.limits.min_pitch {
            Some(m) => m.max(self.geo.min_pitch()),
            None => self.geo.min_pitch(),
        };
        let hi = match self.limits.max_pitch {
            Some(m) => m.min(self.geo.max_pitch()),
            None => self.geo.max_pitch(),
        };
        (lo, hi)
    }

    fn yaw_bounds(&self) -> (f64, f64) {
        (self.limits.min_yaw.unwrap_or(0.0), self.limits.max_yaw.unwrap_or(360.0))
    }

    /// Set the horizontal FOV, respecting the maximum and the vertical extent
    /// of the image. Returns true if something changed.
    pub fn set_hfov(&mut self, hfov: f64) -> bool {
        let old = self.hfov;
        let mut h = hfov.clamp(self.min_hfov(), MAX_HFOV);
        let (lo, hi) = self.pitch_bounds();
        let avail_v = hi - lo;
        if vfov_from_hfov(h, self.vw, self.vh) > avail_v {
            h = hfov_from_vfov(avail_v, self.vw, self.vh).min(h);
        }
        let (ylo, yhi) = self.yaw_bounds();
        if ylo > 0.0 || yhi < 360.0 {
            h = h.min(yhi - ylo);
        }
        self.hfov = h;
        let changed_p = self.set_pitch(self.pitch);
        let changed_y = self.set_yaw(self.yaw);
        (self.hfov - old).abs() > 1e-9 || changed_p || changed_y
    }

    pub fn set_pitch(&mut self, pitch: f64) -> bool {
        let old = self.pitch;
        let (lo, hi) = self.pitch_bounds();
        let half = self.vfov() * 0.5;
        let lo_c = if lo <= -90.0 + 1e-9 { -90.0 } else { lo + half };
        let hi_c = if hi >= 90.0 - 1e-9 { 90.0 } else { hi - half };
        self.pitch = if lo_c > hi_c { (lo + hi) * 0.5 } else { pitch.clamp(lo_c, hi_c) };
        (self.pitch - old).abs() > 1e-9
    }

    pub fn set_yaw(&mut self, yaw: f64) -> bool {
        let old = self.yaw;
        let (lo, hi) = self.yaw_bounds();
        let half = self.hfov * 0.5;
        let mut y = yaw;
        if lo > 0.0 || hi < 360.0 {
            y = y.clamp(lo + half, (hi - half).max(lo + half));
        } else {
            y = y.rem_euclid(360.0);
        }
        self.yaw = y;
        (self.yaw - old).abs() > 1e-9
    }

    pub fn add_yaw(&mut self, d: f64) -> bool {
        self.set_yaw(self.yaw + d)
    }
    pub fn add_pitch(&mut self, d: f64) -> bool {
        self.set_pitch(self.pitch + d)
    }
    pub fn zoom(&mut self, factor: f64) -> bool {
        self.set_hfov(self.hfov * factor)
    }

    pub fn set_viewport(&mut self, w: f64, h: f64) {
        if (w - self.vw).abs() > 0.5 || (h - self.vh).abs() > 0.5 {
            self.vw = w.max(1.0);
            self.vh = h.max(1.0);
            self.set_hfov(self.hfov);
        }
    }

    /// Orthonormal camera basis (right, up, forward).
    fn basis(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (st, ct) = d2r(self.yaw).sin_cos();
        let (sp, cp) = d2r(self.pitch).sin_cos();
        let f = [st * cp, sp, ct * cp];
        let r = [ct, 0.0, -st];
        let u = [-st * sp, cp, -ct * sp];
        (r, u, f)
    }

    /// Like [`Self::screen_to_angles`], with the camera set up once for
    /// mapping many points.
    pub fn screen_mapper(&self) -> impl Fn(f64, f64) -> (f64, f64) {
        let (r, u, f) = self.basis();
        let (fl, cx, cy) = (self.focal(), self.vw * 0.5, self.vh * 0.5);
        move |sx, sy| {
            let (x, y) = (sx - cx, -(sy - cy));
            let d = [
                r[0] * x + u[0] * y + f[0] * fl,
                r[1] * x + u[1] * y + f[1] * fl,
                r[2] * x + u[2] * y + f[2] * fl,
            ];
            (d[0].atan2(d[2]).to_degrees().rem_euclid(360.0), d[1].atan2(d[0].hypot(d[2])).to_degrees())
        }
    }

    /// Screen pixel (relative to the viewport's top-left) → (yaw, pitch).
    pub fn screen_to_angles(&self, sx: f64, sy: f64) -> (f64, f64) {
        let (r, u, f) = self.basis();
        let x = sx - self.vw * 0.5;
        let y = -(sy - self.vh * 0.5);
        let fl = self.focal();
        let d = [
            r[0] * x + u[0] * y + f[0] * fl,
            r[1] * x + u[1] * y + f[1] * fl,
            r[2] * x + u[2] * y + f[2] * fl,
        ];
        let lon = d[0].atan2(d[2]).to_degrees().rem_euclid(360.0);
        let lat = d[1].atan2(d[0].hypot(d[2])).to_degrees();
        (lon, lat)
    }

    /// (yaw, pitch) → screen pixel, or None if behind the camera.
    pub fn angles_to_screen(&self, yaw: f64, pitch: f64) -> Option<(f64, f64)> {
        let (r, u, f) = self.basis();
        let (st, ct) = d2r(yaw).sin_cos();
        let (sp, cp) = d2r(pitch).sin_cos();
        let d = [st * cp, sp, ct * cp];
        let dot = |a: [f64; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
        let z = dot(f);
        if z <= 1e-6 {
            return None;
        }
        let fl = self.focal();
        Some((self.vw * 0.5 + fl * dot(r) / z, self.vh * 0.5 - fl * dot(u) / z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_projection() {
        let mut v = View::new(PanoGeometry::new(8000, 4000, None));
        v.set_viewport(1200.0, 800.0);
        v.yaw = 33.0;
        v.pitch = 12.0;
        v.hfov = 80.0;
        for &(sx, sy) in &[(600.0, 400.0), (10.0, 20.0), (1190.0, 790.0)] {
            let (a, b) = v.screen_to_angles(sx, sy);
            let (x, y) = v.angles_to_screen(a, b).unwrap();
            assert!((x - sx).abs() < 1e-6 && (y - sy).abs() < 1e-6);
        }
        let (a, b) = v.screen_to_angles(600.0, 400.0);
        assert!((a - 33.0).abs() < 1e-9 && (b - 12.0).abs() < 1e-9);
    }

    #[test]
    fn partial_pano_limits() {
        // 360x60 degree strip, horizon centred.
        let mut v = View::new(PanoGeometry::new(6000, 1000, Some(50.0)));
        v.set_viewport(1000.0, 1000.0);
        v.set_hfov(120.0);
        assert!(v.vfov() <= 60.0 + 1e-6);
        v.set_pitch(80.0);
        assert!(v.pitch + v.vfov() / 2.0 <= 30.0 + 1e-6);
    }
}
