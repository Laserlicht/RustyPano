// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

//! Momentum for grab panning: after a quick drag the view keeps gliding in
//! the same direction and slows down smoothly, like in map applications.
//!
//! Everything is in screen pixels and seconds; the caller converts the
//! displacement to angles with the current zoom.

use std::collections::VecDeque;

/// Movement of the last part of a drag that sets the release speed.
const WINDOW: f64 = 0.08;
/// Time constant of the exponential slow-down: a glide covers about the
/// release speed times this.
const TAU: f64 = 0.33;
/// Slower releases do not glide (a careful drag should just stop).
const MIN_START_SPEED: f64 = 150.0;
/// A glide ends when it gets this slow.
const MIN_SPEED: f64 = 20.0;
const MAX_SPEED: f64 = 8000.0;

#[derive(Default)]
pub struct Inertia {
    /// (time, position summed over the drag) of recent drag events.
    samples: VecDeque<(f64, [f64; 2])>,
    pos: [f64; 2],
    /// Current glide speed in pixels per second, and time of the last step.
    glide: Option<([f64; 2], f64)>,
}

impl Inertia {
    /// A drag starts at time `t`.
    pub fn start(&mut self, t: f64) {
        self.glide = None;
        self.pos = [0.0; 2];
        self.samples.clear();
        self.samples.push_back((t, self.pos));
    }

    /// The pointer moved by `d` pixels during the drag.
    pub fn drag(&mut self, t: f64, d: [f64; 2]) {
        self.pos = [self.pos[0] + d[0], self.pos[1] + d[1]];
        self.samples.push_back((t, self.pos));
        // Keep one sample older than the window as its starting point.
        while self.samples.len() > 2 && self.samples[1].0 <= t - WINDOW {
            self.samples.pop_front();
        }
    }

    /// The drag ended at time `t`: glide on if it was moving fast enough.
    pub fn release(&mut self, t: f64) {
        // Speed over the last part of the drag. A pointer that rested before
        // being released has no recent samples, so its speed is zero.
        let base = self.samples.iter().rev().find(|s| s.0 <= t - WINDOW).or(self.samples.front()).copied();
        self.samples.clear();
        let Some((t0, p0)) = base else { return };
        let dt = (t - t0).max(1.0 / 120.0);
        let mut v = [(self.pos[0] - p0[0]) / dt, (self.pos[1] - p0[1]) / dt];
        let speed = v[0].hypot(v[1]);
        if speed < MIN_START_SPEED {
            return;
        }
        if speed > MAX_SPEED {
            v = [v[0] * MAX_SPEED / speed, v[1] * MAX_SPEED / speed];
        }
        self.glide = Some((v, t));
    }

    /// Displacement since the last step, or `None` when not gliding. The
    /// slow-down is integrated exactly, so the path does not depend on the
    /// frame rate.
    pub fn step(&mut self, t: f64) -> Option<[f64; 2]> {
        let (v, last) = self.glide?;
        let dt = (t - last).clamp(0.0, 0.25);
        let decay = (-dt / TAU).exp();
        let k = TAU * (1.0 - decay);
        let d = [v[0] * k, v[1] * k];
        let v = [v[0] * decay, v[1] * decay];
        self.glide = (v[0].hypot(v[1]) >= MIN_SPEED).then_some((v, t));
        Some(d)
    }

    #[cfg(test)]
    pub fn is_gliding(&self) -> bool {
        self.glide.is_some()
    }

    pub fn stop(&mut self) {
        self.glide = None;
    }

    /// Stop the horizontal (`axis` 0) or vertical (1) part, e.g. at a limit
    /// of the view.
    pub fn stop_axis(&mut self, axis: usize) {
        if let Some((v, _)) = &mut self.glide {
            v[axis] = 0.0;
            if v[0].hypot(v[1]) < MIN_SPEED {
                self.glide = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drag at `speed` px/s for `secs` at `fps`, then release.
    fn flick(speed: f64, secs: f64, fps: f64, rest: f64) -> Inertia {
        let mut m = Inertia::default();
        m.start(0.0);
        let n = (secs * fps) as usize;
        for i in 1..=n {
            m.drag(i as f64 / fps, [speed / fps, 0.0]);
        }
        m.release(n as f64 / fps + rest);
        m
    }

    fn glide_distance(m: &mut Inertia, from: f64, fps: f64) -> (f64, f64) {
        let (mut x, mut t) = (0.0, from);
        while let Some(d) = m.step(t + 1.0 / fps) {
            t += 1.0 / fps;
            x += d[0];
            assert!(t - from < 10.0, "glide never ends");
        }
        (x, t - from)
    }

    #[test]
    fn quick_drag_glides_and_slows_down() {
        let mut m = flick(1000.0, 0.3, 60.0, 0.0);
        assert!(m.is_gliding());
        let (x, secs) = glide_distance(&mut m, 0.3, 60.0);
        // Total ≈ v·τ (minus the tail below the minimum speed).
        assert!((x - 1000.0 * TAU).abs() < 15.0, "glided {x}");
        assert!(secs > 0.5 && secs < 3.0, "lasted {secs}");
    }

    #[test]
    fn path_does_not_depend_on_frame_rate() {
        let a = glide_distance(&mut flick(2000.0, 0.2, 30.0, 0.0), 0.2, 30.0).0;
        let b = glide_distance(&mut flick(2000.0, 0.2, 144.0, 0.0), 0.2, 144.0).0;
        assert!((a - b).abs() < 10.0, "{a} vs {b}");
    }

    #[test]
    fn no_glide_after_resting_or_slow_drag() {
        assert!(!flick(1000.0, 0.3, 60.0, 0.2).is_gliding(), "pointer rested before release");
        assert!(!flick(80.0, 0.5, 60.0, 0.0).is_gliding(), "slow drag");
        let mut m = flick(1000.0, 0.3, 60.0, 0.0);
        m.start(1.0);
        assert!(!m.is_gliding(), "a new drag stops the glide");
    }

    #[test]
    fn short_flick_and_limits() {
        // A 50 ms flick still glides, the speed is capped.
        let mut m = flick(50_000.0, 0.05, 120.0, 0.0);
        let d = m.step(0.05 + 1.0 / 60.0).unwrap();
        assert!(d[0] > 0.0 && d[0] <= MAX_SPEED / 60.0 + 1e-6);
        m.stop_axis(0);
        assert!(!m.is_gliding());
    }
}
