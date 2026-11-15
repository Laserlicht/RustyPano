// SPDX-License-Identifier: MIT
// Copyright (c) 2026 RustyPano contributors

// Equirectangular → rectilinear reprojection, evaluated per screen pixel.
//
// The panorama is a pyramid of resolution levels cut into tiles. Resident
// tiles live in the layers ("slots") of `tex`; the page table maps
// (level, tile) to slot + 1 (0 = not resident). Each pixel uses the level its
// footprint needs and falls back to coarser levels until a resident tile is
// found; the coarsest level is always resident.

struct U {
    vp: vec4f,        // viewport origin (px), viewport size (px)
    img: vec4f,       // full-resolution W, H, horizon row, focal length (px)
    cam: vec4f,       // yaw, pitch (radians), gutter, max in-slot lod
    tile: vec4f,      // tile content cx, cy; slot size sx, sy
    flags: vec4u,     // interpolator, encode sRGB on output, grid flags, base level
    grid: vec4f,      // rows distance (deg), columns distance (deg), line width (px), _
    grid_color: vec4f,
    bg: vec4f,
    // Per level: (nx | ny << 16, first page-table row, width, height).
    levels: array<vec4u, 16>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp_lin: sampler;
@group(0) @binding(3) var page_table: texture_2d<u32>;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn to_srgb(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

struct TileCoord {
    local: vec2f, // pixel position inside the slot at its level (includes gutter)
    layer: i32,   // slot, or -1 if the tile is not resident
};

// Tile of pyramid level `k` under full-resolution pixel `g`.
fn lookup(k: u32, g: vec2f) -> TileCoord {
    let l = u.levels[k];
    let n = vec2f(f32(l.x & 0xffffu), f32(l.x >> 16u));
    let gk = g * vec2f(f32(l.z) / u.img.x, f32(l.w) / u.img.y);
    let c = u.tile.xy;
    let t = clamp(floor(gk / c), vec2f(0.0), n - 1.0);
    let v = textureLoad(page_table, vec2i(i32(t.x), i32(l.y) + i32(t.y)), 0).r;
    var r: TileCoord;
    r.local = gk - t * c + vec2f(u.cam.z);
    r.layer = i32(v) - 1;
    return r;
}

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-4) { return 1.0; }
    let px = PI * x;
    return sin(px) / px;
}

fn lanczos3(x: f32) -> f32 {
    if (abs(x) >= 3.0) { return 0.0; }
    return sinc(x) * sinc(x / 3.0);
}

fn catmull_rom(x0: f32) -> f32 {
    let x = abs(x0);
    if (x < 1.0) { return 1.5 * x * x * x - 2.5 * x * x + 1.0; }
    if (x < 2.0) { return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0; }
    return 0.0;
}

// Separable kernel evaluated with textureLoad at one integer mip level.
fn sample_kernel(tc: TileCoord, level: i32, lanczos: bool) -> vec3f {
    let scale = exp2(f32(level));
    let s = tc.local / scale - 0.5;
    let i0 = floor(s);
    let f = s - i0;
    let lo = select(-1, -2, lanczos);
    let hi = select(2, 3, lanczos);
    var wx: array<f32, 6>;
    var wy: array<f32, 6>;
    var sx = 0.0;
    var sy = 0.0;
    for (var k = lo; k <= hi; k++) {
        let a = f32(k) - f.x;
        let b = f32(k) - f.y;
        let wa = select(catmull_rom(a), lanczos3(a), lanczos);
        let wb = select(catmull_rom(b), lanczos3(b), lanczos);
        wx[k - lo] = wa;
        wy[k - lo] = wb;
        sx += wa;
        sy += wb;
    }
    var acc = vec3f(0.0);
    let base = vec2i(i0);
    for (var j = lo; j <= hi; j++) {
        var row = vec3f(0.0);
        for (var i = lo; i <= hi; i++) {
            row += wx[i - lo] * textureLoad(tex, base + vec2i(i, j), tc.layer, level).rgb;
        }
        acc += wy[j - lo] * row;
    }
    // Negative lobes can overshoot; clamp to the displayable range.
    return clamp(acc / (sx * sy), vec3f(0.0), vec3f(1.0));
}

fn grid_alpha(dist_deg: f32, deg_per_px: f32, width: f32) -> f32 {
    let d_px = dist_deg / max(deg_per_px, 1e-6);
    return clamp(width * 0.5 + 0.5 - d_px, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let p = pos.xy - u.vp.xy;
    let x = p.x - u.vp.z * 0.5;
    let y = -(p.y - u.vp.w * 0.5);
    let yaw = u.cam.x;
    let pitch = u.cam.y;
    let st = sin(yaw); let ct = cos(yaw);
    let sp = sin(pitch); let cp = cos(pitch);
    let fwd = vec3f(st * cp, sp, ct * cp);
    let right = vec3f(ct, 0.0, -st);
    let up = vec3f(-st * sp, cp, -ct * sp);
    let d = right * x + up * y + fwd * u.img.w;

    let lon = atan2(d.x, d.z);                  // -π..π
    let lat = atan2(d.y, length(d.xz));        // -π/2..π/2
    let w = u.img.x;
    let gx = fract(lon / TAU + 1.0) * w;
    let gy = u.img.z - lat * w / TAU;

    // Screen-space footprint (computed in uniform control flow).
    var dlx = dpdx(lon); var dly = dpdy(lon);
    dlx = select(dlx, dlx - sign(dlx) * TAU, abs(dlx) > PI);
    dly = select(dly, dly - sign(dly) * TAU, abs(dly) > PI);
    let dlatx = dpdx(lat); let dlaty = dpdy(lat);
    let coslat = cos(lat);
    let px_per_rad = w / TAU;
    let fx = length(vec2f(dlx * coslat, dlatx)) * px_per_rad;
    let fy = length(vec2f(dly * coslat, dlaty)) * px_per_rad;
    let lod = log2(max(max(fx, fy), 1e-6));
    let deg_lat = degrees(max(abs(dlatx), abs(dlaty)));
    let deg_lon = degrees(max(abs(dlx), abs(dly)));

    if (gy < 0.0 || gy >= u.img.y) {
        return u.bg;
    }

    // Finest resident level at or above the one the footprint asks for.
    let base = u.flags.w;
    let g = vec2f(gx, gy);
    var k = u32(clamp(floor(lod), 0.0, f32(base)));
    var tc = lookup(k, g);
    for (var i = 0u; i < 16u; i++) {
        if (tc.layer >= 0 || k >= base) {
            break;
        }
        k += 1u;
        tc = lookup(k, g);
    }
    if (tc.layer < 0) {
        return u.bg;
    }

    // Level of detail relative to the tile's own resolution: inside [0, 1)
    // when the wanted level is resident, below 0 while a finer one streams in.
    let rel = lod - f32(k);
    let max_lod = u.cam.w;
    let rel_c = clamp(rel, 0.0, max_lod);
    let uv = tc.local / u.tile.zw;
    var col: vec3f;
    switch u.flags.x {
        case 0u: {
            let lvl = i32(round(rel_c));
            let ip = vec2i(floor(tc.local / exp2(f32(lvl))));
            col = textureLoad(tex, ip, tc.layer, lvl).rgb;
        }
        case 2u, 3u: {
            let lanczos = u.flags.x == 3u;
            // Kernels reach ±3 texels: the gutter allows them up to slot mip 3.
            if (rel > 3.0) {
                col = textureSampleLevel(tex, samp_lin, uv, tc.layer, rel_c).rgb;
            } else if (rel <= 0.0) {
                col = sample_kernel(tc, 0, lanczos);
            } else {
                let l0 = i32(floor(rel));
                let fr = rel - f32(l0);
                let a = sample_kernel(tc, l0, lanczos);
                if (fr < 0.02) {
                    col = a;
                } else {
                    col = mix(a, sample_kernel(tc, l0 + 1, lanczos), fr);
                }
            }
        }
        default: {
            col = textureSampleLevel(tex, samp_lin, uv, tc.layer, rel_c).rgb;
        }
    }

    // Grid overlay: rows every N degrees of pitch from the horizon,
    // columns every N degrees of yaw from the left edge of the image.
    let gflags = u.flags.z;
    if (gflags != 0u) {
        var a = 0.0;
        let lw = u.grid.z;
        if ((gflags & 1u) != 0u) {
            let latd = degrees(lat);
            let rd = u.grid.x;
            let dist = abs(latd - round(latd / rd) * rd);
            a = max(a, grid_alpha(dist, deg_lat, lw));
            a = max(a, grid_alpha(abs(latd), deg_lat, lw * 2.0)); // horizon
        }
        if ((gflags & 2u) != 0u) {
            let lond = gx / w * 360.0;
            let cd = u.grid.y;
            let dist = abs(lond - round(lond / cd) * cd);
            a = max(a, grid_alpha(dist, deg_lon, lw));
        }
        col = mix(col, u.grid_color.rgb, a * u.grid_color.a);
    }

    if (u.flags.y != 0u) {
        col = to_srgb(col);
    }
    return vec4f(col, 1.0);
}
