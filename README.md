# RustyPano

A GPU-accelerated viewer for equirectangular panoramas, written in Rust with
**wgpu** for rendering, **winit** for the window and **egui** for the UI (via
`eframe`). Supports `.fsv` panorama tour files with hotspots, GPano XMP
metadata, partial panoramas, a grid overlay and colour management.

## Build & run

```sh
cargo build --release
./target/release/rustypano [options] [pano.jpg | pano.tif | pano.png | tour.fsv]
```

Options: `-y <yaw>`, `-p <pitch>`, `-hf <hfov>`, `-fs`
(full screen), `-hson`, and `-left -top -width -height` (all four together).

Test panoramas and a small hotspot tour are in `../testdata/`:
`./target/release/rustypano ../testdata/tour.fsv`

## Controls

| Action | Input |
|---|---|
| Pan | drag with left mouse button (or arrow keys) |
| Zoom | mouse wheel, `A`/`Shift` in, `Z`/`Ctrl` out, `+`/`-` |
| Previous / next image in folder | `Alt+Left` / `Alt+Right` |
| Full screen | `F11` (`Esc` leaves) |
| Toggle hotspots / grid | `H` / `G` |
| Reset to initial view | `Home` |
| Open | `Ctrl+O`, or drop a file on the window |

Mouse behaviour (Edit → Settings → Mouse): *Grab* (default, the image follows
the pointer) or *Joystick* (speed grows with the distance from the click point,
with adjustable speed).

## Features

- JPEG, TIFF (8/16-bit) and PNG equirectangular panoramas, full or partial
  (vertical extent from `HORIZONPOSITION` or GPano cropped-area tags).
- `.fsv` files: `IMAGENAME`, `WINDOWTITLE`, `YAW`, `PITCH`, `HFOV`,
  `HORIZONPOSITION`, `MIN/MAXYAW`, `MIN/MAXPITCH`, `AUTOSPIN`, and
  `BEGIN HOTSPOT … END HOTSPOT` blocks (`X`, `Y` in percent, `DESCRIPTION`,
  `TARGET`, `INITIALYAW/PITCH/HFOV`).
- Hotspots: drawn markers, hover descriptions, click to jump, Jump to Hotspot list.
- Grid overlay with the horizon line, with configurable spacing, width and colour.
- Tools: Go to View (and make it the start view), Photo Sphere Tags (view
  GPano tags, write the start view or horizon height into the JPEG's XMP),
  Find Horizon, Benchmark.
- Colour management (qcms): embedded ICC → sRGB or a display ICC file.
- Filtering: nearest, trilinear (mipmapped), bicubic, Lanczos3, chosen
  separately for a still view and while moving.
- Recent files, settings in `~/.config/rustypano/settings.toml`.

## How rendering works

`src/loader.rs` decodes on a background thread, applies the ICC transform, then
cuts the panorama into square tiles (≤ 4096 px) with a 32 px gutter of real
neighbouring pixels (wrapping horizontally). Each tile becomes one layer of a
`texture_2d_array`, with 6 gamma-correct mip levels built on the CPU. Because
of the gutter, filtering never shows seams up to mip level 5, so panoramas far
beyond the GPU's maximum texture size work.

`src/pano.wgsl` casts a ray per screen pixel, converts it to longitude/latitude,
picks the tile and mip level from screen-space derivatives, and samples with
the selected kernel. The grid overlay is drawn analytically in the same pass.

After uploading, the loader reads back a few texels to verify the upload.
Some OpenGL drivers fail silently when video memory runs out; in that case the
panorama is downscaled by 2 and uploaded again. There is also a configurable
GPU memory budget (Settings → System).

## Layout

| File | Purpose |
|---|---|
| `src/main.rs` | command line, window and wgpu device setup |
| `src/app.rs` | UI, menus, toolbar, status bar, dialogs, input handling |
| `src/view.rs` | view state, limits, projection maths (shared with picking) |
| `src/loader.rs` | decode, colour management, tiling, mipmaps, upload |
| `src/render.rs`, `src/pano.wgsl` | wgpu pipeline and shader |
| `src/fsv.rs` | `.fsv` parser and GPano → view parameters |
| `src/xmp.rs` | GPano XMP read (JPEG/TIFF/PNG) and write (JPEG) |
| `src/settings.rs` | persisted preferences |

Debug aids: `RUSTYPANO_SCREENSHOT=out.png` saves one rendered frame and quits;
`RUSTYPANO_GRID=1`, `RUSTYPANO_INTERP=nearest|bilinear|bicubic|lanczos` and
`RUSTYPANO_DIALOG=options|metadata|hotspots|view|about` set up that frame.

## License

MIT, see [LICENSE](LICENSE).
