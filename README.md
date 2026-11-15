# RustyPano

A GPU-accelerated viewer for equirectangular panoramas, written in Rust with
**wgpu** for rendering, **winit** for the window and **egui** for the UI (via
`eframe`). Supports `.fsv` panorama tour files with hotspots, GPano XMP
metadata, partial panoramas, a grid overlay and colour management.

## Downloads

Every push to `main` is built by GitHub Actions (`.github/workflows/build.yml`) for:

| Platform | Package |
|---|---|
| Linux x64 / ARM64 | `RustyPano-<version>-linux-<arch>.AppImage` (glibc 2.35+) |
| Windows x64 / ARM64 | `RustyPano-<version>-windows-<arch>.zip` |
| macOS ARM64 (11+) | `RustyPano-<version>-macos-arm64.zip` (contains `RustyPano.app`) |
| Web (WebAssembly) | `RustyPano-<version>-web.zip` (static files for any web server) |

The packages are attached to each workflow run as artifacts, and to a GitHub
release when a `v*` tag is pushed. The macOS app is only ad-hoc signed, not
notarized: after unzipping, run `xattr -dr com.apple.quarantine RustyPano.app`
or right-click → Open the first time.

## Build & run

```sh
cargo build --release
./target/release/rustypano [options] [panorama image | tour.fsv]
```

Options: `-y <yaw>`, `-p <pitch>`, `-hf <hfov>`, `-fs`
(full screen), `-hson`, and `-left -top -width -height` (all four together).

Test panoramas and a small hotspot tour are in `../testdata/`:
`./target/release/rustypano ../testdata/tour.fsv`

## Web version

`packaging/web/build-web.sh` builds the browser version into `dist/web`. It needs
the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` at the same version as
the `wasm-bindgen` crate in `Cargo.lock`, and uses `wasm-opt` when installed:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
packaging/web/build-web.sh
python3 -m http.server -d dist/web 8080   # then open http://localhost:8080
```

Every push to `main` also deploys the web version to the repository's GitHub
Pages site (enable it once under Settings → Pages → Source: *GitHub Actions*).

The page shows the app icon while the WebAssembly module loads, then the viewer
fills the whole browser window. Rendering uses WebGPU where the browser offers it
and falls back to WebGL2. Fullscreen uses the browser's Fullscreen API (`F11` key
in the app, the Fullscreen button, `Esc` to leave).

Opening panoramas in the browser:

- **Open… / drag and drop:** pick or drop a `.fsv` tour *together with* its
  images; the tour finds them by file name.
- **URL:** `index.html?pano=<url>` loads a panorama or tour from the web (the
  server must allow it; same origin always works). Tour references are
  resolved relative to the tour's URL. `yaw`, `pitch`, `hfov` and `fs=1` set the
  initial view and fullscreen.

Settings are kept in the browser's local storage. Not available in the
browser: previous/next image in folder, recent files, saving Photo Sphere tags
into files, display ICC profiles from files, and panoramas above 300 megapixels.

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

- Equirectangular panoramas in JPEG, TIFF (8/16-bit), PNG, WebP, BMP and
  JPEG XL, full or partial (vertical extent from `HORIZONPOSITION` or GPano
  cropped-area tags). Embedded ICC profiles are honoured; GPano XMP is read
  from all formats except BMP and written into JPEG only.
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
- English and German UI. The language follows the system (`LANGUAGE`,
  `LC_ALL`, `LC_MESSAGES`, `LANG` on Linux) and can be changed under
  Edit → Settings → View → Language.

## How rendering works

Panoramas are streamed in multiple resolutions, so GPU memory depends on the
screen, not on the image size:

- `src/loader.rs` decodes (on a background thread natively), applies the ICC
  transform and builds a **resolution pyramid** on the CPU: full size, half,
  quarter, … (gamma-correct 2× downscaling).
- `src/tiles.rs` cuts every level into tiles. The tile width and height are
  chosen per image (multiples of 32 between 512 and 1024 px) so the finest
  levels divide almost evenly, e.g. 672×1024 for 20000×10000.
- On the GPU there is only a **tile cache**: a `texture_2d_array` whose layers
  ("slots") hold one tile each, with a 32 px gutter of real neighbouring pixels
  and 6 gamma-correct mip levels, plus a small **page table** texture mapping
  (level, tile) → slot. The coarsest level is always resident.
- Every frame the visible tiles at the resolution the screen needs are worked
  out on the CPU; missing ones are built by worker threads (in the browser:
  within ~8 ms per frame) and the least recently used tiles are evicted.
- `src/pano.wgsl` casts a ray per screen pixel, converts it to longitude/
  latitude, picks the pyramid level from screen-space derivatives, looks the
  tile up in the page table (falling back to coarser levels while finer tiles
  stream in) and samples with the selected kernel. The grid overlay is drawn
  analytically in the same pass.

The cache size is set under Settings → System → GPU tile cache (default 768 MB,
384 MB in the browser; a 4K screen needs well under 200 MB). Some OpenGL drivers
fail silently when video memory runs out, so the cache is verified after
allocation and made smaller if needed. The full image stays in main memory
(about 4 bytes per pixel including the pyramid).

## Translations

User-visible text is wrapped in `tr!("English text")`, or
`tr!("Loaded {n} files", n = count)` when it has named placeholders. The English text
is the lookup key. Catalogs live in `i18n/<lang>.toml` and map the English text to
the translation:

```toml
"Open a panorama (Ctrl+O)" = "Panorama öffnen (Strg+O)"
```

`cargo test` fails if a `tr!` string has no German entry, if the catalog has entries
the code no longer uses, or if a translation changes the `{placeholders}`. To add a
language, add a variant to `Lang` and `LanguageChoice` in `src/i18n.rs` and a new
catalog file.

## Layout

| File | Purpose |
|---|---|
| `src/main.rs` | command line, window and wgpu device setup |
| `src/app.rs` | UI, menus, toolbar, status bar, dialogs, input handling |
| `src/view.rs` | view state, limits, projection maths (shared with picking) |
| `src/loader.rs` | read, decode, colour management, pyramid, cache setup |
| `src/tiles.rs` | tile grid, pyramid, GPU tile cache, streaming, visibility |
| `src/render.rs`, `src/pano.wgsl` | wgpu pipeline and shader |
| `src/fsv.rs` | `.fsv` parser and GPano → view parameters |
| `src/xmp.rs` | GPano XMP read (JPEG/TIFF/PNG) and write (JPEG) |
| `src/settings.rs` | persisted preferences |
| `src/i18n.rs`, `i18n/de.toml` | translations and language detection |
| `src/source.rs` | where files come from: local paths, in-memory files, URLs |
| `src/web.rs` | browser glue (fetch, local storage, fullscreen, startup screen) |
| `build.rs`, `packaging/`, `.github/workflows/` | Windows icon, AppImage/macOS/web packaging, CI |

Debug aids: `RUSTYPANO_SCREENSHOT=out.png` saves one rendered frame and quits;
`RUSTYPANO_GRID=1`, `RUSTYPANO_INTERP=nearest|bilinear|bicubic|lanczos` and
`RUSTYPANO_DIALOG=options|metadata|hotspots|view|about` set up that frame.

## License

MIT, see [LICENSE](LICENSE).
