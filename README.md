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

Settings are kept in the browser's local storage and tiles in its Cache
Storage (per tab, deleted when the tab is closed; anything a closed tab could
not delete in time is removed the next time the page is opened). Not available in the
browser: previous/next image in folder, recent files, saving Photo Sphere tags
into files and display ICC profiles from files. Size limits in the browser:
300 megapixels for JPEG and WebP (decoded as a whole), 2 gigapixels for PNG,
TIFF, JPEG XL and BMP (read in stripes).

## Controls

| Action | Input |
|---|---|
| Pan | drag with left mouse button (or arrow keys) |
| Zoom | mouse wheel, `A`/`Shift` in, `Z`/`Ctrl` out, `+`/`-` |
| Previous / next image in folder | `Alt+Left` / `Alt+Right` |
| Full screen | `F11` (`Esc` leaves) |
| Toggle hotspots / grid | `H` / `G` |
| Reset to initial view | `Home` |
| Open | `Ctrl+O`, or drag files onto the window (an overlay shows what will open) |

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
- Tile cache: the tiles of opened panoramas are stored instead of keeping the
  image in memory (on disk under `~/.cache/rustypano/tiles`, in the browser in
  Cache Storage); opening a panorama again in the same session is instant. The
  cache belongs to the running app and is deleted when it closes; leftovers of
  crashed sessions are removed on the next start. Size limit and "Clear" under
  Settings → System (default 10 GB, 4 GB in the browser).
- Recent files, settings in `~/.config/rustypano/settings.toml`.
- English and German UI. The language follows the system (`LANGUAGE`,
  `LC_ALL`, `LC_MESSAGES`, `LANG` on Linux) and can be changed under
  Edit → Settings → View → Language.

## How rendering works

Panoramas are shown in multiple resolutions from a tile cache, so neither the
GPU nor main memory has to hold the whole image:

- **Reading** (`src/pyramid.rs`): PNG, TIFF (strips or tiles), JPEG XL and BMP
  are decoded stripe by stripe. JPEG and WebP decoders can only produce the
  whole image; it is handed on row by row and freed right afterwards.
- **Building tiles:** each stripe is colour-managed (ICC → sRGB or display
  profile) and fed into a streaming pyramid builder. For every level (full
  size, half, quarter, …; gamma-correct 2× downscaling) it keeps only a band of
  about one tile row and emits tiles as soon as their rows are complete. Tile
  width and height are chosen per image (multiples of 32 between 512 and
  1024 px) so the finest levels divide almost evenly; every tile carries a
  32 px gutter of real neighbouring pixels (wrapping horizontally).
- **Tile cache** (`src/store.rs`): tiles are stored as JPEG (quality 92, 4:4:4)
  on disk or in the browser's Cache Storage, keyed by the file's identity and
  the colour settings, in a directory (or set of caches) belonging to the
  running app. Opening the same panorama again in the session does not decode
  the image at all; the least recently used panoramas are removed when the size
  limit is reached, and everything is deleted when the app closes. Each running
  app signals once a minute that it is alive, so a new start only removes
  caches of sessions that are gone. `RUSTYPANO_CACHE_DIR` overrides the
  location.
- **GPU cache** (`src/tiles.rs`): a `texture_2d_array` whose layers ("slots")
  hold one tile each with 6 gamma-correct mip levels, plus a page table
  texture mapping (level, tile) → slot. The coarsest level is always resident.
  Every frame the visible tiles at the needed resolution are worked out on the
  CPU, missing ones are loaded from the tile cache (by worker threads; in the
  browser asynchronously) and the least recently used ones are evicted.
  While the view pans or zooms, its speed is measured and the tiles for where
  it will be in 0.25, 0.5 and 0.8 s are loaded too (after the visible ones, at
  most ¾ of a view ahead), so moving does not briefly show a coarser level.
- **Shader** (`src/pano.wgsl`): casts a ray per screen pixel, converts it to
  longitude/latitude, picks the pyramid level from screen-space derivatives,
  looks the tile up in the page table (falling back to coarser levels while
  finer tiles stream in) and samples with the selected kernel. The grid
  overlay is drawn analytically in the same pass.

The GPU cache size is set under Settings → System (default 768 MB, 384 MB in the
browser). Some OpenGL drivers fail silently when video memory runs out, so the
cache is verified after allocation and made smaller if needed.

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
| `src/loader.rs` | load pipeline: cache lookup, building tiles, GPU cache setup |
| `src/pyramid.rs` | stripe-wise image reading, streaming pyramid builder |
| `src/store.rs` | persistent tile cache (disk, browser Cache Storage, memory) |
| `src/tiles.rs` | tile grid, GPU tile cache, streaming, visibility |
| `src/render.rs`, `src/pano.wgsl` | wgpu pipeline and shader |
| `src/fsv.rs` | `.fsv` parser and GPano → view parameters |
| `src/xmp.rs` | GPano XMP read (JPEG/TIFF/PNG) and write (JPEG) |
| `src/settings.rs` | persisted preferences |
| `src/i18n.rs`, `i18n/de.toml` | translations and language detection |
| `src/source.rs` | where files come from: local paths, in-memory files, URLs |
| `src/web.rs` | browser glue (fetch, local storage, fullscreen, startup screen) |
| `build.rs`, `packaging/`, `.github/workflows/` | Windows icon, AppImage/macOS/web packaging, CI |

Debug aids: `RUSTYPANO_SCREENSHOT=out.png` saves one rendered frame and quits
(`RUSTYPANO_SCREENSHOT_FRAMES=n` after n frames even while moving);
`RUSTYPANO_NO_PREFETCH=1` turns look-ahead loading off; with `RUST_LOG=info`
the app logs how many frames had to show a coarser level;
`RUSTYPANO_GRID=1`, `RUSTYPANO_INTERP=nearest|bilinear|bicubic|lanczos` and
`RUSTYPANO_DIALOG=options|metadata|hotspots|view|about` and
`RUSTYPANO_DROP_HINT=a.jpg,b.fsv` (drag-and-drop overlay) set up that frame.

## License

MIT, see [LICENSE](LICENSE).

Cached tiles are encoded with the `jpeg-encoder` crate, which is based in part
on the work of the Independent JPEG Group.
