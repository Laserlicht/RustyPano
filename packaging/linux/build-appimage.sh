#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 RustyPano contributors
#
# Package a release build as an AppImage.
#   packaging/linux/build-appimage.sh <cargo target triple> <output dir>
# Needs network access to download appimagetool for the host architecture.
set -euo pipefail

target="${1:?target triple}"
out="${2:?output directory}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
version="$(grep -m1 '^version' "$root/Cargo.toml" | cut -d'"' -f2)"
case "$target" in
  x86_64-*) arch=x86_64 ;;
  aarch64-*) arch=aarch64 ;;
  *) echo "unsupported target $target" >&2; exit 1 ;;
esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
appdir="$work/RustyPano.AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" "$appdir/usr/share/icons/hicolor/256x256/apps"
install -m755 "$root/target/$target/release/rustypano" "$appdir/usr/bin/rustypano"
install -m644 "$root/packaging/linux/rustypano.desktop" "$appdir/usr/share/applications/rustypano.desktop"
install -m644 "$root/packaging/linux/rustypano.desktop" "$appdir/rustypano.desktop"
install -m644 "$root/assets/icon256.png" "$appdir/usr/share/icons/hicolor/256x256/apps/rustypano.png"
install -m644 "$root/assets/icon256.png" "$appdir/rustypano.png"
ln -s rustypano.png "$appdir/.DirIcon"
ln -s usr/bin/rustypano "$appdir/AppRun"

tool="$work/appimagetool"
curl -fsSL -o "$tool" \
  "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$arch.AppImage"
chmod +x "$tool"

mkdir -p "$out"
# Run appimagetool without FUSE (not available on CI runners).
APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$arch" "$tool" --no-appstream "$appdir" \
  "$out/RustyPano-$version-linux-$arch.AppImage"
