#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 RustyPano contributors
#
# Build the browser version into <output dir> (default: dist/web).
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli with the same
# version as the wasm-bindgen crate in Cargo.lock. Uses wasm-opt if installed.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$root/dist/web}"
cd "$root"

want="$(grep -A1 '^name = "wasm-bindgen"$' Cargo.lock | sed -n 's/^version = "\(.*\)"/\1/p')"
have="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')"
if [ "$want" != "$have" ]; then
  echo "wasm-bindgen-cli $want is required (found: ${have:-none})." >&2
  echo "Install it with: cargo install wasm-bindgen-cli --version $want --locked" >&2
  exit 1
fi

cargo build --release --locked --target wasm32-unknown-unknown
rm -rf "$out"
mkdir -p "$out"
wasm-bindgen --target web --no-typescript --out-dir "$out" --out-name rustypano \
  target/wasm32-unknown-unknown/release/rustypano.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O2 --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$out/rustypano_bg.wasm" -o "$out/rustypano_bg.wasm"
fi
cp packaging/web/index.html "$out/"
cp assets/icon256.png "$out/"
echo "Web build written to $out"
