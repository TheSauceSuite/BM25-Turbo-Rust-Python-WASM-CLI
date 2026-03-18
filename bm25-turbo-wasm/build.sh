#!/usr/bin/env bash
# Build the BM25 Turbo WASM package for npm publishing.
#
# Prerequisites:
#   - wasm-pack: cargo install wasm-pack
#   - wasm32-unknown-unknown target: rustup target add wasm32-unknown-unknown
#
# Output is written to ./pkg/ which can be published directly with `npm publish`.
#
# Options:
#   --target web     (default) Browser ESM output
#   --target nodejs  Node.js CommonJS output
#   --target bundler Bundler-compatible output (webpack, rollup, etc.)
#
# To skip TypeScript declaration generation, add --no-typescript.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building bm25-turbo-wasm..."
wasm-pack build --target web --out-dir pkg --release

echo "Build complete. Output in ./pkg/"
echo "To publish: cd pkg && npm publish"
