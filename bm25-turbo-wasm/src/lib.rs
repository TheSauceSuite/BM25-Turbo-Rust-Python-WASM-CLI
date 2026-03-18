//! WASM npm package for BM25 Turbo.
//!
//! This crate is a thin wrapper that re-exports the `WasmBM25` class from
//! `bm25_turbo::wasm`. The actual implementation lives in
//! `bm25_turbo/src/wasm.rs` — this crate exists solely to produce the
//! `cdylib` target that `wasm-pack` needs to generate the npm package.

use wasm_bindgen::prelude::*;

// Re-export the WasmBM25 class so wasm-bindgen discovers it.
pub use bm25_turbo::wasm::WasmBM25;

/// Called automatically when the WASM module is instantiated.
/// Sets up `console_error_panic_hook` so that Rust panics produce
/// readable stack traces in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}
