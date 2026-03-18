//! BM25 Turbo -- The fastest BM25 information retrieval engine in any language.
//!
//! # Quick Start
//!
//! ```no_run
//! use bm25_turbo::{BM25Builder, Method};
//!
//! let index = BM25Builder::new()
//!     .method(Method::Lucene)
//!     .k1(1.5)
//!     .b(0.75)
//!     .build_from_corpus(&["document one", "document two"])
//!     .expect("failed to build index");
//!
//! let results = index.search("query", 10).expect("search failed");
//! for (doc_id, score) in results.doc_ids.iter().zip(results.scores.iter()) {
//!     println!("doc {} score {:.4}", doc_id, score);
//! }
//! ```
//!
//! # Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `parallel` | yes | Multi-threaded indexing and querying via Rayon |
//! | `persistence` | yes | Save/load indexes with memory-mapped I/O |
//! | `simd` | yes | SIMD-accelerated score accumulation |
//! | `server` | no | Axum HTTP server |
//! | `distributed` | no | gRPC distributed query via Tonic |
//! | `mcp` | no | Model Context Protocol server |
//! | `huggingface` | no | HuggingFace Hub push/pull |
//! | `ann` | no | WAND / Block-Max WAND approximate search |
//! | `wasm` | no | WASM target support (mutually exclusive with parallel, persistence, simd) |
//! | `full` | no | All features except wasm |

// Compile-time guard: wasm and cli are mutually exclusive.
// The cli crate enables `full` which conflicts with wasm constraints.
#[cfg(all(feature = "wasm", feature = "server"))]
compile_error!("The `wasm` and `server` features are mutually exclusive. WASM targets cannot use server functionality.");

#[cfg(all(feature = "wasm", feature = "parallel"))]
compile_error!("The `wasm` and `parallel` features are mutually exclusive. WASM targets cannot use threading.");

#[cfg(all(feature = "wasm", feature = "persistence"))]
compile_error!("The `wasm` and `persistence` features are mutually exclusive. WASM targets cannot use filesystem/mmap.");

// Always-present modules (no feature gate needed).
pub mod csc;
pub mod error;
pub mod index;
pub mod scoring;
pub mod selection;
pub mod stopwords;
pub mod tokenizer;
pub mod types;

// Phase 1: Performance optimization modules (always present).
pub mod query_cache;
pub mod streaming;

// WAL is always compiled (uses in-memory mode without persistence).
pub mod wal;

// Feature-gated modules.
#[cfg(feature = "persistence")]
pub mod persistence;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "ann")]
pub mod wand;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience.
pub use error::{Error, Result};
pub use index::{BM25Builder, BM25Index};
pub use query_cache::QueryCache;
pub use streaming::StreamingBuilder;
pub use tokenizer::Tokenizer;
pub use types::{BM25Params, Method, Results, Tokenized};
