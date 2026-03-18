//! WASM bindings for BM25 Turbo via wasm-bindgen.
//!
//! Provides a `WasmBM25` class that can be used from JavaScript/TypeScript
//! in the browser or Node.js. Uses `MemoryStorage` backend (no filesystem),
//! single-threaded execution (no rayon), and communicates via `JsValue`.
//!
//! This module is gated behind the `wasm` feature and is mutually exclusive
//! with `parallel`, `persistence`, and `simd` features.
//!
//! # Usage from JavaScript
//!
//! ```js
//! import { WasmBM25 } from 'bm25-turbo-wasm';
//!
//! const bm25 = WasmBM25.build(["document one", "document two", "document three"]);
//! const results = bm25.search("document", 2);
//! console.log(results); // { doc_ids: [0, 1], scores: [0.42, 0.38] }
//! ```

use wasm_bindgen::prelude::*;

use crate::csc::CscMatrix;
use crate::error::Error;
use crate::index::BM25Builder;
use crate::tokenizer::Tokenizer;
use crate::types::{BM25Params, Method, Results};

/// Parse a method string into a Method enum.
fn parse_method(method: &str) -> Result<Method, JsError> {
    match method.to_lowercase().as_str() {
        "robertson" => Ok(Method::Robertson),
        "lucene" => Ok(Method::Lucene),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::Bm25l),
        "bm25plus" | "bm25+" => Ok(Method::Bm25Plus),
        _ => Err(JsError::new(&format!("unknown method: {}", method))),
    }
}

/// BM25 search engine for WebAssembly.
///
/// Provides index construction, search, and serialization capabilities
/// entirely in WASM linear memory. No filesystem access, no threading.
#[wasm_bindgen]
pub struct WasmBM25 {
    matrix: CscMatrix,
    vocab: std::collections::HashMap<String, u32>,
    vocab_inv: Vec<String>,
    params: BM25Params,
    avg_doc_len: f32,
    num_docs: u32,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl WasmBM25 {
    /// Build a BM25 index from an array of document strings.
    ///
    /// # Arguments
    /// * `documents` - JavaScript array of strings
    /// * `method` - BM25 variant: "robertson", "lucene", "atire", "bm25l", "bm25plus"
    /// * `k1` - Term frequency saturation parameter (default: 1.5)
    /// * `b` - Document length normalization parameter (default: 0.75)
    #[wasm_bindgen(constructor)]
    pub fn new(
        documents: Vec<String>,
        method: Option<String>,
        k1: Option<f32>,
        b: Option<f32>,
    ) -> Result<WasmBM25, JsError> {
        let method_enum = match method {
            Some(ref m) => parse_method(m)?,
            None => Method::Lucene,
        };

        let corpus_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

        let mut builder = BM25Builder::new().method(method_enum);
        if let Some(k1) = k1 {
            builder = builder.k1(k1);
        }
        if let Some(b) = b {
            builder = builder.b(b);
        }

        let index = builder
            .build_from_corpus(&corpus_refs)
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(WasmBM25 {
            matrix: index.matrix,
            vocab: index.vocab,
            vocab_inv: index.vocab_inv,
            params: index.params,
            avg_doc_len: index.avg_doc_len,
            num_docs: index.num_docs,
            tokenizer: index.tokenizer,
        })
    }

    /// Search the index and return the top-k results.
    ///
    /// Returns a JavaScript object with `doc_ids` (Uint32Array) and `scores` (Float32Array).
    pub fn search(&self, query: &str, k: usize) -> Result<JsValue, JsError> {
        if k == 0 {
            return Err(JsError::new("k must be > 0"));
        }

        let query_tokens = self.tokenizer.tokenize(query);
        let token_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.vocab.get(t.as_str()).copied())
            .collect();

        if token_ids.is_empty() {
            let result = Results {
                doc_ids: Vec::new(),
                scores: Vec::new(),
            };
            return serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsError::new(&e.to_string()));
        }

        // Accumulate scores (single-threaded, no SIMD).
        let mut scores = vec![0.0f32; self.num_docs as usize];
        for &tid in &token_ids {
            if tid < self.matrix.vocab_size {
                let (col_scores, col_indices) = self.matrix.column(tid);
                for (i, &doc_id) in col_indices.iter().enumerate() {
                    scores[doc_id as usize] += col_scores[i];
                }
            }
        }

        let results = crate::selection::top_k(&scores, k);
        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Return the number of documents in the index.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.matrix.vocab_size
    }

    /// Return index statistics as a JSON string.
    pub fn stats(&self) -> String {
        format!(
            r#"{{"num_docs":{},"vocab_size":{},"method":"{}","k1":{},"b":{},"delta":{},"avg_doc_len":{}}}"#,
            self.num_docs,
            self.matrix.vocab_size,
            self.params.method,
            self.params.k1,
            self.params.b,
            self.params.delta,
            self.avg_doc_len
        )
    }

    /// Serialize the index to bytes for storage/transfer.
    ///
    /// The returned bytes can be loaded back via `WasmBM25.load_bytes()`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        let serializable = SerializableIndex {
            matrix: &self.matrix,
            vocab_inv: &self.vocab_inv,
            params: &self.params,
            avg_doc_len: self.avg_doc_len,
            num_docs: self.num_docs,
        };
        bincode::serialize(&serializable).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Load a pre-built index from bytes.
    ///
    /// Accepts bytes produced by `to_bytes()` or by the native Rust library's
    /// bincode serialization of the same format.
    #[wasm_bindgen(js_name = "loadBytes")]
    pub fn load_bytes(data: &[u8]) -> Result<WasmBM25, JsError> {
        let deser: DeserializableIndex =
            bincode::deserialize(data).map_err(|e| JsError::new(&e.to_string()))?;

        let mut vocab = std::collections::HashMap::with_capacity(deser.vocab_inv.len());
        for (id, token) in deser.vocab_inv.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }

        Ok(WasmBM25 {
            matrix: deser.matrix,
            vocab,
            vocab_inv: deser.vocab_inv,
            params: deser.params,
            avg_doc_len: deser.avg_doc_len,
            num_docs: deser.num_docs,
            tokenizer: Tokenizer::default(),
        })
    }
}

/// Serializable view of the index for bincode.
#[derive(serde::Serialize)]
struct SerializableIndex<'a> {
    matrix: &'a CscMatrix,
    vocab_inv: &'a Vec<String>,
    params: &'a BM25Params,
    avg_doc_len: f32,
    num_docs: u32,
}

/// Deserializable index from bincode.
#[derive(serde::Deserialize)]
struct DeserializableIndex {
    matrix: CscMatrix,
    vocab_inv: Vec<String>,
    params: BM25Params,
    avg_doc_len: f32,
    num_docs: u32,
}
