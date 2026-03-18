//! Core type definitions for BM25 Turbo.
//!
//! Contains the BM25 scoring method enum, search results, tokenized document
//! representation, and other shared types used across modules.

use std::fmt;

use serde::{Deserialize, Serialize};

/// BM25 scoring variant.
///
/// Each variant defines a different IDF and term-frequency component formula.
/// See the crate-level documentation for the mathematical definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Method {
    /// Robertson BM25 (original 1994 formulation).
    Robertson,
    /// Lucene BM25 (log(1 + ...) IDF, used by Apache Lucene/Solr/Elasticsearch).
    #[default]
    Lucene,
    /// ATIRE BM25 (simplified IDF, (k1+1) numerator in TFC).
    Atire,
    /// BM25L (Lv & Zhai 2011, adds delta to term frequency).
    Bm25l,
    /// BM25+ (Lv & Zhai 2011, adds delta after TFC).
    Bm25Plus,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Method::Robertson => write!(f, "Robertson"),
            Method::Lucene => write!(f, "Lucene"),
            Method::Atire => write!(f, "ATIRE"),
            Method::Bm25l => write!(f, "BM25L"),
            Method::Bm25Plus => write!(f, "BM25+"),
        }
    }
}

/// Search results returned by a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    /// Document indices into the original corpus, sorted by descending score.
    pub doc_ids: Vec<u32>,
    /// BM25 scores corresponding to each document ID.
    pub scores: Vec<f32>,
}

/// A tokenized document: the integer token IDs and the document's token count.
#[derive(Debug, Clone)]
pub struct Tokenized {
    /// Token IDs (indices into the vocabulary).
    pub token_ids: Vec<u32>,
    /// Number of tokens in the document (before deduplication).
    pub length: u32,
}

/// Parameters for BM25 scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BM25Params {
    /// Term frequency saturation parameter. Default: 1.5.
    pub k1: f32,
    /// Document length normalization parameter. Default: 0.75.
    pub b: f32,
    /// Additive smoothing for BM25L and BM25+. Default: 0.5.
    pub delta: f32,
    /// Scoring variant.
    pub method: Method,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
            method: Method::default(),
        }
    }
}

impl BM25Params {
    /// Validate that all parameters are within their acceptable ranges.
    ///
    /// Delegates to [`crate::scoring::validate_params`].
    pub fn validate(&self) -> crate::error::Result<()> {
        crate::scoring::validate_params(self)
    }
}
