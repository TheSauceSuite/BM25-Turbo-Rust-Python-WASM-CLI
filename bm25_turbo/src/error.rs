//! Error types for BM25 Turbo.
//!
//! All fallible operations in the library return [`Result<T>`] which uses
//! the [`Error`] enum. Consumers can match on specific variants for
//! programmatic error handling.

use thiserror::Error;

/// All errors produced by BM25 Turbo.
#[derive(Debug, Error)]
pub enum Error {
    /// A parameter (k1, b, delta, top_k, etc.) was out of valid range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// The index file is corrupt or incompatible.
    #[error("index is corrupted: {0}")]
    IndexCorrupted(String),

    /// Tokenization failed (bad regex, encoding issue, etc.).
    #[error("tokenization failed: {0}")]
    TokenizationError(String),

    /// The index has not been built yet.
    #[error("index not built: call build() or load() before querying")]
    IndexNotBuilt,

    /// An I/O error occurred during persistence.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A feature-gated capability was requested but not compiled in.
    #[error("feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// A write-ahead log operation failed.
    #[error("WAL error: {0}")]
    WalError(String),

    /// A distributed query operation failed.
    #[error("distributed error: {0}")]
    DistributedError(String),

    /// Memory-mapped I/O is not available on the current platform (e.g., WASM).
    #[error("mmap is not available on this platform")]
    MmapUnavailable,

    /// The stored checksum does not match the computed checksum.
    #[error("checksum mismatch: index file may be corrupted")]
    ChecksumMismatch,

    /// The index file format version is not supported by this build.
    #[error("unsupported format version: {0}")]
    UnsupportedVersion(u16),

    /// The index file was truncated or incomplete.
    #[error("file is truncated or incomplete")]
    FileTruncated,

    /// A HuggingFace Hub operation failed.
    #[error("HuggingFace Hub error: {0}")]
    HuggingFaceError(String),
}

/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;
