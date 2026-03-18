//! Index persistence: save and load with optional mmap support.
//!
//! The binary format stores:
//! - A fixed-size header (64 bytes: magic, version, flags, dimensions, checksum)
//! - CSC arrays: data [f32], indices [u32], indptr [u64]
//! - Metadata sections: BM25Params JSON, vocabulary JSON, avg_doc_len f32
//! - Footer: three u64 offsets (params_offset, vocab_offset, avg_doc_len_offset)
//!
//! Checksum covers all bytes after the header (payload + metadata + footer).
//!
//! When the `persistence` feature is enabled, `load_mmap()` uses memory-mapped I/O
//! for zero-copy access to the CSC arrays.

use std::collections::HashMap;
use std::io::{Cursor, Read as _, Write as _};
use std::path::Path;

use tracing::instrument;

#[cfg(feature = "persistence")]
use bytemuck;
#[cfg(feature = "persistence")]
use memmap2::Mmap;
#[cfg(feature = "persistence")]
use xxhash_rust::xxh64::xxh64;

use crate::csc::CscMatrix;
use crate::error::{Error, Result};
use crate::index::BM25Index;
use crate::tokenizer::Tokenizer;
use crate::types::{BM25Params, Method};

#[cfg(feature = "persistence")]
use crate::csc::CscHeader;

/// Magic number for BM25 Turbo index files: "BM25" in ASCII.
pub const MAGIC: [u8; 4] = *b"BM25";

/// Current format version.
pub const FORMAT_VERSION: u16 = 1;

/// Footer size: 3 x u64 offsets = 24 bytes.
const FOOTER_SIZE: usize = 24;

/// Header size: 64 bytes.
const HEADER_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Method <-> u32 flag encoding
// ---------------------------------------------------------------------------

#[cfg(feature = "persistence")]
fn method_to_flags(method: Method) -> u32 {
    match method {
        Method::Robertson => 0,
        Method::Lucene => 1,
        Method::Atire => 2,
        Method::Bm25l => 3,
        Method::Bm25Plus => 4,
    }
}

#[cfg(feature = "persistence")]
fn flags_to_method(flags: u32) -> Result<Method> {
    match flags {
        0 => Ok(Method::Robertson),
        1 => Ok(Method::Lucene),
        2 => Ok(Method::Atire),
        3 => Ok(Method::Bm25l),
        4 => Ok(Method::Bm25Plus),
        _ => Err(Error::IndexCorrupted(format!(
            "unknown method flag: {}",
            flags
        ))),
    }
}

// ===========================================================================
// F-009: save()
// ===========================================================================

/// Save an index to a file.
///
/// Writes to a temporary file first, then atomically renames to the target path.
///
/// File format:
/// ```text
/// [CscHeader: 64 bytes]
/// [data: nnz * 4 bytes (f32 LE)]
/// [indices: nnz * 4 bytes (u32 LE)]
/// [indptr: (vocab_size+1) * 8 bytes (u64 LE)]
/// [params_json: variable length]
/// [vocab_json: variable length]
/// [avg_doc_len: 4 bytes (f32 LE)]
/// [params_offset: 8 bytes (u64 LE)]  \
/// [vocab_offset: 8 bytes (u64 LE)]    > footer (24 bytes)
/// [avg_doc_len_offset: 8 bytes (u64 LE)] /
/// ```
///
/// The header's `checksum` field covers all bytes after the header.
#[cfg(feature = "persistence")]
#[instrument(skip(index), fields(path = %path.display()))]
pub fn save(index: &BM25Index, path: &Path) -> Result<()> {
    // Build temp path in the same directory for atomic rename.
    let tmp_path = path.with_extension("tmp");

    // Prepare header with checksum = 0 (will be filled in later).
    let header = CscHeader {
        magic: CscHeader::MAGIC,
        version: CscHeader::VERSION,
        flags: method_to_flags(index.params.method),
        num_docs: index.num_docs as u64,
        num_terms: index.matrix.vocab_size as u64,
        nnz: index.matrix.data.len() as u64,
        checksum: 0,
        reserved: [0u8; 16],
    };

    // Serialize metadata sections.
    let params_json = serde_json::to_vec(&index.params)?;

    // Vocabulary: serialize as Vec<String> in token-ID order.
    let vocab_ordered: Vec<&str> = {
        let mut v = vec![""; index.vocab.len()];
        for (token, &id) in &index.vocab {
            v[id as usize] = token.as_str();
        }
        v
    };
    let vocab_json = serde_json::to_vec(&vocab_ordered)?;

    // Build the full payload buffer (everything after header).
    let nnz = index.matrix.data.len();
    let indptr_len = index.matrix.indptr.len(); // vocab_size + 1
    let data_bytes_len = nnz * 4;
    let indices_bytes_len = nnz * 4;
    let indptr_bytes_len = indptr_len * 8;
    let payload_size = data_bytes_len
        + indices_bytes_len
        + indptr_bytes_len
        + params_json.len()
        + vocab_json.len()
        + 4 // avg_doc_len f32
        + FOOTER_SIZE;

    let mut payload = Vec::with_capacity(payload_size);

    // Write CSC data array (f32 LE).
    for &val in &index.matrix.data {
        payload.extend_from_slice(&val.to_le_bytes());
    }

    // Write CSC indices array (u32 LE).
    for &val in &index.matrix.indices {
        payload.extend_from_slice(&val.to_le_bytes());
    }

    // Write CSC indptr array (u64 LE).
    for &val in &index.matrix.indptr {
        payload.extend_from_slice(&val.to_le_bytes());
    }

    // Record offsets relative to start of file.
    let params_offset = (HEADER_SIZE + payload.len()) as u64;
    payload.extend_from_slice(&params_json);

    let vocab_offset = (HEADER_SIZE + payload.len()) as u64;
    payload.extend_from_slice(&vocab_json);

    let avg_doc_len_offset = (HEADER_SIZE + payload.len()) as u64;
    payload.extend_from_slice(&index.avg_doc_len.to_le_bytes());

    // Footer: three u64 offsets.
    payload.extend_from_slice(&params_offset.to_le_bytes());
    payload.extend_from_slice(&vocab_offset.to_le_bytes());
    payload.extend_from_slice(&avg_doc_len_offset.to_le_bytes());

    // Compute xxhash64 checksum over the entire payload.
    let checksum = xxh64(&payload, 0);

    // Write header with correct checksum.
    let final_header = CscHeader {
        checksum,
        ..header
    };
    let final_header_bytes: &[u8] = bytemuck::bytes_of(&final_header);

    // Write to temp file.
    {
        let mut file = std::fs::File::create(&tmp_path)?;
        file.write_all(final_header_bytes)?;
        file.write_all(&payload)?;
        file.flush()?;
        file.sync_all()?;
    }

    // Atomic rename.
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

// Non-persistence stub so the function signature exists regardless of feature.
#[cfg(not(feature = "persistence"))]
pub fn save(_index: &BM25Index, _path: &Path) -> Result<()> {
    Err(Error::FeatureNotEnabled("persistence".into()))
}

// ===========================================================================
// F-010: load()
// ===========================================================================

/// Load an index from a file.
///
/// Reads the entire file into memory, validates the header and checksum,
/// then reconstructs the `BM25Index`.
#[cfg(feature = "persistence")]
#[instrument(fields(path = %path.display()))]
pub fn load(path: &Path) -> Result<BM25Index> {
    let data = std::fs::read(path)?;
    load_from_bytes(&data)
}

#[cfg(not(feature = "persistence"))]
pub fn load(_path: &Path) -> Result<BM25Index> {
    Err(Error::FeatureNotEnabled("persistence".into()))
}

/// Internal: load from a byte slice (shared by load and load_mmap).
#[cfg(feature = "persistence")]
fn load_from_bytes(data: &[u8]) -> Result<BM25Index> {
    // Validate minimum size for header.
    if data.len() < HEADER_SIZE {
        return Err(Error::FileTruncated);
    }

    // Parse header via bytemuck.
    let header: &CscHeader = bytemuck::from_bytes(&data[..HEADER_SIZE]);

    // Validate magic.
    if header.magic != CscHeader::MAGIC {
        return Err(Error::IndexCorrupted("invalid magic".into()));
    }

    // Validate version.
    if header.version != CscHeader::VERSION {
        return Err(Error::UnsupportedVersion(header.version as u16));
    }

    // Decode method from flags.
    let method = flags_to_method(header.flags)?;

    let num_docs = header.num_docs;
    let num_terms = header.num_terms;
    let nnz = header.nnz as usize;
    let stored_checksum = header.checksum;

    // Compute expected minimum file size.
    let data_bytes_len = nnz * 4;
    let indices_bytes_len = nnz * 4;
    let indptr_bytes_len = (num_terms as usize + 1) * 8;
    let min_size = HEADER_SIZE + data_bytes_len + indices_bytes_len + indptr_bytes_len + FOOTER_SIZE;

    if data.len() < min_size {
        return Err(Error::FileTruncated);
    }

    // Validate checksum over all bytes after header.
    let payload = &data[HEADER_SIZE..];
    let computed_checksum = xxh64(payload, 0);
    if computed_checksum != stored_checksum {
        return Err(Error::ChecksumMismatch);
    }

    // Parse CSC arrays.
    let mut cursor = Cursor::new(payload);

    let csc_data = read_f32_vec(&mut cursor, nnz)?;
    let csc_indices = read_u32_vec(&mut cursor, nnz)?;
    let csc_indptr = read_u64_vec(&mut cursor, num_terms as usize + 1)?;

    // Read footer (last 24 bytes of file).
    let footer_start = data.len() - FOOTER_SIZE;
    let params_offset = u64::from_le_bytes(data[footer_start..footer_start + 8].try_into().unwrap());
    let vocab_offset = u64::from_le_bytes(data[footer_start + 8..footer_start + 16].try_into().unwrap());
    let avg_doc_len_offset = u64::from_le_bytes(data[footer_start + 16..footer_start + 24].try_into().unwrap());

    // Parse params JSON.
    let params_start = params_offset as usize;
    let params_end = vocab_offset as usize;
    if params_start > data.len() || params_end > data.len() || params_start > params_end {
        return Err(Error::FileTruncated);
    }
    let mut params: BM25Params = serde_json::from_slice(&data[params_start..params_end])?;
    // Ensure the method from the header is consistent.
    params.method = method;

    // Parse vocabulary JSON.
    let vocab_start = vocab_offset as usize;
    let vocab_end = avg_doc_len_offset as usize;
    if vocab_start > data.len() || vocab_end > data.len() || vocab_start > vocab_end {
        return Err(Error::FileTruncated);
    }
    let vocab_ordered: Vec<String> = serde_json::from_slice(&data[vocab_start..vocab_end])?;

    // Build vocab HashMap and inverse.
    let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_ordered.len());
    for (id, token) in vocab_ordered.iter().enumerate() {
        vocab.insert(token.clone(), id as u32);
    }
    let vocab_inv = vocab_ordered;

    // Parse avg_doc_len.
    let adl_start = avg_doc_len_offset as usize;
    if adl_start + 4 > data.len() {
        return Err(Error::FileTruncated);
    }
    let avg_doc_len = f32::from_le_bytes(data[adl_start..adl_start + 4].try_into().unwrap());

    let matrix = CscMatrix {
        data: csc_data,
        indices: csc_indices,
        indptr: csc_indptr,
        num_docs: num_docs as u32,
        vocab_size: num_terms as u32,
    };

    // Reconstruct per-term document frequencies from the CSC indptr.
    let doc_freqs: Vec<u32> = (0..matrix.vocab_size)
        .map(|t| {
            let start = matrix.indptr[t as usize];
            let end = matrix.indptr[t as usize + 1];
            (end - start) as u32
        })
        .collect();

    Ok(BM25Index {
        matrix,
        vocab,
        vocab_inv,
        params,
        avg_doc_len,
        num_docs: num_docs as u32,
        tokenizer: Tokenizer::default(),
        doc_freqs,
        cache: None,
        #[cfg(feature = "ann")]
        block_max_index: None,
    })
}

// ===========================================================================
// F-011: load_mmap()
// ===========================================================================

/// Load an index using memory-mapped I/O for zero-copy access to CSC arrays.
///
/// The vocabulary and params are still read into heap memory (they are JSON),
/// but the CSC data, indices, and indptr arrays are served directly from the
/// mmap'd region without copying.
///
/// Returns an `MmapBM25Index` that holds the mmap handle and provides the
/// same query interface as `BM25Index`.
#[cfg(feature = "persistence")]
#[instrument(fields(path = %path.display()))]
pub fn load_mmap(path: &Path) -> Result<MmapBM25Index> {
    #[cfg(target_arch = "wasm32")]
    return Err(Error::MmapUnavailable);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data: &[u8] = &mmap;

        // Validate minimum size for header.
        if data.len() < HEADER_SIZE {
            return Err(Error::FileTruncated);
        }

        // Parse header via bytemuck.
        let header: &CscHeader = bytemuck::from_bytes(&data[..HEADER_SIZE]);

        // Validate magic.
        if header.magic != CscHeader::MAGIC {
            return Err(Error::IndexCorrupted("invalid magic".into()));
        }

        // Validate version.
        if header.version != CscHeader::VERSION {
            return Err(Error::UnsupportedVersion(header.version as u16));
        }

        let method = flags_to_method(header.flags)?;
        let num_docs = header.num_docs;
        let num_terms = header.num_terms;
        let nnz = header.nnz as usize;
        let stored_checksum = header.checksum;

        let data_bytes_len = nnz * 4;
        let indices_bytes_len = nnz * 4;
        let indptr_bytes_len = (num_terms as usize + 1) * 8;
        let min_size =
            HEADER_SIZE + data_bytes_len + indices_bytes_len + indptr_bytes_len + FOOTER_SIZE;

        if data.len() < min_size {
            return Err(Error::FileTruncated);
        }

        // Validate checksum.
        let payload = &data[HEADER_SIZE..];
        let computed_checksum = xxh64(payload, 0);
        if computed_checksum != stored_checksum {
            return Err(Error::ChecksumMismatch);
        }

        // Compute byte ranges for CSC arrays within the file.
        let data_start = HEADER_SIZE;
        let data_end = data_start + data_bytes_len;
        let indices_start = data_end;
        let indices_end = indices_start + indices_bytes_len;
        let indptr_start = indices_end;
        let indptr_end = indptr_start + indptr_bytes_len;

        // Read footer.
        let footer_start = data.len() - FOOTER_SIZE;
        let params_offset =
            u64::from_le_bytes(data[footer_start..footer_start + 8].try_into().unwrap()) as usize;
        let vocab_offset = u64::from_le_bytes(
            data[footer_start + 8..footer_start + 16].try_into().unwrap(),
        ) as usize;
        let avg_doc_len_offset = u64::from_le_bytes(
            data[footer_start + 16..footer_start + 24].try_into().unwrap(),
        ) as usize;

        // Parse params JSON.
        if params_offset > data.len() || vocab_offset > data.len() || params_offset > vocab_offset {
            return Err(Error::FileTruncated);
        }
        let mut params: BM25Params =
            serde_json::from_slice(&data[params_offset..vocab_offset])?;
        params.method = method;

        // Parse vocabulary JSON.
        if avg_doc_len_offset > data.len() || vocab_offset > avg_doc_len_offset {
            return Err(Error::FileTruncated);
        }
        let vocab_ordered: Vec<String> =
            serde_json::from_slice(&data[vocab_offset..avg_doc_len_offset])?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vocab_ordered.len());
        for (id, token) in vocab_ordered.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
        }

        // Parse avg_doc_len.
        if avg_doc_len_offset + 4 > data.len() {
            return Err(Error::FileTruncated);
        }
        let avg_doc_len = f32::from_le_bytes(
            data[avg_doc_len_offset..avg_doc_len_offset + 4]
                .try_into()
                .unwrap(),
        );

        Ok(MmapBM25Index {
            _mmap: mmap,
            data_range: (data_start, data_end),
            indices_range: (indices_start, indices_end),
            indptr_range: (indptr_start, indptr_end),
            num_docs: num_docs as u32,
            vocab_size: num_terms as u32,
            vocab,
            vocab_inv: vocab_ordered,
            params,
            avg_doc_len,
            tokenizer: Tokenizer::default(),
        })
    }
}

/// An index backed by a memory-mapped file.
///
/// The CSC arrays (data, indices, indptr) are served directly from the mmap
/// region with no heap copy. The vocabulary and params are owned in memory.
/// The `_mmap` handle keeps the mapping alive for the lifetime of this struct.
#[cfg(feature = "persistence")]
pub struct MmapBM25Index {
    /// The memory-mapped file handle. Must stay alive as long as we reference its bytes.
    _mmap: Mmap,
    /// Byte range of the data array within the mmap.
    data_range: (usize, usize),
    /// Byte range of the indices array within the mmap.
    indices_range: (usize, usize),
    /// Byte range of the indptr array within the mmap.
    indptr_range: (usize, usize),
    /// Number of documents.
    pub num_docs: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Token -> ID mapping.
    pub vocab: HashMap<String, u32>,
    /// ID -> token mapping.
    pub vocab_inv: Vec<String>,
    /// Scoring parameters.
    pub params: BM25Params,
    /// Average document length.
    pub avg_doc_len: f32,
    /// Tokenizer for queries.
    pub tokenizer: Tokenizer,
}

#[cfg(feature = "persistence")]
impl MmapBM25Index {
    /// Get the CSC data array as a slice (zero-copy from mmap).
    fn data_slice(&self) -> &[f32] {
        let bytes = &self._mmap[self.data_range.0..self.data_range.1];
        bytemuck::cast_slice(bytes)
    }

    /// Get the CSC indices array as a slice (zero-copy from mmap).
    fn indices_slice(&self) -> &[u32] {
        let bytes = &self._mmap[self.indices_range.0..self.indices_range.1];
        bytemuck::cast_slice(bytes)
    }

    /// Get the CSC indptr array as a slice (zero-copy from mmap).
    fn indptr_slice(&self) -> &[u64] {
        let bytes = &self._mmap[self.indptr_range.0..self.indptr_range.1];
        bytemuck::cast_slice(bytes)
    }

    /// Get a column (term) from the CSC matrix: (scores, doc_ids).
    #[inline]
    pub fn column(&self, term_id: u32) -> (&[f32], &[u32]) {
        let indptr = self.indptr_slice();
        let start = indptr[term_id as usize] as usize;
        let end = indptr[term_id as usize + 1] as usize;
        let data = self.data_slice();
        let indices = self.indices_slice();
        (&data[start..end], &indices[start..end])
    }

    /// Query the mmap'd index and return the top-k results.
    pub fn search(&self, query: &str, k: usize) -> Result<crate::types::Results> {
        if k == 0 {
            return Err(Error::InvalidParameter("k must be > 0".to_string()));
        }

        let query_tokens = self.tokenizer.tokenize(query);
        let token_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.vocab.get(t.as_str()).copied())
            .collect();

        if token_ids.is_empty() {
            return Ok(crate::types::Results {
                doc_ids: Vec::new(),
                scores: Vec::new(),
            });
        }

        let mut scores = vec![0.0f32; self.num_docs as usize];
        for &tid in &token_ids {
            if tid < self.vocab_size {
                let (col_scores, col_indices) = self.column(tid);
                for (i, &doc_id) in col_indices.iter().enumerate() {
                    scores[doc_id as usize] += col_scores[i];
                }
            }
        }

        Ok(crate::selection::top_k(&scores, k))
    }

    /// Convert this mmap'd index into an owned `BM25Index` by copying the CSC arrays.
    pub fn into_owned(self) -> BM25Index {
        let data = self.data_slice().to_vec();
        let indices = self.indices_slice().to_vec();
        let indptr = self.indptr_slice().to_vec();

        let matrix = CscMatrix {
            data,
            indices,
            indptr,
            num_docs: self.num_docs,
            vocab_size: self.vocab_size,
        };

        // Reconstruct per-term document frequencies from the CSC indptr.
        let doc_freqs: Vec<u32> = (0..matrix.vocab_size)
            .map(|t| {
                let start = matrix.indptr[t as usize];
                let end = matrix.indptr[t as usize + 1];
                (end - start) as u32
            })
            .collect();

        BM25Index {
            matrix,
            vocab: self.vocab,
            vocab_inv: self.vocab_inv,
            params: self.params,
            avg_doc_len: self.avg_doc_len,
            num_docs: self.num_docs,
            tokenizer: self.tokenizer,
            doc_freqs,
            cache: None,
            #[cfg(feature = "ann")]
            block_max_index: None,
        }
    }
}

/// Try mmap first, fall back to regular load.
///
/// Returns an owned `BM25Index` in both cases (mmap version is converted via
/// `into_owned()` on failure or by design for a unified return type).
#[cfg(feature = "persistence")]
pub fn mmap_or_load(path: &Path) -> Result<BM25Index> {
    match load_mmap(path) {
        Ok(mmap_index) => Ok(mmap_index.into_owned()),
        Err(_) => load(path),
    }
}

#[cfg(not(feature = "persistence"))]
pub fn mmap_or_load(_path: &Path) -> Result<BM25Index> {
    Err(Error::FeatureNotEnabled("persistence".into()))
}

// ===========================================================================
// Helper functions
// ===========================================================================

#[cfg(feature = "persistence")]
fn read_f32_vec(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    cursor.read_exact(&mut buf).map_err(|_| Error::FileTruncated)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(feature = "persistence")]
fn read_u32_vec(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<u32>> {
    let mut buf = vec![0u8; count * 4];
    cursor.read_exact(&mut buf).map_err(|_| Error::FileTruncated)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(feature = "persistence")]
fn read_u64_vec(cursor: &mut Cursor<&[u8]>, count: usize) -> Result<Vec<u64>> {
    let mut buf = vec![0u8; count * 8];
    cursor.read_exact(&mut buf).map_err(|_| Error::FileTruncated)?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::BM25Builder;
    use tempfile::tempdir;

    fn build_test_index() -> BM25Index {
        let corpus = &[
            "the quick brown fox",
            "the lazy dog",
            "brown fox jumps over the lazy dog",
            "quick quick quick fox",
            "the dog sat on the mat",
        ];
        BM25Builder::new()
            .build_from_corpus(corpus)
            .expect("build should succeed")
    }

    /// TEST-P5-001: Save/load round-trip produces identical query results.
    #[cfg(feature = "persistence")]
    #[test]
    fn save_load_round_trip() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.bm25");

        save(&index, &path).expect("save should succeed");

        let loaded = load(&path).expect("load should succeed");

        // Verify identical search results.
        let queries = &["quick", "dog", "brown fox", "mat"];
        for query in queries {
            let original = index.search(query, 5).unwrap();
            let restored = loaded.search(query, 5).unwrap();
            assert_eq!(
                original.doc_ids, restored.doc_ids,
                "doc_ids mismatch for query '{}'",
                query
            );
            assert_eq!(
                original.scores, restored.scores,
                "scores mismatch for query '{}'",
                query
            );
        }
    }

    /// TEST-P5-002: Save/mmap round-trip produces identical query results.
    #[cfg(feature = "persistence")]
    #[test]
    fn save_mmap_round_trip() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mmap.bm25");

        save(&index, &path).expect("save should succeed");

        let mmap_index = load_mmap(&path).expect("load_mmap should succeed");

        let queries = &["quick", "dog", "brown fox", "mat"];
        for query in queries {
            let original = index.search(query, 5).unwrap();
            let mmap_results = mmap_index.search(query, 5).unwrap();
            assert_eq!(
                original.doc_ids, mmap_results.doc_ids,
                "doc_ids mismatch for query '{}' (mmap)",
                query
            );
            assert_eq!(
                original.scores, mmap_results.scores,
                "scores mismatch for query '{}' (mmap)",
                query
            );
        }
    }

    /// TEST-P5-003: Corrupt magic bytes produce IndexCorrupted error.
    #[cfg(feature = "persistence")]
    #[test]
    fn corrupt_magic_bytes() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt_magic.bm25");

        save(&index, &path).unwrap();

        // Corrupt the first 4 bytes.
        let mut data = std::fs::read(&path).unwrap();
        data[0] = 0xFF;
        data[1] = 0xFF;
        data[2] = 0xFF;
        data[3] = 0xFF;
        std::fs::write(&path, &data).unwrap();

        match load(&path) {
            Err(Error::IndexCorrupted(_)) => {} // expected
            Err(e) => panic!("Expected IndexCorrupted, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    /// TEST-P5-004: Truncated file produces FileTruncated error.
    #[cfg(feature = "persistence")]
    #[test]
    fn truncated_file() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.bm25");

        save(&index, &path).unwrap();

        // Write only the header + a few bytes.
        let data = std::fs::read(&path).unwrap();
        let truncated = &data[..HEADER_SIZE + 10];
        std::fs::write(&path, truncated).unwrap();

        match load(&path) {
            Err(Error::FileTruncated) => {} // expected
            Err(e) => panic!("Expected FileTruncated, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    /// TEST-P5-005: Checksum mismatch produces ChecksumMismatch error.
    #[cfg(feature = "persistence")]
    #[test]
    fn checksum_mismatch() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("checksum_bad.bm25");

        save(&index, &path).unwrap();

        // Flip a byte in the payload (not the header).
        let mut data = std::fs::read(&path).unwrap();
        let payload_byte = HEADER_SIZE + 5;
        if payload_byte < data.len() {
            data[payload_byte] ^= 0xFF;
        }
        std::fs::write(&path, &data).unwrap();

        match load(&path) {
            Err(Error::ChecksumMismatch) => {} // expected
            Err(e) => panic!("Expected ChecksumMismatch, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    /// TEST-P5-006: Atomic write -- no .tmp file remains after save.
    #[cfg(feature = "persistence")]
    #[test]
    fn atomic_write_no_tmp_remains() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("atomic.bm25");

        save(&index, &path).unwrap();

        // The target file should exist.
        assert!(path.exists(), "Target file should exist");

        // No .tmp file should remain.
        let tmp_path = path.with_extension("tmp");
        assert!(!tmp_path.exists(), "Temp file should not remain after save");
    }

    /// Verify all five BM25 variants round-trip correctly.
    #[cfg(feature = "persistence")]
    #[test]
    fn round_trip_all_variants() {
        let corpus = &[
            "hello world",
            "foo bar baz",
            "cat dog fish",
            "one two three four",
            "alpha beta gamma",
            "hello again here",
        ];
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        let dir = tempdir().unwrap();

        for method in &methods {
            let index = BM25Builder::new()
                .method(*method)
                .build_from_corpus(corpus)
                .unwrap();

            let path = dir.path().join(format!("variant_{:?}.bm25", method));
            save(&index, &path).unwrap();
            let loaded = load(&path).unwrap();

            let original = index.search("hello", 3).unwrap();
            let restored = loaded.search("hello", 3).unwrap();
            assert_eq!(
                original.doc_ids, restored.doc_ids,
                "doc_ids mismatch for {:?}",
                method
            );
            assert_eq!(
                original.scores, restored.scores,
                "scores mismatch for {:?}",
                method
            );
        }
    }

    /// Verify mmap_or_load fallback works.
    #[cfg(feature = "persistence")]
    #[test]
    fn mmap_or_load_works() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("mmap_or_load.bm25");

        save(&index, &path).unwrap();
        let loaded = mmap_or_load(&path).unwrap();

        let original = index.search("quick", 3).unwrap();
        let restored = loaded.search("quick", 3).unwrap();
        assert_eq!(original.doc_ids, restored.doc_ids);
        assert_eq!(original.scores, restored.scores);
    }

    /// TEST-P5-VERSION: Writing a future version produces UnsupportedVersion error.
    #[cfg(feature = "persistence")]
    #[test]
    fn version_mismatch_returns_unsupported_version() {
        let index = build_test_index();
        let dir = tempdir().unwrap();
        let path = dir.path().join("version_bad.bm25");

        save(&index, &path).unwrap();

        // Overwrite the version field in the header with a future version (99).
        // Version is a u32 at byte offset 8 (after 8-byte magic).
        let mut data = std::fs::read(&path).unwrap();
        let future_version: u32 = 99;
        data[8..12].copy_from_slice(&future_version.to_le_bytes());

        // Also need to fix the checksum since we changed the header, but checksum
        // is validated over payload (after header), so changing the header version
        // alone should trigger UnsupportedVersion before checksum validation.
        std::fs::write(&path, &data).unwrap();

        match load(&path) {
            Err(Error::UnsupportedVersion(v)) => {
                assert_eq!(v, 99, "Expected version 99, got {}", v);
            }
            Err(e) => panic!("Expected UnsupportedVersion, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    /// TEST-P5-LARGE: Large index (10K docs) save/load round-trip.
    #[cfg(feature = "persistence")]
    #[test]
    fn large_index_10k_docs_round_trip() {
        // Generate a corpus of 10,000 documents with some vocabulary overlap.
        let words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
            "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "search",
            "index", "query", "document", "retrieval", "ranking", "score", "term",
        ];

        let mut corpus_strings: Vec<String> = Vec::with_capacity(10_000);
        let mut rng_state = 12345u64;

        for _ in 0..10_000 {
            // Each doc gets 5-15 words.
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let num_words = 5 + ((rng_state >> 33) as usize % 11);
            let mut doc = Vec::with_capacity(num_words);
            for _ in 0..num_words {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let word_idx = (rng_state >> 33) as usize % words.len();
                doc.push(words[word_idx]);
            }
            corpus_strings.push(doc.join(" "));
        }

        let corpus_refs: Vec<&str> = corpus_strings.iter().map(|s| s.as_str()).collect();

        let index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .expect("build 10K doc index should succeed");
        assert_eq!(index.num_docs(), 10_000);

        let dir = tempdir().unwrap();
        let path = dir.path().join("large_10k.bm25");

        save(&index, &path).expect("save 10K doc index should succeed");
        let loaded = load(&path).expect("load 10K doc index should succeed");

        assert_eq!(loaded.num_docs(), 10_000);

        // Verify search results match across multiple queries.
        let queries = &["alpha", "search query", "gamma delta", "omega"];
        for query in queries {
            let original = index.search(query, 10).unwrap();
            let restored = loaded.search(query, 10).unwrap();
            assert_eq!(
                original.doc_ids, restored.doc_ids,
                "doc_ids mismatch for query '{}' in 10K index",
                query
            );
            assert_eq!(
                original.scores, restored.scores,
                "scores mismatch for query '{}' in 10K index",
                query
            );
        }

        // Also verify mmap round-trip on the large index.
        let mmap_index = load_mmap(&path).expect("load_mmap 10K doc index should succeed");
        for query in queries {
            let original = index.search(query, 10).unwrap();
            let mmap_results = mmap_index.search(query, 10).unwrap();
            assert_eq!(
                original.doc_ids, mmap_results.doc_ids,
                "mmap doc_ids mismatch for query '{}' in 10K index",
                query
            );
            assert_eq!(
                original.scores, mmap_results.scores,
                "mmap scores mismatch for query '{}' in 10K index",
                query
            );
        }
    }
}
