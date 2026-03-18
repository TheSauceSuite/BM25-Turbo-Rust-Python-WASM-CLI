//! Custom Compressed Sparse Column (CSC) matrix.
//!
//! The CSC matrix stores precomputed BM25 scores in column-major format where:
//! - Each column represents a vocabulary term
//! - Each row represents a document
//! - Non-zero entries are the BM25 score for (document, term)
//!
//! Three contiguous arrays:
//! - `data: Vec<f32>` -- BM25 scores
//! - `indices: Vec<u32>` -- document IDs for each score
//! - `indptr: Vec<u64>` -- column boundary offsets (length = vocab_size + 1)
//!
//! This layout is mmap-friendly and cache-line aligned for sequential access.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CscHeader -- 64-byte binary header for persistence (feature-gated)
// ---------------------------------------------------------------------------

/// Binary file header for serialized CSC matrices.
///
/// Exactly 64 bytes with `repr(C)` layout for mmap compatibility.
/// Uses bytemuck `Pod` + `Zeroable` for safe zero-copy casting.
#[cfg(feature = "persistence")]
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CscHeader {
    /// Magic bytes: `b"BM25\0\0\0\0"`.
    pub magic: [u8; 8],
    /// Format version (currently 1).
    pub version: u32,
    /// Flags: Method variant encoded as bits.
    pub flags: u32,
    /// Number of documents (rows).
    pub num_docs: u64,
    /// Number of vocabulary terms (columns).
    pub num_terms: u64,
    /// Number of non-zero entries.
    pub nnz: u64,
    /// xxhash64 checksum of payload bytes.
    pub checksum: u64,
    /// Reserved for future use, zero-filled.
    pub reserved: [u8; 16],
}

#[cfg(feature = "persistence")]
const _: () = assert!(std::mem::size_of::<CscHeader>() == 64);

#[cfg(feature = "persistence")]
impl CscHeader {
    /// The expected magic bytes for a BM25 Turbo index file.
    pub const MAGIC: [u8; 8] = *b"BM25\0\0\0\0";

    /// The current format version.
    pub const VERSION: u32 = 1;
}

// ---------------------------------------------------------------------------
// CscMatrix
// ---------------------------------------------------------------------------

/// Compressed Sparse Column matrix for BM25 score storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CscMatrix {
    /// BM25 scores (non-zero values).
    pub data: Vec<f32>,
    /// Document IDs corresponding to each score.
    pub indices: Vec<u32>,
    /// Column pointers: `indptr[i]..indptr[i+1]` is the range of entries for term `i`.
    pub indptr: Vec<u64>,
    /// Number of rows (documents).
    pub num_docs: u32,
    /// Number of columns (vocabulary terms).
    pub vocab_size: u32,
}

// Compile-time proof that CscMatrix is Send + Sync.
const _: () = {
    const fn _assert<T: Send + Sync>() {}
    _assert::<CscMatrix>();
};

impl CscMatrix {
    /// Create an empty CSC matrix with the given dimensions.
    pub fn new(num_docs: u32, vocab_size: u32) -> Self {
        Self {
            data: Vec::new(),
            indices: Vec::new(),
            indptr: vec![0; vocab_size as usize + 1],
            num_docs,
            vocab_size,
        }
    }

    /// Build a CSC matrix from COO-format triplets `(term_id, doc_id, score)`.
    ///
    /// Uses O(n) counting sort: count entries per column, compute `indptr` via
    /// prefix sum, then scatter `data` and `indices` into position.
    ///
    /// Duplicate `(term_id, doc_id)` pairs are merged by summing their scores.
    /// Row indices within each column are sorted after construction for
    /// cache-friendly access and dedup detection.
    pub fn from_triplets(
        triplets: &[(u32, u32, f32)],
        num_docs: u32,
        vocab_size: u32,
    ) -> Result<Self> {
        let vs = vocab_size as usize;
        let nnz = triplets.len();

        // Early validation.
        for &(term_id, doc_id, _score) in triplets {
            if term_id >= vocab_size {
                return Err(Error::InvalidParameter(format!(
                    "term_id {} >= vocab_size {}",
                    term_id, vocab_size
                )));
            }
            if doc_id >= num_docs {
                return Err(Error::InvalidParameter(format!(
                    "doc_id {} >= num_docs {}",
                    doc_id, num_docs
                )));
            }
        }

        // --- Step 1: Count entries per column ---
        let mut col_counts = vec![0u64; vs];
        for &(term_id, _, _) in triplets {
            col_counts[term_id as usize] += 1;
        }

        // --- Step 2: Compute indptr via prefix sum ---
        let mut indptr = vec![0u64; vs + 1];
        for i in 0..vs {
            indptr[i + 1] = indptr[i] + col_counts[i];
        }

        // --- Step 3: Scatter data and indices into position ---
        let mut data = vec![0.0f32; nnz];
        let mut indices = vec![0u32; nnz];
        // write_pos tracks the next write position for each column.
        let mut write_pos = vec![0u64; vs];

        for &(term_id, doc_id, score) in triplets {
            let col = term_id as usize;
            let pos = (indptr[col] + write_pos[col]) as usize;
            data[pos] = score;
            indices[pos] = doc_id;
            write_pos[col] += 1;
        }

        // --- Step 4: Sort row indices within each column and merge duplicates ---
        // We sort (doc_id, score) pairs within each column by doc_id using
        // an in-place sort on slices (small per-column, not full-array sort).
        let mut final_data = Vec::with_capacity(nnz);
        let mut final_indices = Vec::with_capacity(nnz);
        let mut final_indptr = vec![0u64; vs + 1];

        for col in 0..vs {
            let start = indptr[col] as usize;
            let end = indptr[col + 1] as usize;

            if start == end {
                final_indptr[col + 1] = final_indptr[col];
                continue;
            }

            // Build (doc_id, score) pairs for this column and sort by doc_id.
            let col_slice_indices = &indices[start..end];
            let col_slice_data = &data[start..end];
            let mut pairs: Vec<(u32, f32)> = col_slice_indices
                .iter()
                .zip(col_slice_data.iter())
                .map(|(&idx, &val)| (idx, val))
                .collect();
            pairs.sort_unstable_by_key(|&(doc_id, _)| doc_id);

            // Merge duplicates by summing scores.
            let mut prev_doc = pairs[0].0;
            let mut prev_score = pairs[0].1;
            let mut merged_count = 0u64;

            for &(doc_id, score) in &pairs[1..] {
                if doc_id == prev_doc {
                    // Duplicate -- sum scores.
                    prev_score += score;
                } else {
                    final_indices.push(prev_doc);
                    final_data.push(prev_score);
                    merged_count += 1;
                    prev_doc = doc_id;
                    prev_score = score;
                }
            }
            // Push the last entry.
            final_indices.push(prev_doc);
            final_data.push(prev_score);
            merged_count += 1;

            final_indptr[col + 1] = final_indptr[col] + merged_count;
        }

        let matrix = CscMatrix {
            data: final_data,
            indices: final_indices,
            indptr: final_indptr,
            num_docs,
            vocab_size,
        };

        Ok(matrix)
    }

    /// Return the number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Return the scores and document IDs for a given term (column).
    ///
    /// Returns `(&[f32], &[u32])` -- no allocation on the query path.
    #[inline]
    pub fn column(&self, term_id: u32) -> (&[f32], &[u32]) {
        let start = self.indptr[term_id as usize] as usize;
        let end = self.indptr[term_id as usize + 1] as usize;
        (&self.data[start..end], &self.indices[start..end])
    }

    /// Validate structural invariants of this CSC matrix.
    ///
    /// Checks:
    /// 1. `indptr` length equals `vocab_size + 1`.
    /// 2. `indptr` is monotonically non-decreasing.
    /// 3. Row indices within each column are sorted (strictly ascending).
    /// 4. No duplicate `(row, col)` entries.
    /// 5. All row indices are less than `num_docs`.
    pub fn validate(&self) -> Result<()> {
        // Check 1: indptr length.
        let expected_len = self.vocab_size as usize + 1;
        if self.indptr.len() != expected_len {
            return Err(Error::InvalidParameter(format!(
                "indptr length {} != vocab_size + 1 ({})",
                self.indptr.len(),
                expected_len
            )));
        }

        // Check 2: indptr monotonically non-decreasing.
        for i in 1..self.indptr.len() {
            if self.indptr[i] < self.indptr[i - 1] {
                return Err(Error::InvalidParameter(format!(
                    "indptr is not monotonic at index {}: {} < {}",
                    i,
                    self.indptr[i],
                    self.indptr[i - 1]
                )));
            }
        }

        // The total nnz implied by indptr must match data/indices lengths.
        let total_nnz = *self.indptr.last().unwrap_or(&0) as usize;
        if self.data.len() != total_nnz {
            return Err(Error::InvalidParameter(format!(
                "data length {} != indptr total nnz {}",
                self.data.len(),
                total_nnz
            )));
        }
        if self.indices.len() != total_nnz {
            return Err(Error::InvalidParameter(format!(
                "indices length {} != indptr total nnz {}",
                self.indices.len(),
                total_nnz
            )));
        }

        // Check 3, 4, 5: per-column validation.
        for col in 0..self.vocab_size as usize {
            let start = self.indptr[col] as usize;
            let end = self.indptr[col + 1] as usize;
            let col_indices = &self.indices[start..end];

            for i in 0..col_indices.len() {
                // Check 5: all indices < num_docs.
                if col_indices[i] >= self.num_docs {
                    return Err(Error::InvalidParameter(format!(
                        "index {} in column {} >= num_docs {}",
                        col_indices[i], col, self.num_docs
                    )));
                }

                // Check 3 & 4: strictly ascending (sorted and no duplicates).
                if i > 0 && col_indices[i] <= col_indices[i - 1] {
                    if col_indices[i] == col_indices[i - 1] {
                        return Err(Error::InvalidParameter(format!(
                            "duplicate doc_id {} in column {}",
                            col_indices[i], col
                        )));
                    }
                    return Err(Error::InvalidParameter(format!(
                        "unsorted indices in column {}: {} after {}",
                        col,
                        col_indices[i],
                        col_indices[i - 1]
                    )));
                }
            }
        }

        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix() {
        let m = CscMatrix::new(100, 50);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.indptr.len(), 51);
        assert!(m.validate().is_ok());
    }

    /// TEST-P2-001: Round-trip triplets.
    #[test]
    fn round_trip_triplets() {
        // (term_id, doc_id, score)
        let triplets = vec![
            (0u32, 0u32, 1.0f32),
            (0, 2, 2.5),
            (1, 1, 3.0),
            (2, 0, 0.5),
            (2, 1, 1.5),
            (2, 3, 4.0),
        ];
        let m = CscMatrix::from_triplets(&triplets, 4, 3).unwrap();
        assert!(m.validate().is_ok());
        assert_eq!(m.nnz(), 6);

        // Verify each triplet is found.
        for &(term_id, doc_id, score) in &triplets {
            let (scores, doc_ids) = m.column(term_id);
            let pos = doc_ids.iter().position(|&d| d == doc_id);
            assert!(pos.is_some(), "doc_id {} not found in column {}", doc_id, term_id);
            let pos = pos.unwrap();
            assert!(
                (scores[pos] - score).abs() < f32::EPSILON,
                "score mismatch for ({}, {}): expected {}, got {}",
                term_id,
                doc_id,
                score,
                scores[pos]
            );
        }
    }

    /// TEST-P2-002: Empty triplets produce valid empty CSC.
    #[test]
    fn from_triplets_empty() {
        let m = CscMatrix::from_triplets(&[], 10, 5).unwrap();
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.indptr.len(), 6);
        assert!(m.validate().is_ok());
        // Every column should be empty.
        for col in 0..5 {
            let (scores, ids) = m.column(col);
            assert!(scores.is_empty());
            assert!(ids.is_empty());
        }
    }

    /// TEST-P2-003: Single-column matrix.
    #[test]
    fn single_column_matrix() {
        let triplets = vec![
            (2u32, 0u32, 1.0f32),
            (2, 3, 2.0),
            (2, 1, 0.5),
        ];
        let m = CscMatrix::from_triplets(&triplets, 5, 4).unwrap();
        assert!(m.validate().is_ok());
        assert_eq!(m.nnz(), 3);

        // Columns 0, 1, 3 should be empty.
        for col in [0, 1, 3] {
            let (scores, _) = m.column(col);
            assert!(scores.is_empty());
        }
        // Column 2 should have 3 entries, sorted by doc_id.
        let (scores, doc_ids) = m.column(2);
        assert_eq!(doc_ids, &[0, 1, 3]);
        assert!((scores[0] - 1.0).abs() < f32::EPSILON);
        assert!((scores[1] - 0.5).abs() < f32::EPSILON);
        assert!((scores[2] - 2.0).abs() < f32::EPSILON);
    }

    /// TEST-P2-004: Duplicate (term_id, doc_id) pairs are merged by summing.
    #[test]
    fn duplicate_triplets_merged() {
        let triplets = vec![
            (0u32, 1u32, 1.0f32),
            (0, 1, 2.0), // duplicate
            (0, 2, 3.0),
        ];
        let m = CscMatrix::from_triplets(&triplets, 3, 1).unwrap();
        assert!(m.validate().is_ok());
        // Duplicates merged: only 2 unique entries in column 0.
        assert_eq!(m.nnz(), 2);
        let (scores, doc_ids) = m.column(0);
        assert_eq!(doc_ids, &[1, 2]);
        assert!((scores[0] - 3.0).abs() < f32::EPSILON); // 1.0 + 2.0
        assert!((scores[1] - 3.0).abs() < f32::EPSILON);
    }

    /// TEST-P2-005: Large-scale construction.
    #[test]
    fn large_scale_construction() {
        let num_docs = 1000u32;
        let vocab_size = 500u32;
        let mut triplets = Vec::with_capacity(100_000);
        // Deterministic pseudo-random generation using simple LCG.
        let mut rng_state = 42u64;
        for _ in 0..100_000 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let term_id = ((rng_state >> 32) as u32) % vocab_size;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let doc_id = ((rng_state >> 32) as u32) % num_docs;
            let score = ((rng_state & 0xFFFF) as f32) / 65535.0;
            triplets.push((term_id, doc_id, score));
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, vocab_size).unwrap();
        assert!(m.validate().is_ok());
        // nnz may be less than 100K due to duplicate merging.
        assert!(m.nnz() <= 100_000);
        assert!(m.nnz() > 0);

        // Verify column counts sum to nnz.
        let mut total = 0usize;
        for col in 0..vocab_size {
            let (scores, _) = m.column(col);
            total += scores.len();
        }
        assert_eq!(total, m.nnz());
    }

    /// TEST-P2-006: CscHeader is exactly 64 bytes.
    #[cfg(feature = "persistence")]
    #[test]
    fn header_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<CscHeader>(), 64);
    }

    /// TEST-P2-007: validate() catches invalid matrices.
    #[test]
    fn validate_catches_bad_indptr_length() {
        let m = CscMatrix {
            data: vec![],
            indices: vec![],
            indptr: vec![0, 0], // Wrong length for vocab_size=2
            num_docs: 10,
            vocab_size: 2,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_catches_non_monotonic_indptr() {
        let m = CscMatrix {
            data: vec![1.0, 2.0],
            indices: vec![0, 0],
            indptr: vec![0, 2, 1], // Non-monotonic
            num_docs: 10,
            vocab_size: 2,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_catches_out_of_range_index() {
        let m = CscMatrix {
            data: vec![1.0],
            indices: vec![99], // 99 >= num_docs (5)
            indptr: vec![0, 1],
            num_docs: 5,
            vocab_size: 1,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_catches_unsorted_indices() {
        let m = CscMatrix {
            data: vec![1.0, 2.0],
            indices: vec![3, 1], // Unsorted within column
            indptr: vec![0, 2],
            num_docs: 5,
            vocab_size: 1,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_catches_duplicate_indices() {
        let m = CscMatrix {
            data: vec![1.0, 2.0],
            indices: vec![2, 2], // Duplicate
            indptr: vec![0, 2],
            num_docs: 5,
            vocab_size: 1,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_catches_data_length_mismatch() {
        let m = CscMatrix {
            data: vec![1.0, 2.0, 3.0], // 3 entries but indptr says 2
            indices: vec![0, 1, 2],
            indptr: vec![0, 2],
            num_docs: 5,
            vocab_size: 1,
        };
        assert!(m.validate().is_err());
    }

    /// Verify from_triplets rejects out-of-range term_id.
    #[test]
    fn from_triplets_rejects_bad_term_id() {
        let triplets = vec![(5u32, 0u32, 1.0f32)]; // term_id 5 >= vocab_size 3
        let result = CscMatrix::from_triplets(&triplets, 10, 3);
        assert!(result.is_err());
    }

    /// Verify from_triplets rejects out-of-range doc_id.
    #[test]
    fn from_triplets_rejects_bad_doc_id() {
        let triplets = vec![(0u32, 20u32, 1.0f32)]; // doc_id 20 >= num_docs 10
        let result = CscMatrix::from_triplets(&triplets, 10, 3);
        assert!(result.is_err());
    }

    /// Single-entry matrix: one triplet produces a valid CSC with nnz == 1.
    #[test]
    fn single_entry_matrix() {
        let triplets = vec![(0u32, 0u32, 42.0f32)];
        let m = CscMatrix::from_triplets(&triplets, 1, 1).unwrap();
        assert!(m.validate().is_ok());
        assert_eq!(m.nnz(), 1);
        let (scores, doc_ids) = m.column(0);
        assert_eq!(doc_ids, &[0]);
        assert!((scores[0] - 42.0).abs() < f32::EPSILON);
    }

    /// CscHeader bytemuck Pod+Zeroable: cast to bytes and back round-trips.
    #[cfg(feature = "persistence")]
    #[test]
    fn header_bytemuck_cast_round_trip() {
        let header = CscHeader {
            magic: CscHeader::MAGIC,
            version: CscHeader::VERSION,
            flags: 0x02,
            num_docs: 1_000_000,
            num_terms: 50_000,
            nnz: 123_456_789,
            checksum: 0xDEAD_BEEF_CAFE_BABE,
            reserved: [0u8; 16],
        };

        // Cast to bytes and back using bytemuck.
        let bytes: &[u8] = bytemuck::bytes_of(&header);
        assert_eq!(bytes.len(), 64);
        let restored: &CscHeader = bytemuck::from_bytes(bytes);
        assert_eq!(restored.magic, CscHeader::MAGIC);
        assert_eq!(restored.version, CscHeader::VERSION);
        assert_eq!(restored.flags, 0x02);
        assert_eq!(restored.num_docs, 1_000_000);
        assert_eq!(restored.num_terms, 50_000);
        assert_eq!(restored.nnz, 123_456_789);
        assert_eq!(restored.checksum, 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(restored.reserved, [0u8; 16]);
    }

    /// Counting sort verification: from_triplets uses O(n) counting sort,
    /// not a comparison sort on the full triplet array. We verify this
    /// indirectly by checking that the output is correct for a large input
    /// with many columns, and that indptr boundaries are consistent --
    /// which is only possible if the counting-sort scatter was correct.
    #[test]
    fn counting_sort_produces_correct_column_boundaries() {
        // Create triplets spread across many columns in reverse order.
        // If a comparison sort were used on the full array, the order would
        // change; counting sort scatters by column directly.
        let num_cols = 200u32;
        let num_docs = 100u32;
        let mut triplets = Vec::new();
        // Insert in reverse column order to stress the scatter.
        for col in (0..num_cols).rev() {
            for doc in 0..3 {
                triplets.push((col, doc, (col * 10 + doc) as f32));
            }
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, num_cols).unwrap();
        assert!(m.validate().is_ok());
        assert_eq!(m.nnz(), (num_cols * 3) as usize);

        // Each column should have exactly 3 entries.
        for col in 0..num_cols {
            let (scores, doc_ids) = m.column(col);
            assert_eq!(doc_ids.len(), 3, "column {} should have 3 entries", col);
            // doc_ids should be sorted: [0, 1, 2].
            assert_eq!(doc_ids, &[0, 1, 2]);
            // Scores should match the formula.
            for doc in 0..3u32 {
                let expected = (col * 10 + doc) as f32;
                assert!(
                    (scores[doc as usize] - expected).abs() < f32::EPSILON,
                    "score mismatch at col={}, doc={}", col, doc
                );
            }
        }

        // Verify indptr is a strict staircase: each step is exactly 3.
        for i in 0..num_cols as usize {
            assert_eq!(
                m.indptr[i + 1] - m.indptr[i], 3,
                "indptr step at column {} should be 3", i
            );
        }
    }

    /// Multiple duplicates across different columns are all merged correctly.
    #[test]
    fn multiple_duplicates_across_columns() {
        let triplets = vec![
            (0u32, 0u32, 1.0f32),
            (0, 0, 2.0),
            (0, 0, 3.0), // triple duplicate in col 0
            (1, 5, 10.0),
            (1, 5, 20.0), // double duplicate in col 1
            (1, 3, 7.0),  // unique in col 1
        ];
        let m = CscMatrix::from_triplets(&triplets, 10, 2).unwrap();
        assert!(m.validate().is_ok());
        // Col 0: one merged entry (doc 0, score 6.0).
        assert_eq!(m.nnz(), 3); // 1 (col 0) + 2 (col 1)
        let (scores, doc_ids) = m.column(0);
        assert_eq!(doc_ids, &[0]);
        assert!((scores[0] - 6.0).abs() < f32::EPSILON);
        // Col 1: two entries (doc 3 -> 7.0, doc 5 -> 30.0).
        let (scores, doc_ids) = m.column(1);
        assert_eq!(doc_ids, &[3, 5]);
        assert!((scores[0] - 7.0).abs() < f32::EPSILON);
        assert!((scores[1] - 30.0).abs() < f32::EPSILON);
    }
}
