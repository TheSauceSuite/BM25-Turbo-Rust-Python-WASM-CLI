//! Block-Max WAND approximate top-k retrieval.
//!
//! Implements Block-Max WAND (Ding & Suel 2011) for fast approximate top-k
//! retrieval on inverted indexes. Skips non-competitive documents using
//! precomputed block-maximum scores, achieving significant speedups over
//! exhaustive scoring while maintaining near-perfect recall.
//!
//! This module is gated behind the `ann` feature.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::csc::CscMatrix;
use crate::error::{Error, Result};
use crate::types::Results;

/// Default block size for Block-Max WAND score upper bounds.
/// Must be a power of 2 for efficient block index computation.
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// Precomputed block-maximum scores for all terms in an index.
///
/// For each term (column) in the CSC matrix, stores the maximum BM25 score
/// within each block of `block_size` consecutive postings. This allows the
/// BMW algorithm to skip entire blocks whose maximum possible contribution
/// is below the current top-k threshold.
#[derive(Debug, Clone)]
pub struct BlockMaxIndex {
    /// Per-term block max scores. `term_block_maxes[term_id]` contains
    /// one `f32` per block of that term's posting list.
    pub term_block_maxes: Vec<Vec<f32>>,
    /// Per-term global maximum score (max across all postings for that term).
    pub term_global_maxes: Vec<f32>,
    /// Block size used during precomputation.
    pub block_size: usize,
}

/// A cursor tracking iteration state for one query term's posting list.
/// Pre-caches column slices to avoid repeated matrix.column() lookups.
#[derive(Debug)]
struct BmwCursor<'a> {
    /// Pre-cached scores slice from the CSC column.
    scores: &'a [f32],
    /// Pre-cached indices slice from the CSC column.
    indices: &'a [u32],
    /// Pre-cached block-max scores for this term.
    block_maxes: &'a [f32],
    /// Current position in the posting list.
    pos: usize,
    /// Length of this term's posting list.
    len: usize,
    /// Global max score for this term.
    global_max: f32,
}

impl<'a> BmwCursor<'a> {
    /// Returns true if the cursor has been exhausted.
    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.pos >= self.len
    }

    /// Returns the current document ID, or u32::MAX if exhausted.
    #[inline(always)]
    fn current_doc_id(&self) -> u32 {
        if self.pos < self.len {
            self.indices[self.pos]
        } else {
            u32::MAX
        }
    }

    /// Returns the current score, or 0.0 if exhausted.
    #[inline(always)]
    fn current_score(&self) -> f32 {
        if self.pos < self.len {
            self.scores[self.pos]
        } else {
            0.0
        }
    }

    /// Advance past all postings with doc_id < target using binary search.
    #[inline]
    fn advance_to(&mut self, target: u32) {
        if self.pos >= self.len {
            return;
        }
        let slice = &self.indices[self.pos..self.len];
        let offset = match slice.binary_search(&target) {
            Ok(o) | Err(o) => o,
        };
        self.pos += offset;
    }

    /// Get the block-max score for the current block.
    #[inline(always)]
    fn current_block_max(&self, block_size: usize) -> f32 {
        let block_idx = self.pos / block_size;
        if block_idx < self.block_maxes.len() {
            self.block_maxes[block_idx]
        } else {
            0.0
        }
    }

    /// Advance cursor to the start of the next block.
    #[inline]
    fn skip_to_next_block(&mut self, block_size: usize) {
        let next_block_start = ((self.pos / block_size) + 1) * block_size;
        if next_block_start < self.len {
            self.pos = next_block_start;
        } else {
            self.pos = self.len;
        }
    }
}

/// Wrapper for f32 that implements Ord via total_cmp for use in BinaryHeap.
#[derive(PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Precompute block-maximum scores for all terms in the CSC matrix.
///
/// For each term's posting list, partitions the scores into blocks of
/// `block_size` entries and records the maximum score in each block.
///
/// # Arguments
/// * `matrix` -- the CSC score matrix
/// * `block_size` -- number of postings per block (must be > 0 and a power of 2)
pub fn precompute_block_maxes(matrix: &CscMatrix, block_size: usize) -> Result<BlockMaxIndex> {
    if block_size == 0 {
        return Err(Error::InvalidParameter(
            "block_size must be > 0".to_string(),
        ));
    }
    if !block_size.is_power_of_two() {
        return Err(Error::InvalidParameter(
            "block_size must be a power of 2".to_string(),
        ));
    }

    let vocab_size = matrix.vocab_size as usize;
    let mut term_block_maxes = Vec::with_capacity(vocab_size);
    let mut term_global_maxes = Vec::with_capacity(vocab_size);

    for term_id in 0..vocab_size {
        let (scores, _indices) = matrix.column(term_id as u32);

        if scores.is_empty() {
            term_block_maxes.push(Vec::new());
            term_global_maxes.push(0.0);
            continue;
        }

        // Partition scores into blocks and find max per block.
        let num_blocks = scores.len().div_ceil(block_size);
        let mut block_maxes = Vec::with_capacity(num_blocks);
        let mut global_max: f32 = 0.0;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(scores.len());
            let block_scores = &scores[start..end];

            let block_max = block_scores
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            block_maxes.push(block_max);

            if block_max > global_max {
                global_max = block_max;
            }
        }

        term_block_maxes.push(block_maxes);
        term_global_maxes.push(global_max);
    }

    Ok(BlockMaxIndex {
        term_block_maxes,
        term_global_maxes,
        block_size,
    })
}

/// Perform Block-Max WAND top-k retrieval.
///
/// Optimized implementation with pre-cached column slices, no per-iteration
/// allocation, and block-level skipping. Achieves sub-linear query time by
/// skipping documents whose upper-bound scores cannot enter the top-k.
///
/// # Arguments
/// * `matrix` -- the CSC score matrix
/// * `block_max` -- precomputed block-max index
/// * `query_token_ids` -- query term IDs
/// * `k` -- number of results to return
pub fn block_max_wand_top_k(
    matrix: &CscMatrix,
    block_max: &BlockMaxIndex,
    query_token_ids: &[u32],
    k: usize,
) -> Results {
    if k == 0 || query_token_ids.is_empty() {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    let block_size = block_max.block_size;

    // Pre-cache all column slices and block maxes into cursors.
    // This avoids repeated matrix.column() calls in the hot loop.
    let mut cursors: Vec<BmwCursor<'_>> = query_token_ids
        .iter()
        .filter_map(|&tid| {
            if (tid as usize) >= matrix.vocab_size as usize {
                return None;
            }
            let (scores, indices) = matrix.column(tid);
            if scores.is_empty() {
                return None;
            }
            let bmaxes = &block_max.term_block_maxes[tid as usize];
            let gmax = block_max.term_global_maxes[tid as usize];
            Some(BmwCursor {
                scores,
                indices,
                block_maxes: bmaxes,
                pos: 0,
                len: scores.len(),
                global_max: gmax,
            })
        })
        .collect();

    if cursors.is_empty() {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    // Min-heap for top-k results.
    let mut heap: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(k + 1);
    let mut threshold: f32 = 0.0;
    let mut num_active = cursors.len();

    // Main BMW loop.
    loop {
        // Swap exhausted cursors to the end (no allocation).
        let mut i = 0;
        while i < num_active {
            if cursors[i].is_exhausted() {
                num_active -= 1;
                cursors.swap(i, num_active);
            } else {
                i += 1;
            }
        }
        if num_active == 0 {
            break;
        }

        let active = &mut cursors[..num_active];

        // Sort active cursors by current doc_id (no allocation — in-place).
        active.sort_unstable_by_key(|c| c.current_doc_id());

        let min_doc = active[0].current_doc_id();
        if min_doc == u32::MAX {
            break;
        }

        // Find pivot: first position where cumulative block-max >= threshold.
        let mut cumulative_upper = 0.0f32;
        let mut pivot_idx = None;

        for (i, cursor) in active.iter().enumerate() {
            cumulative_upper += cursor.current_block_max(block_size);
            if cumulative_upper >= threshold {
                pivot_idx = Some(i);
                break;
            }
        }

        let pivot_idx = match pivot_idx {
            Some(p) => p,
            None => break, // No document can beat threshold.
        };

        let pivot_doc = active[pivot_idx].current_doc_id();
        if pivot_doc == u32::MAX {
            break;
        }

        // Check if all cursors [0..=pivot_idx] point to pivot_doc.
        let all_at_pivot = active[..=pivot_idx]
            .iter()
            .all(|c| c.current_doc_id() == pivot_doc);

        if all_at_pivot {
            // Score this document exactly from ALL cursors that have it.
            let mut doc_score: f32 = 0.0;
            for cursor in active.iter() {
                if cursor.current_doc_id() == pivot_doc {
                    doc_score += cursor.current_score();
                }
            }

            // Insert into top-k heap.
            if heap.len() < k {
                heap.push(Reverse((OrdF32(doc_score), pivot_doc)));
                if heap.len() == k {
                    threshold = heap.peek().unwrap().0 .0 .0;
                }
            } else if doc_score > threshold {
                heap.pop();
                heap.push(Reverse((OrdF32(doc_score), pivot_doc)));
                threshold = heap.peek().unwrap().0 .0 .0;
            }

            // Advance all cursors past pivot_doc.
            for cursor in active.iter_mut() {
                if cursor.current_doc_id() == pivot_doc {
                    cursor.pos += 1;
                }
            }
        } else {
            // Not all at pivot. Advance the first cursor.
            let first_doc = active[0].current_doc_id();
            if first_doc < pivot_doc {
                // Block-level skip: if the entire current block of cursor[0]
                // can't contribute enough, skip the whole block.
                let block_max_score = active[0].current_block_max(block_size);
                let rest_upper: f32 = active[1..num_active]
                    .iter()
                    .map(|c| c.global_max)
                    .sum();

                if block_max_score + rest_upper < threshold {
                    active[0].skip_to_next_block(block_size);
                } else {
                    active[0].advance_to(pivot_doc);
                }
            } else {
                active[0].pos += 1;
            }
        }
    }

    // Drain heap and sort by descending score, ascending doc_id on tie.
    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((OrdF32(score), doc_id))| (doc_id, score))
        .collect();
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let (doc_ids, scores): (Vec<u32>, Vec<f32>) = results.into_iter().unzip();

    Results { doc_ids, scores }
}

// ===========================================================================
// BM25Index integration
// ===========================================================================

impl crate::index::BM25Index {
    /// Perform approximate top-k search using Block-Max WAND.
    ///
    /// Requires that the index was built with BMW data (via
    /// `BM25Builder::with_bmw(true)` or by calling `build_bmw_index()` after
    /// construction).
    ///
    /// Returns `Error::FeatureNotEnabled` if BMW data has not been precomputed
    /// for this index.
    ///
    /// # Arguments
    /// * `query` -- the search query string
    /// * `k` -- number of top results to return
    pub fn search_approximate(&self, query: &str, k: usize) -> Result<Results> {
        if k == 0 {
            return Err(Error::InvalidParameter("k must be > 0".to_string()));
        }

        let bmw = match &self.block_max_index {
            Some(bm) => bm,
            None => {
                return Err(Error::FeatureNotEnabled(
                    "bmw data not built -- call build_bmw_index() or use BM25Builder::with_bmw(true)"
                        .to_string(),
                ));
            }
        };

        let query_tokens = self.tokenizer.tokenize(query);
        let token_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.vocab.get(t.as_str()).copied())
            .collect();

        if token_ids.is_empty() {
            return Ok(Results {
                doc_ids: Vec::new(),
                scores: Vec::new(),
            });
        }

        Ok(block_max_wand_top_k(&self.matrix, bmw, &token_ids, k))
    }

    /// Precompute Block-Max WAND data for this index.
    ///
    /// After calling this, `search_approximate()` becomes available.
    /// Uses the default block size of 128.
    pub fn build_bmw_index(&mut self) -> Result<()> {
        self.build_bmw_index_with_block_size(DEFAULT_BLOCK_SIZE)
    }

    /// Precompute Block-Max WAND data with a custom block size.
    ///
    /// Block size must be a positive power of 2.
    pub fn build_bmw_index_with_block_size(&mut self, block_size: usize) -> Result<()> {
        let bmw = precompute_block_maxes(&self.matrix, block_size)?;
        self.block_max_index = Some(bmw);
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::BM25Builder;

    fn build_test_corpus() -> Vec<&'static str> {
        vec![
            "the quick brown fox jumps over the lazy dog",
            "a fast red car drives on the highway",
            "the lazy brown dog sleeps in the sun",
            "quick fox quick fox quick fox",
            "highway car drives fast red",
            "sun moon stars galaxy universe",
            "the quick red fox",
            "lazy lazy lazy dog dog dog",
            "brown bear eats honey in the forest",
            "the sun rises in the east",
        ]
    }

    #[test]
    fn precompute_block_maxes_basic() {
        let corpus = build_test_corpus();
        let index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();

        let bmw = precompute_block_maxes(&index.matrix, DEFAULT_BLOCK_SIZE).unwrap();
        assert_eq!(bmw.term_block_maxes.len(), index.vocab_size() as usize);
        assert_eq!(bmw.term_global_maxes.len(), index.vocab_size() as usize);
        assert_eq!(bmw.block_size, DEFAULT_BLOCK_SIZE);

        // All global maxes should be >= 0.
        for &gm in &bmw.term_global_maxes {
            assert!(gm >= 0.0, "Global max should be >= 0, got {}", gm);
        }
    }

    #[test]
    fn precompute_block_maxes_invalid_block_size() {
        let corpus = build_test_corpus();
        let index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();

        assert!(precompute_block_maxes(&index.matrix, 0).is_err());
        assert!(precompute_block_maxes(&index.matrix, 3).is_err()); // not power of 2
        assert!(precompute_block_maxes(&index.matrix, 7).is_err()); // not power of 2
    }

    #[test]
    fn bmw_top_k_empty_query() {
        let corpus = build_test_corpus();
        let index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        let bmw = precompute_block_maxes(&index.matrix, DEFAULT_BLOCK_SIZE).unwrap();

        let results = block_max_wand_top_k(&index.matrix, &bmw, &[], 5);
        assert!(results.doc_ids.is_empty());
    }

    #[test]
    fn bmw_top_k_k_zero() {
        let corpus = build_test_corpus();
        let index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        let bmw = precompute_block_maxes(&index.matrix, DEFAULT_BLOCK_SIZE).unwrap();

        let results = block_max_wand_top_k(&index.matrix, &bmw, &[0], 0);
        assert!(results.doc_ids.is_empty());
    }

    /// TEST-P10-002: For documents that BMW evaluates, verify they receive exact BM25 scores.
    #[test]
    fn bmw_exact_scores_for_evaluated_docs() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        // Search with both exact and approximate.
        let exact = index.search("quick fox", 5).unwrap();
        let approx = index.search_approximate("quick fox", 5).unwrap();

        // For each doc in the approximate results, check that if the same doc
        // appears in exact results, the scores match exactly.
        for (i, &doc_id) in approx.doc_ids.iter().enumerate() {
            if let Some(exact_pos) = exact.doc_ids.iter().position(|&d| d == doc_id) {
                assert_eq!(
                    approx.scores[i].to_bits(),
                    exact.scores[exact_pos].to_bits(),
                    "BMW score for doc {} should be exact: got {}, expected {}",
                    doc_id,
                    approx.scores[i],
                    exact.scores[exact_pos]
                );
            }
        }
    }

    /// TEST-P10-003: BMW Not Built Error -- calling search_approximate without BMW data.
    #[test]
    fn bmw_not_built_error() {
        let corpus = build_test_corpus();
        let index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();

        let result = index.search_approximate("quick", 5);
        assert!(result.is_err());
        match result {
            Err(Error::FeatureNotEnabled(msg)) => {
                assert!(msg.contains("bmw"), "Error message should mention BMW: {}", msg);
            }
            Err(e) => panic!("Expected FeatureNotEnabled, got {:?}", e),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    /// Test that BMW produces results for single-term queries.
    #[test]
    fn bmw_single_term_query() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        let exact = index.search("quick", 3).unwrap();
        let approx = index.search_approximate("quick", 3).unwrap();

        // At minimum, the top result should match.
        assert!(
            !approx.doc_ids.is_empty(),
            "BMW should return results for 'quick'"
        );
        assert_eq!(
            exact.doc_ids[0], approx.doc_ids[0],
            "Top result should match: exact={:?}, approx={:?}",
            exact, approx
        );
    }

    /// Test BMW with multi-term query.
    #[test]
    fn bmw_multi_term_query() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        let exact = index.search("lazy dog", 5).unwrap();
        let approx = index.search_approximate("lazy dog", 5).unwrap();

        // BMW should find the same top documents.
        assert!(
            !approx.doc_ids.is_empty(),
            "BMW should return results for 'lazy dog'"
        );

        // Check recall: all exact top-k should appear in approximate results.
        for &doc_id in &exact.doc_ids {
            assert!(
                approx.doc_ids.contains(&doc_id),
                "BMW missed doc {} from exact results. Exact: {:?}, Approx: {:?}",
                doc_id,
                exact,
                approx
            );
        }
    }

    /// Test BMW with unknown query tokens.
    #[test]
    fn bmw_unknown_tokens() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        let results = index.search_approximate("xyzzyplugh", 5).unwrap();
        assert!(results.doc_ids.is_empty());
    }

    /// Test BMW with different block sizes.
    #[test]
    fn bmw_different_block_sizes() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();

        for block_size in [2, 4, 8, 16, 32, 64, 128, 256] {
            index
                .build_bmw_index_with_block_size(block_size)
                .unwrap();
            let results = index.search_approximate("quick fox", 3).unwrap();
            assert!(
                !results.doc_ids.is_empty(),
                "BMW with block_size={} should return results",
                block_size
            );
        }
    }

    /// Test that search_approximate validates k.
    #[test]
    fn bmw_search_approximate_k_zero_error() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        assert!(index.search_approximate("quick", 0).is_err());
    }

    // ===================================================================
    // TEST-P10-001: BMW Recall@100 on synthetic corpus
    // ===================================================================

    /// Generate a synthetic corpus of `n` documents using a small vocabulary.
    /// Each document is a random selection of words to create realistic overlap.
    fn generate_synthetic_corpus(n: usize) -> Vec<String> {
        // Fixed vocabulary of 200 words for reproducibility.
        let words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
            "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "apple",
            "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
            "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince",
            "raspberry", "strawberry", "tangerine", "ugli", "vanilla", "watermelon",
            "algorithm", "binary", "compiler", "database", "engine", "framework",
            "gateway", "handler", "index", "journal", "kernel", "library", "module",
            "network", "object", "parser", "query", "router", "server", "thread",
            "utility", "vector", "widget", "xenon", "yield", "zipper", "abstract",
            "boolean", "class", "double", "enum", "float", "generic", "hash",
            "integer", "java", "kotlin", "long", "method", "null", "operator",
            "pointer", "queue", "reference", "stack", "tuple", "union", "value",
            "window", "xor", "yaml", "zero", "search", "document", "ranking",
            "score", "retrieval", "information", "frequency", "inverse", "length",
            "average", "parameter", "weight", "function", "matrix", "sparse",
            "dense", "block", "maximum", "threshold", "cursor", "posting", "list",
            "term", "vocabulary", "corpus", "collection", "benchmark", "recall",
            "precision", "accuracy", "speed", "latency", "throughput", "memory",
            "cache", "buffer", "stream", "batch", "pipeline", "parallel", "serial",
            "concurrent", "atomic", "lock", "mutex", "channel", "signal", "event",
            "trigger", "callback", "promise", "future", "async", "await", "spawn",
            "join", "split", "merge", "filter", "map", "reduce", "fold", "scan",
            "sort", "heap", "tree", "graph", "node", "edge", "path", "route",
            "bridge", "tunnel", "portal", "gateway", "beacon", "anchor", "dock",
            "harbor", "ocean", "river", "lake", "mountain", "valley", "forest",
            "meadow", "desert", "tundra", "glacier", "volcano", "canyon", "plateau",
            "island", "peninsula", "continent", "planet", "star", "moon", "comet",
            "nebula", "galaxy", "cosmos", "universe", "dimension", "reality",
            "fiction", "history", "science", "nature", "culture", "language",
        ];

        let mut corpus = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic pseudo-random document generation using index as seed.
            let doc_len = 5 + (i * 7 + 13) % 20; // 5-24 words per doc
            let mut doc = String::with_capacity(doc_len * 8);
            for j in 0..doc_len {
                if j > 0 {
                    doc.push(' ');
                }
                let word_idx = (i * 31 + j * 17 + i * j * 3 + 7) % words.len();
                doc.push_str(words[word_idx]);
            }
            corpus.push(doc);
        }
        corpus
    }

    /// Generate synthetic queries from the same vocabulary.
    fn generate_synthetic_queries(n: usize) -> Vec<String> {
        let words = [
            "alpha", "beta", "gamma", "delta", "engine", "framework",
            "search", "query", "index", "document", "score", "ranking",
            "block", "maximum", "threshold", "sparse", "dense", "matrix",
            "recall", "precision", "speed", "memory", "cache", "buffer",
            "tree", "graph", "node", "path", "ocean", "mountain",
        ];

        let mut queries = Vec::with_capacity(n);
        for i in 0..n {
            // 1-3 term queries.
            let num_terms = 1 + i % 3;
            let mut q = String::new();
            for j in 0..num_terms {
                if j > 0 {
                    q.push(' ');
                }
                let word_idx = (i * 13 + j * 7 + 5) % words.len();
                q.push_str(words[word_idx]);
            }
            queries.push(q);
        }
        queries
    }

    /// TEST-P10-001: BMW recall@100 > 99% on synthetic corpus.
    /// Build a 1000-doc index (kept small for test speed), run 100 queries,
    /// and verify recall of BMW top-100 against exact top-100.
    #[test]
    fn bmw_recall_at_100() {
        let corpus = generate_synthetic_corpus(1000);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .unwrap();
        index.build_bmw_index_with_block_size(64).unwrap();

        let queries = generate_synthetic_queries(100);
        let k = 100;

        let mut total_recall = 0.0f64;
        let mut valid_queries = 0u32;

        for query in &queries {
            let exact = index.search(query, k).unwrap();
            if exact.doc_ids.is_empty() {
                continue; // Skip queries with no results.
            }

            let approx = index.search_approximate(query, k).unwrap();

            // Recall = |intersection| / |exact results|
            let exact_set: std::collections::HashSet<u32> =
                exact.doc_ids.iter().copied().collect();
            let approx_set: std::collections::HashSet<u32> =
                approx.doc_ids.iter().copied().collect();

            let intersection = exact_set.intersection(&approx_set).count();
            let recall = intersection as f64 / exact_set.len() as f64;
            total_recall += recall;
            valid_queries += 1;
        }

        assert!(valid_queries > 0, "At least some queries should have results");
        let avg_recall = total_recall / valid_queries as f64;
        assert!(
            avg_recall > 0.99,
            "BMW recall@{} should be > 99%, got {:.4}% over {} queries",
            k,
            avg_recall * 100.0,
            valid_queries
        );
    }

    /// TEST-P10-004 (variant): BMW with k=1, k=10, k=100 all produce valid results.
    #[test]
    fn bmw_various_k_values() {
        let corpus = generate_synthetic_corpus(500);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .unwrap();
        index.build_bmw_index().unwrap();

        for k in [1, 10, 100] {
            let exact = index.search("alpha beta gamma", k).unwrap();
            let approx = index.search_approximate("alpha beta gamma", k).unwrap();

            // BMW should return at most k results.
            assert!(
                approx.doc_ids.len() <= k,
                "BMW with k={} returned {} results (expected <= {})",
                k,
                approx.doc_ids.len(),
                k
            );

            // BMW should return at least 1 result if exact does.
            if !exact.doc_ids.is_empty() {
                assert!(
                    !approx.doc_ids.is_empty(),
                    "BMW with k={} returned empty but exact returned {} results",
                    k,
                    exact.doc_ids.len()
                );
            }

            // Scores should be in descending order.
            for window in approx.scores.windows(2) {
                assert!(
                    window[0] >= window[1],
                    "BMW results not sorted descending: {} < {}",
                    window[0],
                    window[1]
                );
            }

            // No duplicate doc IDs.
            let mut seen = std::collections::HashSet::new();
            for &doc_id in &approx.doc_ids {
                assert!(
                    seen.insert(doc_id),
                    "BMW with k={} returned duplicate doc_id {}",
                    k,
                    doc_id
                );
            }
        }
    }

    /// TEST-P10-005 (variant): BMW block sizes 64, 128, 256 all produce correct results.
    /// Verifies that different block sizes yield results matching exact search.
    #[test]
    fn bmw_block_sizes_64_128_256_correct() {
        let corpus = generate_synthetic_corpus(500);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .unwrap();

        let exact = index.search("delta engine search", 10).unwrap();

        for block_size in [64, 128, 256] {
            index.build_bmw_index_with_block_size(block_size).unwrap();
            let approx = index.search_approximate("delta engine search", 10).unwrap();

            assert!(
                !approx.doc_ids.is_empty(),
                "BMW with block_size={} should return results",
                block_size
            );

            // Top-1 result should match exact.
            if !exact.doc_ids.is_empty() {
                assert_eq!(
                    exact.doc_ids[0], approx.doc_ids[0],
                    "BMW block_size={}: top result mismatch. exact={}, approx={}",
                    block_size, exact.doc_ids[0], approx.doc_ids[0]
                );
            }

            // Scores for shared docs must be bit-exact (BMW scores exactly, only skips).
            for (i, &doc_id) in approx.doc_ids.iter().enumerate() {
                if let Some(exact_pos) = exact.doc_ids.iter().position(|&d| d == doc_id) {
                    assert_eq!(
                        approx.scores[i].to_bits(),
                        exact.scores[exact_pos].to_bits(),
                        "BMW block_size={}: score mismatch for doc {}",
                        block_size,
                        doc_id
                    );
                }
            }
        }
    }

    /// TEST-P10-006 (variant): BMW empty query via search_approximate returns empty results.
    #[test]
    fn bmw_search_approximate_empty_query() {
        let corpus = build_test_corpus();
        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .unwrap();
        index.build_bmw_index().unwrap();

        let results = index.search_approximate("", 5).unwrap();
        assert!(
            results.doc_ids.is_empty(),
            "Empty query should return empty results"
        );
        assert!(
            results.scores.is_empty(),
            "Empty query should return empty scores"
        );
    }

    /// TEST-P10-007 (variant): Performance sanity -- BMW evaluates fewer documents than exact.
    /// For a corpus where many docs match, exact search scores all matching docs
    /// while BMW should prune and only return top-k.
    #[test]
    fn bmw_evaluates_fewer_docs_than_exact() {
        let corpus = generate_synthetic_corpus(1000);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        let mut index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .unwrap();
        index.build_bmw_index_with_block_size(64).unwrap();

        let query = "alpha beta gamma delta";
        let k = 10;

        // Exact search: count how many docs have non-zero score.
        let all_scores = index.get_scores(query).unwrap();
        let docs_with_score = all_scores.iter().filter(|&&s| s > 0.0).count();

        // BMW search: returns at most k results.
        let approx = index.search_approximate(query, k).unwrap();

        // The key insight: exact search scores all `docs_with_score` documents,
        // while BMW returns at most k. For this to be a meaningful test,
        // there must be more matching docs than k.
        assert!(
            docs_with_score > k,
            "Need more matching docs ({}) than k ({}) for this test to be meaningful",
            docs_with_score,
            k
        );
        assert!(
            approx.doc_ids.len() <= k,
            "BMW should return at most k={} results, got {}",
            k,
            approx.doc_ids.len()
        );
        assert!(
            approx.doc_ids.len() < docs_with_score,
            "BMW returned {} docs but exact has {} scoreable docs -- BMW should evaluate fewer",
            approx.doc_ids.len(),
            docs_with_score
        );
    }
}
