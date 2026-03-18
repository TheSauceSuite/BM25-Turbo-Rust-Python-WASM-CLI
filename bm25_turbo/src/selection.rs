//! Top-k selection algorithms.
//!
//! Provides efficient methods to find the k highest-scoring documents
//! from a score accumulator array:
//! - `argpartition`: O(n) partial sort using `select_nth_unstable`
//! - Min-heap: O(n log k) streaming selection for when k is small relative to n

use crate::types::Results;

/// Select the top-k documents by score using partial sort.
///
/// Returns document IDs and scores sorted by descending score.
///
/// # Arguments
/// * `scores` — score accumulator array indexed by document ID
/// * `k` — number of top results to return
pub fn top_k(scores: &[f32], k: usize) -> Results {
    let k = k.min(scores.len());
    if k == 0 {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    // Build (doc_id, score) pairs for non-zero scores
    let mut candidates: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .filter(|&(_, s)| *s > 0.0)
        .map(|(i, s)| (i as u32, *s))
        .collect();

    if candidates.is_empty() {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    let k = k.min(candidates.len());

    // Partial sort: move the top-k elements to the front.
    // Tie-break by ascending doc_id for deterministic ordering.
    candidates.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Sort only the top-k by descending score, ascending doc_id on tie.
    candidates.truncate(k);
    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let (doc_ids, result_scores): (Vec<u32>, Vec<f32>) = candidates.into_iter().unzip();

    Results {
        doc_ids,
        scores: result_scores,
    }
}

/// Select top-k using a min-heap. Better when k << n.
///
/// Uses a `BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>>` min-heap of size k.
/// Time complexity: O(n log k). Preferred over `top_k` when k << n.
pub fn top_k_heap(scores: &[f32], k: usize) -> Results {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let k = k.min(scores.len());
    if k == 0 {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    // We wrap f32 in a newtype that implements Ord via total_cmp.
    // This avoids needing the ordered-float crate.
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

    // Min-heap: Reverse makes the smallest element pop first.
    let mut heap: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(k + 1);

    for (doc_id, &score) in scores.iter().enumerate() {
        if score <= 0.0 {
            continue;
        }
        if heap.len() < k {
            heap.push(Reverse((OrdF32(score), doc_id as u32)));
        } else if let Some(&Reverse((OrdF32(min_score), max_doc_id))) = heap.peek() {
            // On tie: prefer the lower doc_id (deterministic tie-breaking).
            if score > min_score || (score == min_score && (doc_id as u32) < max_doc_id) {
                heap.pop();
                heap.push(Reverse((OrdF32(score), doc_id as u32)));
            }
        }
    }

    // Drain the heap and sort by descending score.
    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((OrdF32(score), doc_id))| (doc_id, score))
        .collect();
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let (doc_ids, result_scores): (Vec<u32>, Vec<f32>) = results.into_iter().unzip();

    Results {
        doc_ids,
        scores: result_scores,
    }
}

/// SIMD-accelerated top-k selection using non-zero candidate scan.
///
/// Uses SIMD comparison against 0.0 to efficiently identify non-zero score
/// entries before applying partial sort. This replaces the scalar
/// `.filter(|s| *s > 0.0)` bottleneck when the score array is sparse.
///
/// Results are bit-identical to `top_k()`.
#[cfg(feature = "simd")]
pub fn top_k_simd(scores: &[f32], k: usize) -> Results {
    use pulp::Arch;

    let k = k.min(scores.len());
    if k == 0 {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    // SIMD non-zero scan: compare chunks of floats against 0.0 to build
    // a candidate buffer of (doc_id, score) pairs.
    let arch = Arch::new();
    let candidates: Vec<(u32, f32)> = arch.dispatch(NonZeroScan { scores });

    if candidates.is_empty() {
        return Results {
            doc_ids: Vec::new(),
            scores: Vec::new(),
        };
    }

    let k = k.min(candidates.len());
    let mut candidates = candidates;

    // Partial sort: move the top-k elements to the front.
    // Tie-break by ascending doc_id for deterministic ordering.
    candidates.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Sort only the top-k by descending score, ascending doc_id on tie.
    candidates.truncate(k);
    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let (doc_ids, result_scores): (Vec<u32>, Vec<f32>) = candidates.into_iter().unzip();

    Results {
        doc_ids,
        scores: result_scores,
    }
}

/// WithSimd implementation for non-zero score scanning.
///
/// Uses SIMD `max_f32s` to quickly skip all-zero chunks, then scans
/// individual lanes only when a chunk contains at least one positive value.
#[cfg(feature = "simd")]
struct NonZeroScan<'a> {
    scores: &'a [f32],
}

#[cfg(feature = "simd")]
impl pulp::WithSimd for NonZeroScan<'_> {
    type Output = Vec<(u32, f32)>;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let scores = self.scores;
        let mut candidates = Vec::new();

        let zero = simd.splat_f32s(0.0);
        let (head, tail) = S::as_simd_f32s(scores);

        // Process SIMD-aligned chunks.
        let lane_count = std::mem::size_of::<S::f32s>() / std::mem::size_of::<f32>();
        for (chunk_idx, &chunk) in head.iter().enumerate() {
            let base_idx = chunk_idx * lane_count;
            // Use SIMD max to check if the chunk has any positive values.
            // If the max of the chunk is <= 0.0, skip the entire chunk.
            let chunk_max = simd.reduce_max_f32s(simd.max_f32s(chunk, zero));
            if chunk_max > 0.0 {
                // At least one lane is positive -- scan individual elements.
                // Access the original scores array directly (the SIMD chunks
                // are aligned views into it).
                for lane in 0..lane_count {
                    let idx = base_idx + lane;
                    if idx < scores.len() {
                        let s = scores[idx];
                        if s > 0.0 {
                            candidates.push((idx as u32, s));
                        }
                    }
                }
            }
        }

        // Scalar tail.
        let tail_start = head.len() * lane_count;
        for (i, &s) in tail.iter().enumerate() {
            if s > 0.0 {
                candidates.push(((tail_start + i) as u32, s));
            }
        }

        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_basic() {
        let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let results = top_k(&scores, 2);
        assert_eq!(results.doc_ids, vec![3, 1]);
        assert_eq!(results.scores, vec![0.9, 0.5]);
    }

    #[test]
    fn top_k_empty() {
        let scores: Vec<f32> = vec![0.0; 10];
        let results = top_k(&scores, 5);
        assert!(results.doc_ids.is_empty());
    }

    #[test]
    fn top_k_larger_than_candidates() {
        let scores = vec![0.1, 0.5];
        let results = top_k(&scores, 10);
        assert_eq!(results.doc_ids.len(), 2);
    }

    // ---------------------------------------------------------------
    // TEST-P3-006: top_k_heap matches top_k for various k values.
    // ---------------------------------------------------------------

    #[test]
    fn top_k_heap_basic() {
        let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let results = top_k_heap(&scores, 2);
        assert_eq!(results.doc_ids, vec![3, 1]);
        assert_eq!(results.scores, vec![0.9, 0.5]);
    }

    #[test]
    fn top_k_heap_empty_scores() {
        let scores: Vec<f32> = vec![0.0; 10];
        let results = top_k_heap(&scores, 5);
        assert!(results.doc_ids.is_empty());
    }

    #[test]
    fn top_k_heap_larger_than_candidates() {
        let scores = vec![0.1, 0.5];
        let results = top_k_heap(&scores, 10);
        assert_eq!(results.doc_ids.len(), 2);
    }

    /// TEST-P3-006: top_k_heap and top_k produce identical results for k=1,5,10,100.
    #[test]
    fn top_k_heap_matches_top_k() {
        // Build a score array with varied values.
        let scores: Vec<f32> = (0..200)
            .map(|i| {
                // Deterministic pseudo-random-ish scores.
                let x = ((i as u64).wrapping_mul(2654435761) % 10000) as f32 / 10000.0;
                if i % 7 == 0 { 0.0 } else { x }
            })
            .collect();

        for k in [1, 5, 10, 100] {
            let r1 = top_k(&scores, k);
            let r2 = top_k_heap(&scores, k);
            assert_eq!(
                r1.doc_ids, r2.doc_ids,
                "doc_ids mismatch for k={}: top_k={:?}, top_k_heap={:?}",
                k, r1.doc_ids, r2.doc_ids
            );
            assert_eq!(
                r1.scores.len(), r2.scores.len(),
                "score count mismatch for k={}",
                k
            );
            for (j, (s1, s2)) in r1.scores.iter().zip(r2.scores.iter()).enumerate() {
                assert_eq!(
                    s1.to_bits(), s2.to_bits(),
                    "score mismatch for k={} at position {}: {} vs {}",
                    k, j, s1, s2
                );
            }
        }
    }

    /// k=0 with top_k returns empty (not an error, just empty).
    #[test]
    fn top_k_k_zero_returns_empty() {
        let scores = vec![1.0, 2.0, 3.0];
        let results = top_k(&scores, 0);
        assert!(results.doc_ids.is_empty());
    }

    /// k=0 with top_k_heap returns empty.
    #[test]
    fn top_k_heap_k_zero_returns_empty() {
        let scores = vec![1.0, 2.0, 3.0];
        let results = top_k_heap(&scores, 0);
        assert!(results.doc_ids.is_empty());
    }

    /// k=1 returns exactly the maximum scoring document.
    #[test]
    fn top_k_k_one() {
        let scores = vec![0.1, 0.9, 0.5, 0.3];
        let r1 = top_k(&scores, 1);
        let r2 = top_k_heap(&scores, 1);
        assert_eq!(r1.doc_ids, vec![1]);
        assert_eq!(r2.doc_ids, vec![1]);
    }

    /// k > n (where n = number of non-zero scores) returns all non-zero entries.
    #[test]
    fn top_k_k_greater_than_n() {
        let scores = vec![0.0, 0.5, 0.0, 0.3, 0.0];
        let r1 = top_k(&scores, 100);
        let r2 = top_k_heap(&scores, 100);
        assert_eq!(r1.doc_ids.len(), 2);
        assert_eq!(r2.doc_ids.len(), 2);
        assert_eq!(r1.doc_ids, r2.doc_ids);
    }

    /// Large score array: both methods agree.
    #[test]
    fn top_k_heap_matches_top_k_large() {
        let n = 10_000;
        let scores: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 40) as f32 / 16777216.0;
                if i % 3 == 0 { 0.0 } else { x }
            })
            .collect();

        for k in [1, 10, 50, 500] {
            let r1 = top_k(&scores, k);
            let r2 = top_k_heap(&scores, k);
            assert_eq!(r1.doc_ids, r2.doc_ids, "doc_ids mismatch for large array k={}", k);
        }
    }

    // ---------------------------------------------------------------
    // TEST-P1-006: SIMD top_k matches scalar top_k.
    // ---------------------------------------------------------------

    /// TEST-P1-006: top_k_simd produces bit-identical results to top_k.
    #[cfg(feature = "simd")]
    #[test]
    fn top_k_simd_matches_top_k() {
        // Test with various array sizes and k values.
        for n in [200, 10_000, 100_000] {
            let scores: Vec<f32> = (0..n)
                .map(|i| {
                    let x = ((i as u64).wrapping_mul(2654435761) % 10000) as f32 / 10000.0;
                    if i % 7 == 0 { 0.0 } else { x }
                })
                .collect();

            for k in [1, 5, 10, 100] {
                let r_scalar = top_k(&scores, k);
                let r_simd = super::top_k_simd(&scores, k);
                assert_eq!(
                    r_scalar.doc_ids, r_simd.doc_ids,
                    "doc_ids mismatch for n={}, k={}: scalar={:?}, simd={:?}",
                    n, k, r_scalar.doc_ids, r_simd.doc_ids
                );
                assert_eq!(
                    r_scalar.scores.len(), r_simd.scores.len(),
                    "score count mismatch for n={}, k={}",
                    n, k
                );
                for (j, (s1, s2)) in r_scalar.scores.iter().zip(r_simd.scores.iter()).enumerate() {
                    assert_eq!(
                        s1.to_bits(), s2.to_bits(),
                        "score mismatch for n={}, k={} at position {}: {} vs {}",
                        n, k, j, s1, s2
                    );
                }
            }
        }
    }

    /// SIMD top_k with empty scores.
    #[cfg(feature = "simd")]
    #[test]
    fn top_k_simd_empty() {
        let scores: Vec<f32> = vec![0.0; 10];
        let results = super::top_k_simd(&scores, 5);
        assert!(results.doc_ids.is_empty());
    }

    /// SIMD top_k with k=0.
    #[cfg(feature = "simd")]
    #[test]
    fn top_k_simd_k_zero() {
        let scores = vec![1.0, 2.0, 3.0];
        let results = super::top_k_simd(&scores, 0);
        assert!(results.doc_ids.is_empty());
    }
}
