//! Property-based tests for BM25 scoring and CSC matrix correctness.

use proptest::prelude::*;
use bm25_turbo::csc::CscMatrix;

proptest! {
    /// IDF should be positive for all valid inputs where df < N.
    #[test]
    fn idf_positive_for_valid_inputs(
        num_docs in 2u32..1_000_000,
        df_ratio in 0.001f64..0.999,
    ) {
        let doc_freq = ((num_docs as f64 * df_ratio) as u32).max(1);
        let doc_freq = doc_freq.min(num_docs - 1);

        let methods = [
            bm25_turbo::Method::Lucene,
            bm25_turbo::Method::Atire,
            bm25_turbo::Method::Bm25l,
            bm25_turbo::Method::Bm25Plus,
        ];

        for method in methods {
            let idf = bm25_turbo::scoring::idf(method, num_docs, doc_freq);
            prop_assert!(idf > 0.0, "IDF should be positive for {:?} with N={}, df={}", method, num_docs, doc_freq);
        }
    }

    /// TFC should be non-negative for all valid inputs.
    #[test]
    fn tfc_non_negative(
        tf in 0.1f32..100.0,
        doc_len in 1.0f32..10000.0,
        avg_doc_len in 1.0f32..10000.0,
        k1 in 0.1f32..10.0,
        b in 0.0f32..1.0,
        delta in 0.0f32..2.0,
    ) {
        let methods = [
            bm25_turbo::Method::Robertson,
            bm25_turbo::Method::Lucene,
            bm25_turbo::Method::Atire,
            bm25_turbo::Method::Bm25l,
            bm25_turbo::Method::Bm25Plus,
        ];

        for method in methods {
            let result = bm25_turbo::scoring::tfc(method, tf, doc_len, avg_doc_len, k1, b, delta);
            prop_assert!(result >= 0.0, "TFC should be non-negative for {:?}", method);
            prop_assert!(result.is_finite(), "TFC should be finite for {:?}", method);
        }
    }

    // -----------------------------------------------------------------------
    // CSC Matrix property tests
    // -----------------------------------------------------------------------

    /// Random triplets -> build CSC -> validate invariants always hold.
    /// This is the core property test: any valid set of triplets should
    /// produce a structurally valid CSC matrix.
    #[test]
    fn csc_from_random_triplets_always_valid(
        num_triplets in 1usize..500,
        num_docs in 1u32..200,
        vocab_size in 1u32..100,
        seed in 0u64..u64::MAX,
    ) {
        // Generate deterministic pseudo-random triplets from seed.
        let mut rng_state = seed;
        let mut triplets = Vec::with_capacity(num_triplets);
        for _ in 0..num_triplets {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let term_id = ((rng_state >> 33) as u32) % vocab_size;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let doc_id = ((rng_state >> 33) as u32) % num_docs;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let score = ((rng_state & 0xFFFF) as f32) / 65535.0 * 10.0;
            triplets.push((term_id, doc_id, score));
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, vocab_size).unwrap();
        prop_assert!(m.validate().is_ok(), "CSC matrix should be valid after from_triplets");
    }

    /// Round-trip property: every input triplet's score is accounted for
    /// in the CSC output (after duplicate merging via summation).
    #[test]
    fn csc_round_trip_scores_match(
        num_triplets in 1usize..200,
        num_docs in 1u32..50,
        vocab_size in 1u32..30,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let mut triplets = Vec::with_capacity(num_triplets);
        for _ in 0..num_triplets {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let term_id = ((rng_state >> 33) as u32) % vocab_size;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let doc_id = ((rng_state >> 33) as u32) % num_docs;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let score = ((rng_state & 0xFFFF) as f32) / 65535.0 * 10.0;
            triplets.push((term_id, doc_id, score));
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, vocab_size).unwrap();

        // Build expected merged scores per (term, doc).
        let mut expected: std::collections::HashMap<(u32, u32), f32> = std::collections::HashMap::new();
        for &(term_id, doc_id, score) in &triplets {
            *expected.entry((term_id, doc_id)).or_insert(0.0) += score;
        }

        // Verify CSC nnz matches unique (term, doc) count.
        prop_assert_eq!(m.nnz(), expected.len());

        // Verify each expected entry is present with correct merged score.
        for (&(term_id, doc_id), &exp_score) in &expected {
            let (scores, doc_ids) = m.column(term_id);
            let pos = doc_ids.iter().position(|&d| d == doc_id);
            prop_assert!(pos.is_some(), "doc_id {} not found in column {}", doc_id, term_id);
            let actual = scores[pos.unwrap()];
            prop_assert!(
                (actual - exp_score).abs() < 1e-3,
                "score mismatch for ({}, {}): expected {}, got {}",
                term_id, doc_id, exp_score, actual
            );
        }
    }

    /// CSC column indices are always sorted (strictly ascending) after construction.
    #[test]
    fn csc_columns_always_sorted(
        num_triplets in 1usize..300,
        num_docs in 1u32..100,
        vocab_size in 1u32..50,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let mut triplets = Vec::with_capacity(num_triplets);
        for _ in 0..num_triplets {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let term_id = ((rng_state >> 33) as u32) % vocab_size;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let doc_id = ((rng_state >> 33) as u32) % num_docs;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let score = ((rng_state & 0xFFFF) as f32) / 65535.0;
            triplets.push((term_id, doc_id, score));
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, vocab_size).unwrap();

        for col in 0..vocab_size {
            let (_, doc_ids) = m.column(col);
            for i in 1..doc_ids.len() {
                prop_assert!(
                    doc_ids[i] > doc_ids[i - 1],
                    "column {} has unsorted or duplicate indices at position {}: {} vs {}",
                    col, i, doc_ids[i - 1], doc_ids[i]
                );
            }
        }
    }

    /// nnz of constructed CSC is always <= number of input triplets
    /// (duplicates merge, reducing count).
    #[test]
    fn csc_nnz_leq_input_count(
        num_triplets in 0usize..500,
        num_docs in 1u32..100,
        vocab_size in 1u32..50,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let mut triplets = Vec::with_capacity(num_triplets);
        for _ in 0..num_triplets {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let term_id = ((rng_state >> 33) as u32) % vocab_size;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let doc_id = ((rng_state >> 33) as u32) % num_docs;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let score = ((rng_state & 0xFFFF) as f32) / 65535.0;
            triplets.push((term_id, doc_id, score));
        }

        let m = CscMatrix::from_triplets(&triplets, num_docs, vocab_size).unwrap();
        prop_assert!(m.nnz() <= num_triplets, "nnz {} > input count {}", m.nnz(), num_triplets);
    }

    /// Empty triplets always produce a valid empty CSC matrix.
    #[test]
    fn csc_empty_triplets_always_valid(
        num_docs in 1u32..1000,
        vocab_size in 1u32..500,
    ) {
        let m = CscMatrix::from_triplets(&[], num_docs, vocab_size).unwrap();
        prop_assert!(m.validate().is_ok());
        prop_assert_eq!(m.nnz(), 0);
        prop_assert_eq!(m.indptr.len(), vocab_size as usize + 1);
    }

    // -----------------------------------------------------------------------
    // Phase 3: Index construction & query property tests
    // -----------------------------------------------------------------------

    /// Build-query round trip: for any non-empty corpus, building an index
    /// and searching for a term that exists in the corpus always returns
    /// at least one result with a positive score.
    #[test]
    fn index_build_query_round_trip(
        num_docs in 2usize..20,
        seed in 0u64..u64::MAX,
    ) {
        // Generate a small corpus with a known "needle" term.
        let needle = "xyzzyplugh";
        let mut rng_state = seed;
        let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];

        let mut corpus_strings: Vec<String> = Vec::with_capacity(num_docs);
        for i in 0..num_docs {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let word_count = ((rng_state >> 40) % 5 + 2) as usize;
            let mut doc = String::new();
            for w in 0..word_count {
                if w > 0 { doc.push(' '); }
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = ((rng_state >> 33) as usize) % words.len();
                doc.push_str(words[idx]);
            }
            // Inject needle into the first document.
            if i == 0 {
                doc.push(' ');
                doc.push_str(needle);
            }
            corpus_strings.push(doc);
        }

        let corpus_refs: Vec<&str> = corpus_strings.iter().map(|s| s.as_str()).collect();
        let index = bm25_turbo::BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .unwrap();

        let results = index.search(needle, 5).unwrap();
        prop_assert!(!results.doc_ids.is_empty(), "needle should be found in corpus");
        prop_assert!(results.doc_ids.contains(&0), "doc 0 should contain the needle");
        prop_assert!(results.scores[0] > 0.0, "top score should be positive");
    }

    /// top_k and top_k_heap always agree on random score arrays.
    #[test]
    fn top_k_methods_agree(
        n in 1usize..500,
        k in 1usize..50,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let scores: Vec<f32> = (0..n)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let raw = ((rng_state >> 40) as f32) / 16777216.0;
                // ~30% chance of zero score
                if (rng_state & 0x3) == 0 { 0.0 } else { raw }
            })
            .collect();

        let r1 = bm25_turbo::selection::top_k(&scores, k);
        let r2 = bm25_turbo::selection::top_k_heap(&scores, k);

        prop_assert_eq!(r1.doc_ids.len(), r2.doc_ids.len(), "result count mismatch");
        prop_assert_eq!(r1.doc_ids, r2.doc_ids, "doc_ids mismatch");
        for (i, (s1, s2)) in r1.scores.iter().zip(r2.scores.iter()).enumerate() {
            prop_assert_eq!(
                s1.to_bits(), s2.to_bits(),
                "score mismatch at position {}: {} vs {}", i, s1, s2
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: SIMD property tests
    // -----------------------------------------------------------------------

    /// Proptest: random SIMD scatter_add inputs always produce same result as scalar.
    #[test]
    fn simd_scatter_add_matches_scalar(
        size in 0usize..2000,
        acc_size_factor in 1usize..500,
        seed in 0u64..u64::MAX,
    ) {
        let acc_size = acc_size_factor.max(1);
        let mut acc_simd = vec![0.0f32; acc_size];
        let mut acc_scalar = vec![0.0f32; acc_size];

        let mut rng_state = seed;
        let indices: Vec<u32> = (0..size)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng_state >> 33) as u32) % (acc_size as u32)
            })
            .collect();
        let values: Vec<f32> = (0..size)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng_state & 0xFFFF) as f32) / 65535.0 * 10.0 - 5.0
            })
            .collect();

        bm25_turbo::simd::scatter_add(&mut acc_simd, &indices, &values);
        bm25_turbo::simd::scatter_add_scalar(&mut acc_scalar, &indices, &values);

        for j in 0..acc_size {
            prop_assert_eq!(
                acc_simd[j].to_bits(), acc_scalar[j].to_bits(),
                "scatter_add mismatch at index {} for size {}", j, size
            );
        }
    }

    /// Proptest: random SIMD dot_product inputs always produce same result as scalar.
    #[test]
    fn simd_dot_product_matches_scalar(
        len in 0usize..5000,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let a: Vec<f32> = (0..len)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng_state & 0xFFFF) as f32) / 65535.0 * 20.0 - 10.0
            })
            .collect();
        let b: Vec<f32> = (0..len)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng_state & 0xFFFF) as f32) / 65535.0 * 20.0 - 10.0
            })
            .collect();

        let simd_result = bm25_turbo::simd::dot_product(&a, &b);
        let scalar_result = bm25_turbo::simd::dot_product_scalar(&a, &b);

        prop_assert_eq!(
            simd_result.to_bits(), scalar_result.to_bits(),
            "dot_product mismatch for len {}: simd={} scalar={}", len, simd_result, scalar_result
        );
    }

    /// Proptest: random SIMD max_f32 inputs always produce same result as scalar.
    #[test]
    fn simd_max_f32_matches_scalar(
        len in 0usize..5000,
        seed in 0u64..u64::MAX,
    ) {
        let mut rng_state = seed;
        let values: Vec<f32> = (0..len)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng_state & 0xFFFF) as f32) / 65535.0 * 200.0 - 100.0
            })
            .collect();

        let simd_result = bm25_turbo::simd::max_f32(&values);
        let scalar_result = bm25_turbo::simd::max_f32_scalar(&values);

        prop_assert_eq!(
            simd_result.to_bits(), scalar_result.to_bits(),
            "max_f32 mismatch for len {}: simd={} scalar={}", len, simd_result, scalar_result
        );
    }

    /// All five BM25 variants produce valid (non-NaN, non-infinite) scores
    /// for any reasonable corpus configuration.
    #[test]
    fn all_variants_produce_valid_scores(
        k1 in 0.1f32..5.0,
        b_val in 0.0f32..1.0,
        delta in 0.0f32..2.0,
    ) {
        let corpus = &[
            "the quick brown fox jumps",
            "lazy dog sleeps all day",
            "quick fox and lazy dog",
            "brown brown brown fox fox",
        ];
        let methods = [
            bm25_turbo::Method::Robertson,
            bm25_turbo::Method::Lucene,
            bm25_turbo::Method::Atire,
            bm25_turbo::Method::Bm25l,
            bm25_turbo::Method::Bm25Plus,
        ];
        for method in methods {
            let index = bm25_turbo::BM25Builder::new()
                .method(method)
                .k1(k1)
                .b(b_val)
                .delta(delta)
                .build_from_corpus(corpus)
                .unwrap();
            let results = index.search("fox", 2).unwrap();
            for &score in &results.scores {
                prop_assert!(score.is_finite(), "{:?} produced non-finite score: {}", method, score);
                // All variants should produce positive scores for a query term that
                // appears in a minority of documents (IDF > 0 for all variants when df < N/2).
                prop_assert!(score > 0.0, "{:?} produced non-positive score: {}", method, score);
            }
        }
    }
}
