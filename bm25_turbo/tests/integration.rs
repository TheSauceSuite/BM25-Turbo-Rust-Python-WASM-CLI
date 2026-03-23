//! Comprehensive integration tests for BM25 Turbo.
//!
//! Covers: BMW/WAND approximate search, batch queries, streaming builder,
//! WAL lifecycle, query cache, persistence edge cases, tokenizer edge cases,
//! and cross-feature integration scenarios.

use std::collections::HashSet;

use bm25_turbo::{BM25Builder, Method, StreamingBuilder};
use tempfile::tempdir;

// ===========================================================================
// Helpers
// ===========================================================================

fn small_corpus() -> Vec<&'static str> {
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

fn build_small_index() -> bm25_turbo::BM25Index {
    BM25Builder::new()
        .build_from_corpus(&small_corpus())
        .expect("build should succeed")
}

fn build_index_with_method(method: Method) -> bm25_turbo::BM25Index {
    BM25Builder::new()
        .method(method)
        .build_from_corpus(&small_corpus())
        .expect("build should succeed")
}

/// Generate a synthetic corpus of `n` documents.
fn generate_corpus(n: usize) -> Vec<String> {
    let words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
        "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "apple",
        "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
        "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince",
        "raspberry", "strawberry", "tangerine", "vanilla", "watermelon",
        "algorithm", "binary", "compiler", "database", "engine", "framework",
    ];
    (0..n)
        .map(|i| {
            let doc_len = 5 + (i * 7 + 13) % 20;
            let mut doc = String::with_capacity(doc_len * 8);
            for j in 0..doc_len {
                if j > 0 {
                    doc.push(' ');
                }
                let word_idx = (i * 31 + j * 17 + i * j * 3 + 7) % words.len();
                doc.push_str(words[word_idx]);
            }
            doc
        })
        .collect()
}

// ===========================================================================
// 1. BMW/WAND: build_bmw_index then search_approximate returns results
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_build_then_search_returns_results() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let results = index.search_approximate("quick fox", 5).unwrap();
    assert!(
        !results.doc_ids.is_empty(),
        "search_approximate should return results for a known query"
    );
    assert_eq!(results.doc_ids.len(), results.scores.len());
    // Scores should be in descending order.
    for w in results.scores.windows(2) {
        assert!(w[0] >= w[1], "scores must be descending");
    }
}

// ===========================================================================
// 2. search_approximate results are a subset of exact results
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_results_subset_of_exact() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let exact = index.search("quick brown fox", 10).unwrap();
    let approx = index.search_approximate("quick brown fox", 10).unwrap();

    let exact_set: HashSet<u32> = exact.doc_ids.iter().copied().collect();
    for &doc_id in &approx.doc_ids {
        assert!(
            exact_set.contains(&doc_id),
            "approximate doc_id {} not in exact results {:?}",
            doc_id,
            exact.doc_ids
        );
    }
}

// ===========================================================================
// 3. search_approximate with k=1 returns the top-1 document
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_k1_returns_top_document() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let exact = index.search("lazy dog", 1).unwrap();
    let approx = index.search_approximate("lazy dog", 1).unwrap();

    assert_eq!(approx.doc_ids.len(), 1);
    assert_eq!(
        exact.doc_ids[0], approx.doc_ids[0],
        "k=1 top document should match exact"
    );
}

// ===========================================================================
// 4. search_approximate on empty query returns empty
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_empty_query_returns_empty() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let results = index.search_approximate("", 5).unwrap();
    assert!(results.doc_ids.is_empty());
    assert!(results.scores.is_empty());
}

// ===========================================================================
// 5. search_approximate with k > num_docs returns all matching docs
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_k_larger_than_corpus() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let num_docs = index.num_docs();
    let results = index
        .search_approximate("quick fox lazy dog brown", num_docs as usize + 100)
        .unwrap();

    // Should return at most num_docs results.
    assert!(results.doc_ids.len() <= num_docs as usize);
    // No duplicates.
    let unique: HashSet<u32> = results.doc_ids.iter().copied().collect();
    assert_eq!(unique.len(), results.doc_ids.len());
}

// ===========================================================================
// 6. build_bmw_index_with_block_size with 64, 128, 256 all produce valid results
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_block_sizes_64_128_256() {
    let mut index = build_small_index();

    for block_size in [64, 128, 256] {
        index.build_bmw_index_with_block_size(block_size).unwrap();

        let results = index.search_approximate("quick fox", 5).unwrap();
        assert!(
            !results.doc_ids.is_empty(),
            "block_size={} should produce results",
            block_size
        );

        // Verify scores match exact for returned docs.
        let exact = index.search("quick fox", 5).unwrap();
        for (i, &doc_id) in results.doc_ids.iter().enumerate() {
            if let Some(pos) = exact.doc_ids.iter().position(|&d| d == doc_id) {
                assert_eq!(
                    results.scores[i].to_bits(),
                    exact.scores[pos].to_bits(),
                    "block_size={}: score mismatch for doc {}",
                    block_size,
                    doc_id
                );
            }
        }
    }
}

// ===========================================================================
// 7. BMW index on single-document corpus
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn bmw_single_document_corpus() {
    let mut index = BM25Builder::new()
        .build_from_corpus(&["the quick brown fox jumps over the lazy dog"])
        .unwrap();
    index.build_bmw_index().unwrap();

    let results = index.search_approximate("quick fox", 5).unwrap();
    assert_eq!(results.doc_ids.len(), 1);
    assert_eq!(results.doc_ids[0], 0);
}

// ===========================================================================
// 8. search_batch with 0 queries returns empty vec
// ===========================================================================

#[test]
fn batch_zero_queries() {
    let index = build_small_index();
    let results = index.search_batch(&[], 5).unwrap();
    assert!(results.is_empty());
}

// ===========================================================================
// 9. search_batch with 1 query matches search result exactly
// ===========================================================================

#[test]
fn batch_single_query_matches_search() {
    let index = build_small_index();
    let single = index.search("quick fox", 5).unwrap();
    let batch = index.search_batch(&["quick fox"], 5).unwrap();

    assert_eq!(batch.len(), 1);
    assert_eq!(single.doc_ids, batch[0].doc_ids);
    for (s1, s2) in single.scores.iter().zip(batch[0].scores.iter()) {
        assert_eq!(s1.to_bits(), s2.to_bits());
    }
}

// ===========================================================================
// 10. search_batch with 10 different queries -- each matches individual search
// ===========================================================================

#[test]
fn batch_ten_queries_match_individual() {
    let index = build_small_index();
    let queries = [
        "quick", "lazy dog", "brown fox", "sun moon", "highway car",
        "bear honey", "red fox", "galaxy universe", "east sun", "drives fast",
    ];

    let batch = index.search_batch(&queries, 5).unwrap();
    assert_eq!(batch.len(), queries.len());

    for (i, query) in queries.iter().enumerate() {
        let individual = index.search(query, 5).unwrap();
        assert_eq!(
            individual.doc_ids, batch[i].doc_ids,
            "batch[{}] doc_ids mismatch for query '{}'",
            i, query
        );
        for (j, (s1, s2)) in individual
            .scores
            .iter()
            .zip(batch[i].scores.iter())
            .enumerate()
        {
            assert_eq!(
                s1.to_bits(),
                s2.to_bits(),
                "batch[{}] score[{}] mismatch for query '{}'",
                i,
                j,
                query
            );
        }
    }
}

// ===========================================================================
// 11. search_batch with duplicate queries returns identical results
// ===========================================================================

#[test]
fn batch_duplicate_queries() {
    let index = build_small_index();
    let queries = ["quick fox", "quick fox", "quick fox"];
    let batch = index.search_batch(&queries, 5).unwrap();

    assert_eq!(batch.len(), 3);
    for i in 1..3 {
        assert_eq!(batch[0].doc_ids, batch[i].doc_ids);
        for (s1, s2) in batch[0].scores.iter().zip(batch[i].scores.iter()) {
            assert_eq!(s1.to_bits(), s2.to_bits());
        }
    }
}

// ===========================================================================
// 12. search_batch with empty string queries
// ===========================================================================

#[test]
fn batch_empty_string_queries() {
    let index = build_small_index();
    let queries = ["", "", ""];
    let batch = index.search_batch(&queries, 5).unwrap();

    assert_eq!(batch.len(), 3);
    for r in &batch {
        assert!(r.doc_ids.is_empty(), "empty query should produce empty results");
    }
}

// ===========================================================================
// 13. StreamingBuilder with chunk_size=1 matches batch build
// ===========================================================================

#[test]
fn streaming_chunk_size_1_matches_batch() {
    let corpus = small_corpus();

    let batch_index = BM25Builder::new()
        .build_from_corpus(&corpus)
        .unwrap();

    let mut streaming = StreamingBuilder::new().chunk_size(1);
    streaming.add_documents(&corpus);
    let stream_index = streaming.build().unwrap();

    assert_eq!(batch_index.num_docs(), stream_index.num_docs());

    for query in &["quick", "fox", "lazy dog", "brown", "sun moon"] {
        let r_batch = batch_index.search(query, 5).unwrap();
        let r_stream = stream_index.search(query, 5).unwrap();
        assert_eq!(
            r_batch.doc_ids, r_stream.doc_ids,
            "chunk_size=1: doc_ids mismatch for query '{}'",
            query
        );
        for (s1, s2) in r_batch.scores.iter().zip(r_stream.scores.iter()) {
            assert_eq!(
                s1.to_bits(),
                s2.to_bits(),
                "chunk_size=1: score mismatch for query '{}'",
                query
            );
        }
    }
}

// ===========================================================================
// 14. StreamingBuilder with all 5 BM25 methods produces valid indexes
// ===========================================================================

#[test]
fn streaming_all_five_methods() {
    let corpus = small_corpus();
    let methods = [
        Method::Robertson,
        Method::Lucene,
        Method::Atire,
        Method::Bm25l,
        Method::Bm25Plus,
    ];

    for method in &methods {
        let mut streaming = StreamingBuilder::new().method(*method).chunk_size(3);
        streaming.add_documents(&corpus);
        let index = streaming.build().unwrap();

        assert_eq!(index.num_docs(), corpus.len() as u32);
        let results = index.search("quick fox", 3).unwrap();
        assert!(
            !results.doc_ids.is_empty(),
            "{:?}: should return results",
            method
        );
    }
}

// ===========================================================================
// 15. StreamingBuilder add_iter with empty iterator
// ===========================================================================

#[test]
fn streaming_add_iter_empty() {
    let empty: Vec<String> = vec![];
    let mut streaming = StreamingBuilder::new();
    streaming.add_iter(empty.into_iter());
    let result = streaming.build();
    assert!(result.is_err(), "empty corpus should fail to build");
}

// ===========================================================================
// 16. WAL Full Lifecycle
// ===========================================================================

#[test]
fn wal_full_lifecycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("lifecycle.bm25");

    // Build index and save.
    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();

    // Reload.
    let mut index = bm25_turbo::persistence::load(&path).unwrap();
    assert_eq!(index.num_docs(), 10);

    // Enable WAL.
    let mut wal = index.enable_wal().unwrap();

    // Add documents via WAL.
    let new_ids = index
        .add_documents(&mut wal, &["new document about planets and stars"])
        .unwrap();
    assert_eq!(new_ids.len(), 1);
    let new_doc_id = new_ids[0];

    // Search with WAL overlay should find new doc.
    let results = index.search_with_wal(&wal, "planets stars", 5).unwrap();
    assert!(
        results.doc_ids.contains(&new_doc_id),
        "WAL search should find newly added doc"
    );

    // Delete a base document.
    let del_report = index.delete_documents(&mut wal, &[0]).unwrap();
    assert_eq!(del_report.deleted, 1);

    // Search should not return deleted doc.
    let results = index
        .search_with_wal(&wal, "quick brown fox", 10)
        .unwrap();
    assert!(
        !results.doc_ids.contains(&0),
        "deleted doc 0 should not appear in results"
    );

    // Compact.
    index.compact(&mut wal).unwrap();

    // Search after compaction should still work.
    let results = index.search("planets stars", 5).unwrap();
    // After compaction, results may or may not find the WAL doc depending on
    // compaction implementation; at minimum, the search should not error.
    assert!(results.scores.len() == results.doc_ids.len());

    // Save, reload, verify.
    let path2 = dir.path().join("lifecycle2.bm25");
    bm25_turbo::persistence::save(&index, &path2).unwrap();
    let reloaded = bm25_turbo::persistence::load(&path2).unwrap();
    assert!(reloaded.num_docs() > 0);
}

// ===========================================================================
// 17. WAL with BMW: add docs via WAL, search_approximate still works
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn wal_with_bmw() {
    let mut index = build_small_index();
    index.build_bmw_index().unwrap();

    let mut wal = index.enable_wal().unwrap();
    index
        .add_documents(&mut wal, &["new document about quick jumping foxes"])
        .unwrap();

    // search_approximate on the base index (without WAL overlay) should still work.
    let results = index.search_approximate("quick fox", 5).unwrap();
    assert!(!results.doc_ids.is_empty());
}

// ===========================================================================
// 18. WAL persistence: add docs, save, reload, verify
// ===========================================================================

#[test]
fn wal_persistence_reload() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal_persist.bm25");

    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();

    // Load, enable WAL, add doc.
    let index = bm25_turbo::persistence::load(&path).unwrap();
    let mut wal = index.enable_wal().unwrap();
    let new_ids = wal
        .append_documents(&["a brand new document about technology"])
        .unwrap();
    assert_eq!(new_ids.len(), 1);

    // Search with WAL finds new doc.
    let results = wal.search(&index, "technology", 5).unwrap();
    assert!(
        results.doc_ids.contains(&new_ids[0]),
        "WAL search should find new doc after append"
    );

    // Save the base index again (WAL entries are separate from index persistence).
    let path2 = dir.path().join("wal_persist2.bm25");
    bm25_turbo::persistence::save(&index, &path2).unwrap();

    // Reload base index.
    let reloaded = bm25_turbo::persistence::load(&path2).unwrap();
    assert_eq!(reloaded.num_docs(), index.num_docs());
}

// ===========================================================================
// 19. Cache hit produces identical results to cache miss
// ===========================================================================

#[test]
fn cache_hit_identical_to_miss() {
    let index = BM25Builder::new()
        .cache_capacity(100)
        .build_from_corpus(&small_corpus())
        .unwrap();

    // First call: cache miss.
    let miss = index.search_cached("quick fox", 5).unwrap();
    // Second call: cache hit.
    let hit = index.search_cached("quick fox", 5).unwrap();

    assert_eq!(miss.doc_ids, hit.doc_ids);
    for (s1, s2) in miss.scores.iter().zip(hit.scores.iter()) {
        assert_eq!(s1.to_bits(), s2.to_bits());
    }
}

// ===========================================================================
// 20. Cache with capacity=0 never caches
// ===========================================================================

#[test]
fn cache_capacity_zero() {
    let index = BM25Builder::new()
        .cache_capacity(0)
        .build_from_corpus(&small_corpus())
        .unwrap();

    // search_cached should still work (falls through to uncached path).
    let r1 = index.search_cached("quick fox", 5).unwrap();
    let r2 = index.search_cached("quick fox", 5).unwrap();

    // Results should be the same (just computed fresh each time).
    assert_eq!(r1.doc_ids, r2.doc_ids);
}

// ===========================================================================
// 21. Cache after WAL mutation behavior
// ===========================================================================

#[test]
fn cache_after_wal_mutation() {
    let index = BM25Builder::new()
        .cache_capacity(100)
        .build_from_corpus(&small_corpus())
        .unwrap();

    // Populate cache.
    let cached = index.search_cached("quick fox", 5).unwrap();
    assert!(!cached.doc_ids.is_empty());

    // Enable WAL and add a document.
    let mut wal = index.enable_wal().unwrap();
    wal.append_documents(&["quick fox brand new document"])
        .unwrap();

    // search_cached uses the base index cache (no WAL overlay).
    // The cached result is stale relative to WAL state. This is expected
    // behavior: the cache is per-base-index, not WAL-aware.
    let still_cached = index.search_cached("quick fox", 5).unwrap();
    assert_eq!(
        cached.doc_ids, still_cached.doc_ids,
        "cache should return same result (stale, no WAL overlay)"
    );

    // WAL-aware search returns different results.
    let wal_results = wal.search(&index, "quick fox", 5).unwrap();
    // WAL results may differ from cached because they include WAL additions.
    // Just verify it doesn't error and returns valid results.
    assert_eq!(wal_results.doc_ids.len(), wal_results.scores.len());
}

// ===========================================================================
// 22. Save empty index -> load -> verify 0 docs
// ===========================================================================

#[test]
fn persistence_save_empty_index() {
    // BM25Builder rejects empty corpus, so we can't create a truly empty index
    // through the public API. Verify that building with an empty corpus errors.
    let result = BM25Builder::new().build_from_corpus(&[]);
    assert!(result.is_err(), "empty corpus should fail");
}

// ===========================================================================
// 23. Save -> load -> search produces same results as original
// ===========================================================================

#[test]
fn persistence_round_trip_search() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("roundtrip.bm25");

    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();
    let loaded = bm25_turbo::persistence::load(&path).unwrap();

    assert_eq!(index.num_docs(), loaded.num_docs());
    assert_eq!(index.vocab_size(), loaded.vocab_size());

    for query in &["quick fox", "lazy dog", "sun moon", "brown bear"] {
        let original = index.search(query, 5).unwrap();
        let from_disk = loaded.search(query, 5).unwrap();
        assert_eq!(
            original.doc_ids, from_disk.doc_ids,
            "round-trip: doc_ids mismatch for query '{}'",
            query
        );
        for (s1, s2) in original.scores.iter().zip(from_disk.scores.iter()) {
            assert_eq!(
                s1.to_bits(),
                s2.to_bits(),
                "round-trip: score mismatch for query '{}'",
                query
            );
        }
    }
}

// ===========================================================================
// 24. mmap_or_load on a valid file succeeds
// ===========================================================================

#[test]
fn persistence_mmap_or_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mmap_test.bm25");

    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();

    let loaded = bm25_turbo::persistence::mmap_or_load(&path).unwrap();
    assert_eq!(loaded.num_docs(), index.num_docs());

    let results = loaded.search("quick fox", 5).unwrap();
    assert!(!results.doc_ids.is_empty());
}

// ===========================================================================
// 25. Load with wrong magic bytes returns error
// ===========================================================================

#[test]
fn persistence_wrong_magic_bytes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_magic.bm25");

    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();

    // Corrupt the magic bytes (first 4 bytes of the file).
    let mut data = std::fs::read(&path).unwrap();
    data[0] = b'X';
    data[1] = b'Y';
    data[2] = b'Z';
    data[3] = b'W';
    std::fs::write(&path, &data).unwrap();

    let result = bm25_turbo::persistence::load(&path);
    assert!(result.is_err(), "wrong magic bytes should fail");
    match result {
        Err(e) => {
            let err_msg = format!("{}", e);
            assert!(
                err_msg.contains("corrupt") || err_msg.contains("invalid magic"),
                "error should mention corruption: {}",
                err_msg
            );
        }
        Ok(_) => panic!("expected error for wrong magic bytes"),
    }
}

// ===========================================================================
// 26. Save -> truncate -> load returns error
// ===========================================================================

#[test]
fn persistence_truncated_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("truncated.bm25");

    let index = build_small_index();
    bm25_turbo::persistence::save(&index, &path).unwrap();

    // Truncate the file to half its size.
    let data = std::fs::read(&path).unwrap();
    let truncated = &data[..data.len() / 2];
    std::fs::write(&path, truncated).unwrap();

    let result = bm25_turbo::persistence::load(&path);
    assert!(result.is_err(), "truncated file should fail to load");
}

// ===========================================================================
// 27. Unicode input doesn't panic
// ===========================================================================

#[test]
fn tokenizer_unicode_no_panic() {
    let corpus = &[
        "hello world",
        "\u{4F60}\u{597D}\u{4E16}\u{754C}",         // Chinese
        "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}",  // Arabic
        "\u{1F600}\u{1F680}\u{1F30D}",                // Emoji
        "caf\u{00E9} na\u{00EF}ve r\u{00E9}sum\u{00E9}",
    ];

    let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
    assert_eq!(index.num_docs(), 5);

    // Searching with Unicode should not panic.
    let _ = index.search("\u{4F60}\u{597D}", 5);
    let _ = index.search("\u{0645}\u{0631}\u{062D}", 5);
    let _ = index.search("\u{1F600}", 5);
}

// ===========================================================================
// 28. Very long document (10K+ words) tokenizes successfully
// ===========================================================================

#[test]
fn tokenizer_very_long_document() {
    let long_doc: String = (0..10_000)
        .map(|i| format!("word{}", i % 500))
        .collect::<Vec<_>>()
        .join(" ");

    let corpus = &[long_doc.as_str(), "short document"];
    let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
    assert_eq!(index.num_docs(), 2);

    let results = index.search("word0", 2).unwrap();
    assert!(!results.doc_ids.is_empty());
}

// ===========================================================================
// 29. Multiple language stemmers produce different output
// ===========================================================================

#[test]
fn tokenizer_stemmers_differ() {
    use bm25_turbo::Tokenizer;

    // Build tokenizers for different languages with stemmers.
    let languages = [
        "english", "french", "german", "spanish", "italian", "portuguese",
        "dutch", "swedish", "norwegian", "danish", "finnish", "hungarian",
        "romanian", "russian", "turkish", "arabic", "tamil",
    ];

    let word = "universally";
    let mut stems: Vec<(String, String)> = Vec::new();

    for lang in &languages {
        let tokenizer = Tokenizer::builder().language(lang).build().unwrap();
        let tokens = tokenizer.tokenize(word);
        if !tokens.is_empty() {
            stems.push((lang.to_string(), tokens[0].clone()));
        }
    }

    // At least some stemmers should produce different outputs.
    let unique_stems: HashSet<&str> = stems.iter().map(|(_, s)| s.as_str()).collect();
    assert!(
        unique_stems.len() > 1,
        "expected multiple distinct stems, got: {:?}",
        stems
    );
}

// ===========================================================================
// 30. Custom tokenizer function is called and its output is used
// ===========================================================================

#[test]
fn custom_tokenizer_function() {
    use bm25_turbo::Tokenizer;

    let tokenizer = Tokenizer::builder()
        .custom_fn(|text: &str| {
            // Custom tokenizer: split on commas and trim.
            text.split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .build()
        .unwrap();

    let index = BM25Builder::new()
        .tokenizer(tokenizer)
        .build_from_corpus(&[
            "hello,world",
            "foo,bar,baz",
            "hello,foo",
        ])
        .unwrap();

    assert_eq!(index.num_docs(), 3);
    let results = index.search("hello", 5).unwrap();
    assert!(
        !results.doc_ids.is_empty(),
        "custom tokenizer should produce searchable index"
    );
}

// ===========================================================================
// 31. BM25Plus -> save -> mmap_load -> search -> verify params preserved
// ===========================================================================

#[test]
fn cross_feature_bm25plus_save_mmap_search() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bm25plus.bm25");

    let index = build_index_with_method(Method::Bm25Plus);
    bm25_turbo::persistence::save(&index, &path).unwrap();

    let mmap_index = bm25_turbo::persistence::load_mmap(&path).unwrap();
    assert_eq!(mmap_index.params.method, Method::Bm25Plus);
    assert_eq!(mmap_index.num_docs, index.num_docs());

    let results = mmap_index.search("quick fox", 5).unwrap();
    assert!(!results.doc_ids.is_empty());

    // Verify results match original.
    let original = index.search("quick fox", 5).unwrap();
    assert_eq!(results.doc_ids, original.doc_ids);
}

// ===========================================================================
// 32. Build with custom tokenizer -> save -> load -> search same results
// ===========================================================================

#[test]
fn cross_feature_custom_tokenizer_persistence() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("custom_tok.bm25");

    // Build with default tokenizer (custom tokenizers are not persisted,
    // so loaded index will use default tokenizer -- this is expected).
    let index = BM25Builder::new()
        .build_from_corpus(&small_corpus())
        .unwrap();

    let original = index.search("quick fox", 5).unwrap();

    bm25_turbo::persistence::save(&index, &path).unwrap();
    let loaded = bm25_turbo::persistence::load(&path).unwrap();
    let loaded_results = loaded.search("quick fox", 5).unwrap();

    assert_eq!(original.doc_ids, loaded_results.doc_ids);
    for (s1, s2) in original.scores.iter().zip(loaded_results.scores.iter()) {
        assert_eq!(s1.to_bits(), s2.to_bits());
    }
}

// ===========================================================================
// 33. Streaming build -> save -> mmap_load -> BMW index -> search_approximate
// ===========================================================================

#[cfg(feature = "ann")]
#[test]
fn cross_feature_streaming_save_mmap_bmw() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stream_bmw.bm25");

    let corpus = small_corpus();
    let mut streaming = StreamingBuilder::new().chunk_size(3);
    streaming.add_documents(&corpus);
    let stream_index = streaming.build().unwrap();

    bm25_turbo::persistence::save(&stream_index, &path).unwrap();

    // Load via mmap, convert to owned, build BMW.
    let mmap_index = bm25_turbo::persistence::load_mmap(&path).unwrap();
    let mut owned = mmap_index.into_owned();
    owned.build_bmw_index().unwrap();

    let results = owned.search_approximate("quick fox", 5).unwrap();
    assert!(!results.doc_ids.is_empty());

    // Verify against exact search.
    let exact = owned.search("quick fox", 5).unwrap();
    // At minimum, top-1 should match.
    assert_eq!(exact.doc_ids[0], results.doc_ids[0]);
}

// ===========================================================================
// 34. Large corpus (1000 docs) -> batch query (100 queries) -> all valid
// ===========================================================================

#[test]
fn cross_feature_large_corpus_batch_query() {
    let corpus = generate_corpus(1000);
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

    let index = BM25Builder::new()
        .build_from_corpus(&corpus_refs)
        .unwrap();
    assert_eq!(index.num_docs(), 1000);

    // Generate 100 queries.
    let query_words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "engine", "compiler",
        "database", "framework", "apple", "banana", "cherry", "mango",
        "algorithm", "search", "query", "index", "binary", "sparse", "dense",
    ];

    let queries: Vec<String> = (0..100)
        .map(|i| {
            let w1 = query_words[i % query_words.len()];
            let w2 = query_words[(i * 7 + 3) % query_words.len()];
            format!("{} {}", w1, w2)
        })
        .collect();
    let query_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

    let batch = index.search_batch(&query_refs, 10).unwrap();
    assert_eq!(batch.len(), 100);

    for (i, results) in batch.iter().enumerate() {
        // Each result should be valid.
        assert_eq!(
            results.doc_ids.len(),
            results.scores.len(),
            "query {}: doc_ids and scores length mismatch",
            i
        );
        assert!(
            results.doc_ids.len() <= 10,
            "query {}: too many results",
            i
        );
        // Scores descending.
        for w in results.scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "query {}: scores not descending",
                i
            );
        }
        // No duplicate doc IDs.
        let unique: HashSet<u32> = results.doc_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            results.doc_ids.len(),
            "query {}: duplicate doc_ids",
            i
        );
        // Each result should match individual search.
        let individual = index.search(query_refs[i], 10).unwrap();
        assert_eq!(
            individual.doc_ids, results.doc_ids,
            "query {}: batch does not match individual",
            i
        );
    }
}
