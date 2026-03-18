//! Persistence example: save an index to disk, load it back, and verify results match.
//!
//! Run: cargo run --manifest-path examples/persistence/Cargo.toml

use std::path::Path;

use bm25_turbo::persistence;
use bm25_turbo::BM25Builder;

fn main() {
    // Build an index from a sample corpus.
    let corpus: &[&str] = &[
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces the fox",
        "the lazy cat sleeps all day long",
        "brown bears eat fish in the river",
        "the fox and the dog are friends",
    ];

    let index = BM25Builder::new()
        .build_from_corpus(corpus)
        .expect("failed to build index");

    // Save to a temporary file.
    let path = Path::new("example_index.bm25");
    persistence::save(&index, path).expect("failed to save index");

    // Show file size on disk.
    let metadata = std::fs::metadata(path).expect("failed to read file metadata");
    println!("Index saved to {:?} ({} bytes)", path, metadata.len());

    // Load the index back from disk.
    let loaded = persistence::load(path).expect("failed to load index");
    println!(
        "Loaded index: {} docs, {} terms",
        loaded.num_docs(),
        loaded.vocab_size()
    );

    // Search both indices and verify results match.
    let query = "quick brown fox";
    let original = index.search(query, 3).expect("search failed");
    let restored = loaded.search(query, 3).expect("search failed on loaded index");

    println!("\nQuery: \"{query}\"");
    for (doc_id, score) in original.doc_ids.iter().zip(original.scores.iter()) {
        println!("  [original] doc {doc_id}: {score:.4}");
    }
    for (doc_id, score) in restored.doc_ids.iter().zip(restored.scores.iter()) {
        println!("  [restored] doc {doc_id}: {score:.4}");
    }

    assert_eq!(original.doc_ids, restored.doc_ids, "doc_ids must match");
    for (a, b) in original.scores.iter().zip(restored.scores.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "scores must be bit-identical");
    }
    println!("\nResults match -- persistence round-trip verified.");

    // Clean up.
    std::fs::remove_file(path).ok();
}
