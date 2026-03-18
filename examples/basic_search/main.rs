//! Minimal BM25 Turbo example: build an index and search it.
//!
//! Run: cargo run --manifest-path examples/basic_search/Cargo.toml

use bm25_turbo::{BM25Builder, Method};

fn main() {
    // Sample corpus of 5 documents.
    let corpus: &[&str] = &[
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces the fox",
        "the lazy cat sleeps all day long",
        "brown bears eat fish in the river",
        "the fox and the dog are friends",
    ];

    // All 5 BM25 scoring variants.
    let methods = [
        ("Robertson", Method::Robertson),
        ("Lucene", Method::Lucene),
        ("ATIRE", Method::Atire),
        ("BM25L", Method::Bm25l),
        ("BM25+", Method::Bm25Plus),
    ];

    let query = "quick brown fox";

    for (name, method) in &methods {
        let index = BM25Builder::new()
            .method(*method)
            .build_from_corpus(corpus)
            .expect("failed to build index");

        let results = index.search(query, 3).expect("search failed");

        println!("[{name}] query: \"{query}\"");
        for (doc_id, score) in results.doc_ids.iter().zip(results.scores.iter()) {
            println!("  doc {doc_id}: {score:.4}  \"{}\"", corpus[*doc_id as usize]);
        }
        println!();
    }
}
