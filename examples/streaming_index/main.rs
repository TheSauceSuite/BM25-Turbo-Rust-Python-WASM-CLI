//! Streaming index example: process documents in chunks using StreamingBuilder.
//!
//! This demonstrates how to index a large corpus without loading it all into
//! memory at once. Documents are fed in batches via `add_documents()`.
//!
//! Run: cargo run --manifest-path examples/streaming_index/Cargo.toml

use std::time::Instant;

use bm25_turbo::StreamingBuilder;

fn main() {
    // Sample data: simulate a corpus that arrives in chunks.
    let data: &[&str] = &[
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces the fox",
        "the lazy cat sleeps all day long",
        "brown bears eat fish in the river",
        "the fox and the dog are friends",
        "rivers flow through the brown mountains",
        "a lazy afternoon by the riverside",
        "foxes are clever animals in the wild",
        "the dog chased the cat around the yard",
        "quick thinking saves the day",
        "bears hibernate during the cold winter months",
        "fish swim upstream to spawn in spring",
        "the mountain trail is steep and winding",
        "wild animals roam the forest at night",
        "a clever fox outwits the hound",
        "the river is wide and deep near the delta",
        "cats and dogs can be the best of friends",
        "the afternoon sun warms the lazy garden",
        "brown trout are prized by fly fishers",
        "jumping over obstacles requires quick reflexes",
    ];

    let start = Instant::now();

    // Create a streaming builder with a small chunk size to demonstrate chunking.
    let mut builder = StreamingBuilder::new().chunk_size(5);

    // Feed documents in batches of 7 (simulating chunked I/O).
    let batch_size = 7;
    for (i, chunk) in data.chunks(batch_size).enumerate() {
        builder.add_documents(chunk);
        println!("Added batch {}: {} documents", i + 1, chunk.len());
    }

    // Build the final index (merges all chunks).
    let index = builder.build().expect("failed to build streaming index");
    let elapsed = start.elapsed();

    println!(
        "\nIndex built in {:.2}ms: {} docs, {} terms",
        elapsed.as_secs_f64() * 1000.0,
        index.num_docs(),
        index.vocab_size()
    );

    // Search the index.
    let queries = ["quick brown fox", "lazy cat", "river fish", "wild animals"];
    for query in &queries {
        let results = index.search(query, 3).expect("search failed");
        println!("\nQuery: \"{query}\"");
        for (doc_id, score) in results.doc_ids.iter().zip(results.scores.iter()) {
            println!("  doc {doc_id}: {score:.4}  \"{}\"", data[*doc_id as usize]);
        }
    }
}
