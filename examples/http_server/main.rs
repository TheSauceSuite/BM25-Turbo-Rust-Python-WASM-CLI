//! HTTP server example: wrap a BM25 index in an Axum REST API.
//!
//! Run: cargo run --manifest-path examples/http_server/Cargo.toml
//!
//! Test with curl:
//!   curl http://localhost:3000/health
//!   curl -X POST http://localhost:3000/search \
//!        -H "Content-Type: application/json" \
//!        -d '{"query": "quick brown fox", "top_k": 3}'

use std::sync::Arc;
use std::time::Instant;

use bm25_turbo::server::{router, AppState};
use bm25_turbo::BM25Builder;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    // Build an index from a sample corpus.
    let corpus: &[&str] = &[
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces the fox",
        "the lazy cat sleeps all day long",
        "brown bears eat fish in the river",
        "the fox and the dog are friends",
        "rivers flow through the brown mountains",
        "a lazy afternoon by the riverside",
        "foxes are clever animals in the wild",
    ];

    let index = BM25Builder::new()
        .build_from_corpus(corpus)
        .expect("failed to build index");

    println!(
        "Index built: {} docs, {} terms",
        index.num_docs(),
        index.vocab_size()
    );

    // Create shared application state.
    let state = Arc::new(AppState {
        index: RwLock::new(Arc::new(index)),
        started_at: Instant::now(),
        approximate: false,
    });

    // Build the router using BM25 Turbo's built-in server module.
    let app = router(state);

    // Start the server.
    let addr = "0.0.0.0:3000";
    println!("Listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    axum::serve(listener, app).await.expect("server error");
}
