//! Criterion benchmarks for BM25 query latency.
//!
//! Builds a pre-indexed corpus of 10K documents (fixed seed for
//! reproducibility) and benchmarks single-token, multi-token queries,
//! and varying k values.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Generate a synthetic corpus of `n` documents with Zipf-like word distribution.
fn generate_corpus(n: usize, seed: u64) -> Vec<String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Build a fixed vocabulary.
    let vocab: Vec<String> = (0..5_000)
        .map(|i| {
            let suffix: u32 = rng.gen_range(0..100_000);
            format!("word{}x{}", i, suffix)
        })
        .collect();

    (0..n)
        .map(|_| {
            let doc_len = rng.gen_range(20..200);
            let tokens: Vec<&str> = (0..doc_len)
                .map(|_| {
                    let u: f64 = rng.r#gen();
                    let idx = ((u * u) * vocab.len() as f64) as usize;
                    vocab[idx.min(vocab.len() - 1)].as_str()
                })
                .collect();
            tokens.join(" ")
        })
        .collect()
}

/// Build the shared index used across query benchmarks.
fn build_index() -> bm25_turbo::BM25Index {
    let corpus = generate_corpus(10_000, 42);
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    bm25_turbo::BM25Builder::new()
        .build_from_corpus(&corpus_refs)
        .expect("build failed")
}

/// Pick a single high-frequency token from the corpus (deterministic).
fn single_token_query() -> &'static str {
    // word0 is the most frequent due to Zipf distribution.
    // We use a literal string that will be present after tokenization.
    "word0x"
}

/// Pick a multi-token query (3 terms).
fn multi_token_query() -> &'static str {
    "word0x word1x word2x"
}

fn bench_query_single_token(c: &mut Criterion) {
    let index = build_index();

    // Find a real term from the index vocab.
    let query = single_token_query();

    c.bench_function("query_single_token_k10", |b| {
        b.iter(|| {
            let _ = index.search(query, 10);
        });
    });
}

fn bench_query_multi_token(c: &mut Criterion) {
    let index = build_index();
    let query = multi_token_query();

    c.bench_function("query_multi_token_k10", |b| {
        b.iter(|| {
            let _ = index.search(query, 10);
        });
    });
}

fn bench_query_varying_k(c: &mut Criterion) {
    let index = build_index();
    let query = multi_token_query();

    let mut group = c.benchmark_group("query_varying_k");
    for k in [1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let _ = index.search(query, k);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_query_single_token, bench_query_multi_token, bench_query_varying_k);
criterion_main!(benches);
