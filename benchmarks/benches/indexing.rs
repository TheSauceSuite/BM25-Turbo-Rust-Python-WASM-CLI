//! Criterion benchmarks for BM25 indexing throughput.
//!
//! Generates synthetic corpora with a Zipf-like word distribution using
//! a fixed seed (`ChaCha8Rng`, seed 42) for reproducibility.

use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Fixed vocabulary of 5 000 synthetic "words".
fn build_vocab(rng: &mut ChaCha8Rng, size: usize) -> Vec<String> {
    (0..size)
        .map(|i| {
            // Mix deterministic prefix with a random suffix for realism.
            let suffix: u32 = rng.gen_range(0..100_000);
            format!("word{}x{}", i, suffix)
        })
        .collect()
}

/// Generate a corpus of `n` documents.  Each document has 20-200 tokens
/// drawn with a Zipf-like bias (lower-index words are more frequent).
fn generate_corpus(n: usize, seed: u64) -> Vec<String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vocab = build_vocab(&mut rng, 5_000);

    (0..n)
        .map(|_| {
            let doc_len = rng.gen_range(20..200);
            let tokens: Vec<&str> = (0..doc_len)
                .map(|_| {
                    // Zipf-like: square the uniform random to bias toward lower indices.
                    let u: f64 = rng.r#gen();
                    let idx = ((u * u) * vocab.len() as f64) as usize;
                    vocab[idx.min(vocab.len() - 1)].as_str()
                })
                .collect();
            tokens.join(" ")
        })
        .collect()
}

fn bench_index_1k(c: &mut Criterion) {
    let corpus = generate_corpus(1_000, 42);
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

    c.bench_function("index_1k_docs", |b| {
        b.iter(|| {
            bm25_turbo::BM25Builder::new()
                .build_from_corpus(&corpus_refs)
                .expect("build failed")
        });
    });
}

fn bench_index_10k(c: &mut Criterion) {
    let corpus = generate_corpus(10_000, 42);
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

    c.bench_function("index_10k_docs", |b| {
        b.iter(|| {
            bm25_turbo::BM25Builder::new()
                .build_from_corpus(&corpus_refs)
                .expect("build failed")
        });
    });
}

criterion_group!(benches, bench_index_1k, bench_index_10k);
criterion_main!(benches);
