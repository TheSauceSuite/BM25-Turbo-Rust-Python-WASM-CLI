//! Criterion benchmarks for BEIR dataset indexing and querying.
//!
//! These benchmarks are gated behind the `BEIR_BENCH=1` environment variable
//! and require pre-cached datasets in `./beir_cache/`. They will NOT download
//! datasets during benchmark runs.
//!
//! # Usage
//! ```bash
//! # First, cache datasets:
//! cargo run -p benchmarks --release --bin beir_bench -- --datasets scifact
//!
//! # Then run benchmarks:
//! BEIR_BENCH=1 cargo bench -p benchmarks --bench beir
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::Path;

use bm25_turbo_bench::dataset::{self, Dataset};

/// All datasets we benchmark (when cached).
const DATASETS: &[Dataset] = &[
    Dataset::SciFact,
    Dataset::FiQA,
    Dataset::NQ,
    Dataset::MSMARCO,
];

/// Default cache directory (same as beir_bench binary).
const CACHE_DIR: &str = "./beir_cache";

/// Check if BEIR benchmarks are enabled and datasets are cached.
fn should_run() -> bool {
    std::env::var("BEIR_BENCH").map(|v| v == "1").unwrap_or(false)
}

/// Get list of datasets that are already cached (no downloads).
fn cached_datasets() -> Vec<Dataset> {
    let cache_path = Path::new(CACHE_DIR);
    DATASETS
        .iter()
        .filter(|ds| {
            let dataset_dir = cache_path.join(ds.name());
            dataset_dir.join("corpus.jsonl").exists()
                && dataset_dir.join("queries.jsonl").exists()
        })
        .copied()
        .collect()
}

/// Pre-loaded dataset for benchmarking.
struct LoadedDataset {
    dataset: Dataset,
    corpus_texts: Vec<String>,
    query_texts: Vec<String>,
}

fn load_dataset(ds: Dataset) -> Option<LoadedDataset> {
    let cache_path = Path::new(CACHE_DIR);
    let files = dataset::download_dataset(ds, cache_path).ok()?;
    let corpus = dataset::parse_corpus(&files.corpus_path).ok()?;
    let queries = dataset::parse_queries(&files.queries_path).ok()?;
    Some(LoadedDataset {
        dataset: ds,
        corpus_texts: corpus.texts,
        query_texts: queries.texts,
    })
}

fn bench_beir_index(c: &mut Criterion) {
    if !should_run() {
        eprintln!("BEIR benchmarks skipped: set BEIR_BENCH=1 to enable");
        return;
    }

    let datasets = cached_datasets();
    if datasets.is_empty() {
        eprintln!("BEIR benchmarks skipped: no cached datasets found in {CACHE_DIR}");
        return;
    }

    let mut group = c.benchmark_group("beir_index");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    for ds in &datasets {
        if let Some(loaded) = load_dataset(*ds) {
            let corpus_refs: Vec<&str> = loaded.corpus_texts.iter().map(|s| s.as_str()).collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(loaded.dataset.name()),
                &corpus_refs,
                |b, corpus| {
                    b.iter(|| {
                        bm25_turbo::BM25Builder::new()
                            .method(bm25_turbo::types::Method::Lucene)
                            .k1(1.5)
                            .b(0.75)
                            .build_from_corpus(corpus)
                            .expect("index build failed")
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_beir_query(c: &mut Criterion) {
    if !should_run() {
        return;
    }

    let datasets = cached_datasets();
    if datasets.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("beir_query");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    for ds in &datasets {
        if let Some(loaded) = load_dataset(*ds) {
            let corpus_refs: Vec<&str> = loaded.corpus_texts.iter().map(|s| s.as_str()).collect();
            let index = bm25_turbo::BM25Builder::new()
                .method(bm25_turbo::types::Method::Lucene)
                .k1(1.5)
                .b(0.75)
                .build_from_corpus(&corpus_refs)
                .expect("index build failed");

            let queries: Vec<&str> = loaded.query_texts.iter().map(|s| s.as_str()).collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(loaded.dataset.name()),
                &queries,
                |b, queries| {
                    b.iter(|| {
                        for query in queries.iter() {
                            let _ = index.search(query, 10);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_beir_index, bench_beir_query);
criterion_main!(benches);
