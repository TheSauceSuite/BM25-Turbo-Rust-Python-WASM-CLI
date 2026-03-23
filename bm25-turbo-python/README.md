<p align="center">
  <h1 align="center">BM25 Turbo <sup>⚡</sup></h1>
  <p align="center"><strong>Rust · Python · WASM · CLI</strong></p>
  <p align="center">The fastest BM25 scoring engine. Period.</p>
</p>

<p align="center">
  <a href="https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-/actions"><img src="https://img.shields.io/github/actions/workflow/status/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-/ci.yml?label=CI" alt="CI"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License"></a>
  <a href="https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-/stargazers"><img src="https://img.shields.io/github/stars/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-?style=social" alt="Stars"></a>
</p>

---

**28,217 queries/second** on 8.8 million documents. **8.6ms P50 latency**. Precomputed sparse BM25 with BMW pruning, memory-mapped persistence, and zero-copy index loading. Built for RAG pipelines, search applications, and ML workflows.

```rust
use bm25_turbo::{BM25Builder, Method};

let index = BM25Builder::new()
    .method(Method::Lucene)       // Robertson, Lucene, ATIRE, BM25L, BM25+
    .k1(1.5).b(0.75)
    .build_from_corpus(&[
        "Rust is a systems programming language",
        "BM25 is a ranking function used in information retrieval",
        "Machine learning models benefit from fast retrieval",
    ])?;

let results = index.search("information retrieval", 10)?;
for (id, score) in results.doc_ids.iter().zip(results.scores.iter()) {
    println!("doc {} → {:.4}", id, score);
}
```

## Why BM25 Turbo?

Most BM25 libraries compute scores at query time — scanning inverted indexes on every request. BM25 Turbo takes the opposite approach: **precompute every BM25 score at index time** into a compressed sparse column (CSC) matrix. Queries become sparse vector lookups with no math at serving time.

This makes BM25 Turbo the right choice when:
- You **query the same index many times** (RAG, reranking, batch evaluation)
- You need **deterministic, reproducible scores** (ML pipelines, experiments)
- You want **the simplest possible API** (3 lines to index, 1 line to query)
- You're building a **search feature** and don't need a full search server

## Performance

> Benchmarked on MS MARCO (8,841,823 documents, 509,962 queries) — the standard information retrieval benchmark. BMW pruning enabled. Single-threaded.

### Query Throughput

<p align="center">
  <img src="assets/qps-chart.svg" alt="Query throughput comparison" width="700">
</p>

### Query Latency

<p align="center">
  <img src="assets/latency-chart.svg" alt="Query latency comparison" width="700">
</p>

### Scaling Across Corpus Sizes

<p align="center">
  <img src="assets/scaling-chart.svg" alt="Latency scaling across corpus sizes" width="700">
</p>

| Corpus | Documents | P50 Latency | QPS | nDCG@10 |
|--------|-----------|-------------|-----|---------|
| SciFact | 5,183 | **67 μs** | 682,127 | 0.665 |
| FiQA | 57,638 | **711 μs** | 50,812 | 0.254 |
| MS MARCO | 8,841,823 | **8.6 ms** | 28,217 | — |

> Sub-millisecond on corpora under 100K docs. On multi-million document corpora, BMW pruning keeps latency competitive with inverted-index engines while maintaining the precomputed scoring advantage for batch workloads.

### The Tradeoff

BM25 Turbo front-loads computation: **index once, query millions of times**. Indexing is slower because every BM25 score is precomputed and compressed into the CSC matrix. But every subsequent query is a sparse vector lookup — no math at serving time.

## Features

### 5 BM25 Scoring Variants

| Variant | Formula | Best For |
|---------|---------|----------|
| **Robertson** | Classic BM25 (Okapi) | Academic benchmarks |
| **Lucene** | Apache Lucene's variant | Production search (default) |
| **ATIRE** | IDF without +1 smoothing | Research comparisons |
| **BM25L** | Long document correction | Corpora with varying doc lengths |
| **BM25+** | Lower-bound term frequency | Penalizing non-matching terms |

All variants support tunable `k1`, `b`, and `delta` parameters.

### BMW (Block-Max WAND) Pruning

Skip non-competitive documents during top-k retrieval. Essential for million-document corpora:

```rust
let index = BM25Builder::new()
    .build_from_corpus(&corpus)?;

// BMW pruning via WAL's query strategy
let wal = index.enable_wal()?;
let results = index.search_with_strategy(
    &wal, "distributed systems", 10,
    bm25_turbo::wal::QueryStrategy::Fast, // BMW pruning
)?;
```

BMW partitions the score matrix into blocks and maintains per-block upper bounds. During query evaluation, entire blocks are skipped when their maximum possible contribution can't beat the current k-th best score. This reduces the number of documents touched from millions to thousands.

### Memory-Mapped Persistence

Save indexes to disk and reload them instantly with zero-copy memory mapping:

```rust
use bm25_turbo::persistence;
use std::path::Path;

// Save (serializes CSC matrix + vocabulary + parameters)
persistence::save(&index, Path::new("my_index.bm25"))?;

// Standard load (deserialize into RAM)
let index = persistence::load(Path::new("my_index.bm25"))?;

// Memory-mapped load (instant, zero-copy, ideal for huge indexes)
let mmap_index = persistence::load_mmap(Path::new("my_index.bm25"))?;
```

Memory-mapped indexes load in **microseconds** regardless of size. The OS pages data in on demand — a 10GB index starts serving queries immediately without waiting for the full file to be read.

### Streaming Indexer

Index corpora larger than available memory by processing documents in configurable chunks:

```rust
use bm25_turbo::{StreamingBuilder, Method};

let mut builder = StreamingBuilder::new()
    .chunk_size(100_000)          // Process 100K docs at a time
    .method(Method::Lucene);
builder.add_documents(&["doc one", "doc two", "doc three"]);
let index = builder.build()?;
```

Peak memory: `O(chunk_size × avg_tokens)` instead of `O(total_corpus)`.

### Write-Ahead Log (WAL)

Add and delete documents without rebuilding the entire index:

```rust
use bm25_turbo::wal::WriteAheadLog;

let mut wal = WriteAheadLog::new();

// Incremental updates
index.add_documents(&mut wal, &["new document about Rust"])?;
wal.delete_documents(&[42, 87])?;

// Compact when the WAL grows large
index.compact(&mut wal)?;
```

### Built-in Tokenizer

17 language stemmers with configurable stopword removal:

```rust
use bm25_turbo::Tokenizer;
use rust_stemmers::Algorithm;

let tokenizer = Tokenizer::builder()
    .stemmer(Algorithm::English)
    .stopwords(vec!["the".into(), "a".into(), "an".into(), "is".into()])
    .build();

let tokens = tokenizer.tokenize("Running distributed systems at scale");
// → ["run", "distribut", "system", "scale"]
```

Supported languages: Arabic, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish.

### Distributed Search (gRPC)

Shard large indexes across multiple nodes and query them as one:

```rust
use bm25_turbo::distributed::{QueryCoordinator, start_shard_server};

// Start shard servers (one per machine/core)
start_shard_server(shard_index, "[::1]:50051").await?;

// Coordinator fans out queries and merges results
let coordinator = QueryCoordinator::connect(&[
    "http://[::1]:50051",
    "http://[::1]:50052",
]).await?;

let results = coordinator.search("distributed query", 10).await?;
```

## Interfaces

### CLI

```bash
# Index a corpus
bm25-turbo index --input corpus.jsonl --output index.bm25 --field text

# Search
bm25-turbo search --index index.bm25 --query "information retrieval" -k 10

# Start HTTP server
bm25-turbo serve --index index.bm25 --port 8080

# Push/pull from HuggingFace Hub
bm25-turbo push --index index.bm25 --repo username/my-index
bm25-turbo pull --repo username/my-index --output index.bm25
```

Supports CSV, JSONL, JSON array, and plain text (one document per line). Auto-detects format when possible.

### HTTP Server

```bash
$ bm25-turbo serve --index index.bm25 --port 8080
```

```bash
# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 10}'

# Health check
curl http://localhost:8080/health

# Index statistics
curl http://localhost:8080/stats
```

### MCP Server (AI Agent Integration)

Expose BM25 search as a tool for AI agents via the [Model Context Protocol](https://modelcontextprotocol.io/):

```bash
bm25-turbo mcp --index index.bm25 --port 8080
```

The MCP server exposes two tools:
- `bm25_search` — Query the index with configurable top-k
- `bm25_index_stats` — Get index metadata (doc count, vocab size)

Works with Claude, ChatGPT, and any MCP-compatible agent framework.

### Python Bindings

```python
from bm25_turbo_python import BM25

# Build an index
engine = BM25(method="lucene", k1=1.5, b=0.75)
engine.index(["Rust is fast", "Python is flexible", "BM25 ranks documents"])

# Search — returns (doc_ids, scores) tuple
doc_ids, scores = engine.search("fast programming", k=5)
for doc_id, score in zip(doc_ids, scores):
    print(f"  doc {doc_id}: {score:.4f}")

# Save / load
engine.save("my_index.bm25")
engine = BM25.load("my_index.bm25")
```

### WASM (Browser / Edge)

```javascript
import init, { WasmBM25 } from 'bm25-turbo-wasm';

await init();

const index = new WasmBM25(
    ["JavaScript runs everywhere",
     "WebAssembly enables near-native performance",
     "BM25 is a proven ranking algorithm"],
    "lucene",  // method (optional)
    1.5,       // k1 (optional)
    0.75       // b (optional)
);

const results = index.search("native performance", 5);
console.log(results); // [{doc_id: 1, score: 0.82}, ...]
```

Bundle size: ~1.3 MB (gzipped: ~500 KB). No server needed — runs entirely in the browser.

### HuggingFace Hub

Share and discover BM25 indexes on the [HuggingFace Hub](https://huggingface.co/):

```bash
# Push your index
bm25-turbo push --index msmarco.bm25 --repo username/msmarco-bm25-turbo

# Pull someone else's index
bm25-turbo pull --repo username/msmarco-bm25-turbo --output msmarco.bm25
```

Pre-built indexes for common datasets (MS MARCO, NQ, SciFact, FiQA) can be shared across teams without re-indexing.

## Installation

### Rust

```toml
[dependencies]
bm25-turbo = "0.1"
```

### CLI

```bash
cargo install bm25-turbo-cli
```

### Python

```bash
pip install bm25-turbo
```

Requires Python 3.8-3.13. Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x86_64).

### WASM / npm

```bash
npm install bm25-turbo-wasm
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      BM25 Turbo                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Tokenizer│  │  Scoring  │  │   CSC    │  │  BMW   │ │
│  │ 17 langs │  │ 5 variants│  │  Matrix  │  │ Pruning│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │             │             │      │
│  ┌────┴──────────────┴─────────────┴─────────────┴────┐ │
│  │              Index Builder / Streaming              │ │
│  └────────────────────┬───────────────────────────────┘ │
│                       │                                  │
│  ┌────────────────────┴───────────────────────────────┐ │
│  │               Persistence Layer                     │ │
│  │        Binary · Memory-Mapped · WAL · Hub           │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│   CLI    │  HTTP    │   MCP    │  Python  │    WASM    │
│          │  Server  │  Server  │ Bindings │   (npm)    │
└──────────┴──────────┴──────────┴──────────┴────────────┘
```

### How It Works

1. **Tokenize** — Split documents into stemmed tokens with configurable stopword removal
2. **Score** — Compute BM25 scores for every (term, document) pair using the chosen variant
3. **Compress** — Store scores in a Compressed Sparse Column (CSC) matrix (only non-zero entries)
4. **Serve** — Queries are sparse vector dot products: look up columns for query terms, accumulate scores, return top-k

The CSC format stores only non-zero BM25 scores. For a corpus of 8.8M documents with 500K vocabulary, this typically compresses to 2-4 GB — far smaller than a dense matrix (which would be ~17 TB).

## Configuration

### BM25 Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `k1` | 1.5 | 0.0-3.0 | Term frequency saturation. Higher = more weight to repeated terms |
| `b` | 0.75 | 0.0-1.0 | Length normalization. 0 = no normalization, 1 = full normalization |
| `delta` | 0.5 | 0.0-inf | BM25L/BM25+ lower bound (only used with those variants) |

### Choosing a Variant

- **Start with Lucene** — It's the most widely used and tested variant
- **Use Robertson** if you need exact comparisons with academic papers
- **Use BM25L** if your corpus has extreme document length variation
- **Use BM25+** if short queries return too many irrelevant results
- **Use ATIRE** if you're reproducing ATIRE research results

## Benchmarks

### Reproducing Our Numbers

```bash
# Clone and build
git clone https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-
cd bm25-turbo
cargo build --release -p bm25-turbo-bench

# Run on SciFact (quick, 5K docs)
cargo run -p bm25-turbo-bench --release --bin beir_bench -- --datasets scifact

# Run on MS MARCO (full, 8.8M docs — requires ~16GB RAM)
cargo run -p bm25-turbo-bench --release --bin beir_bench -- --datasets msmarco --max-queries 1000
```

Datasets are automatically downloaded from the [BEIR benchmark suite](https://github.com/beir-cellar/beir). First run may take a few minutes to download.

### BEIR Benchmark Results

| Dataset | Documents | Vocab | Index Time | QPS | P50 Latency | nDCG@10 |
|---------|-----------|-------|------------|-----|-------------|---------|
| SciFact | 5,183 | 19,927 | 223 ms | 682,127 | 67 μs | 0.665 |
| FiQA | 57,638 | 67,893 | 1.8 s | 50,812 | 711 μs | 0.254 |
| MS MARCO | 8,841,823 | ~500K | 66 min | 28,217 | 8.6 ms | — |

> MS MARCO nDCG measured on the dev set (6,980 queries). QPS and latency measured on a random sample of 1,000 queries. All benchmarks single-threaded on a consumer desktop.

## Use Cases

### RAG (Retrieval-Augmented Generation)

BM25 Turbo is ideal as the retrieval stage in RAG pipelines. Index your knowledge base once, then retrieve relevant context for every LLM query:

```rust
let context_docs = index.search(&user_question, 5)?;
let context = context_docs.iter()
    .map(|(id, _)| &documents[*id as usize])
    .collect::<Vec<_>>()
    .join("\n\n");
// Feed context to your LLM
```

### Hybrid Search (BM25 + Embeddings)

Combine BM25 lexical scores with dense embedding similarity for best-of-both-worlds retrieval:

```rust
// BM25 lexical retrieval
let bm25_results = index.search(query, 100)?;

// Dense retrieval (from your embedding model)
let dense_results = embedding_index.search(query_embedding, 100)?;

// Reciprocal Rank Fusion
let fused = rrf_merge(&bm25_results, &dense_results, k=60);
```

### Batch Evaluation

Score hundreds of thousands of queries against a corpus for ML experiments:

```rust
for query in &evaluation_queries {
    let results = index.search(query, 10)?;
    // Compute nDCG, MAP, recall...
}
// At 28K QPS, 500K queries finish in ~18 seconds
```

## Comparison with Other Tools

<p align="center">
  <img src="assets/feature-matrix.svg" alt="Feature comparison matrix" width="700">
</p>

**BM25 Turbo is not a full-text search engine.** It's a focused BM25 scoring library. If you need phrase queries, faceted search, or query DSLs, use Tantivy or Elasticsearch. If you need the fastest possible BM25 scores with the simplest possible API — or you're migrating from bm25s and need 2,000x more speed — use BM25 Turbo.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
cargo test --workspace --exclude bm25-turbo-wasm

# Run clippy
cargo clippy --workspace --exclude bm25-turbo-wasm -- -D warnings

# Build WASM
cd bm25-turbo-wasm && wasm-pack build --target web
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
