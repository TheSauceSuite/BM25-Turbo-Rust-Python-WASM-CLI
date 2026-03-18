//! `bm25-turbo bench` subcommand — run built-in micro-benchmarks.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use bm25_turbo::{BM25Builder, Method};

#[derive(Args)]
pub struct BenchArgs {
    /// Path to a JSONL corpus file to benchmark against.
    /// If not provided, generates a synthetic corpus.
    #[arg(short, long)]
    pub corpus: Option<PathBuf>,

    /// Number of documents to generate if no corpus provided.
    #[arg(long, default_value = "10000")]
    pub num_docs: usize,

    /// Number of queries to run.
    #[arg(long, default_value = "1000")]
    pub num_queries: usize,

    /// Top-k results per query.
    #[arg(long, default_value = "10")]
    pub top_k: usize,

    /// BM25 variant to use.
    #[arg(long, default_value = "lucene")]
    pub method: String,

    /// Output format: json or table.
    #[arg(long, default_value = "table")]
    pub format: String,
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    let method = match args.method.to_lowercase().as_str() {
        "robertson" => Method::Robertson,
        "lucene" => Method::Lucene,
        "atire" => Method::Atire,
        "bm25l" => Method::Bm25l,
        "bm25+" | "bm25plus" => Method::Bm25Plus,
        other => anyhow::bail!("Unknown method: {other}. Use: robertson, lucene, atire, bm25l, bm25+"),
    };

    // Generate or load corpus
    let corpus: Vec<String> = if let Some(path) = &args.corpus {
        eprintln!("Loading corpus from {}...", path.display());
        let content = std::fs::read_to_string(path)?;
        content.lines().filter_map(|line| {
            let line = line.trim();
            if line.is_empty() { return None; }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                v.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
            } else {
                Some(line.to_string())
            }
        }).collect()
    } else {
        eprintln!("Generating synthetic corpus ({} documents)...", args.num_docs);
        generate_synthetic_corpus(args.num_docs)
    };

    let num_docs = corpus.len();
    eprintln!("Corpus: {} documents", num_docs);

    // Benchmark indexing
    eprintln!("Benchmarking indexing ({:?})...", method);
    let index_start = Instant::now();
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let index = BM25Builder::new()
        .method(method)
        .build_from_corpus(&corpus_refs)?;
    let index_duration = index_start.elapsed();

    let index_ms = index_duration.as_secs_f64() * 1000.0;
    let docs_per_sec = num_docs as f64 / index_duration.as_secs_f64();

    // Generate queries from corpus tokens
    let queries = generate_queries_from_corpus(&corpus, args.num_queries);

    // Benchmark querying
    eprintln!("Benchmarking {} queries (top-{})...", queries.len(), args.top_k);
    let mut latencies: Vec<f64> = Vec::with_capacity(queries.len());

    for query in &queries {
        let start = Instant::now();
        let _ = index.search(query, args.top_k);
        latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0); // microseconds
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);
    let p99 = percentile(&latencies, 99.0);
    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let qps = 1_000_000.0 / avg;

    // Output results
    if args.format == "json" {
        let result = serde_json::json!({
            "corpus_size": num_docs,
            "method": format!("{:?}", method),
            "index_time_ms": index_ms,
            "docs_per_sec": docs_per_sec,
            "num_queries": queries.len(),
            "top_k": args.top_k,
            "qps": qps,
            "latency_p50_us": p50,
            "latency_p95_us": p95,
            "latency_p99_us": p99,
            "latency_avg_us": avg,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!();
        println!("BM25 Turbo Benchmark Results");
        println!("============================");
        println!();
        println!("Corpus:       {} documents", num_docs);
        println!("Method:       {:?}", method);
        println!("Top-k:        {}", args.top_k);
        println!();
        println!("Indexing:");
        println!("  Time:       {:.1} ms", index_ms);
        println!("  Throughput: {:.0} docs/sec", docs_per_sec);
        println!();
        println!("Querying ({} queries):", queries.len());
        println!("  QPS:        {:.0}", qps);
        println!("  P50:        {:.1} us", p50);
        println!("  P95:        {:.1} us", p95);
        println!("  P99:        {:.1} us", p99);
        println!("  Avg:        {:.1} us", avg);
    }

    Ok(())
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn generate_synthetic_corpus(num_docs: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let words = [
        "search", "engine", "algorithm", "index", "query", "document", "ranking",
        "score", "term", "frequency", "inverse", "retrieval", "information",
        "text", "processing", "natural", "language", "model", "vector", "sparse",
        "matrix", "column", "compressed", "binary", "storage", "memory", "cache",
        "parallel", "concurrent", "thread", "performance", "benchmark", "latency",
        "throughput", "optimization", "fast", "efficient", "scalable", "distributed",
        "system", "library", "function", "module", "interface", "protocol", "server",
        "client", "request", "response", "data", "structure", "implementation",
    ];

    (0..num_docs).map(|i| {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let seed = hasher.finish();

        let doc_len = 10 + (seed % 40) as usize;
        let mut doc = String::with_capacity(doc_len * 8);
        for j in 0..doc_len {
            if j > 0 { doc.push(' '); }
            let word_idx = ((seed.wrapping_mul(j as u64 + 1).wrapping_add(i as u64)) % words.len() as u64) as usize;
            doc.push_str(words[word_idx]);
        }
        doc
    }).collect()
}

fn generate_queries_from_corpus(corpus: &[String], num_queries: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let all_words: Vec<&str> = corpus.iter()
        .take(1000)
        .flat_map(|doc| doc.split_whitespace())
        .collect();

    if all_words.is_empty() {
        return vec!["search".to_string(); num_queries];
    }

    (0..num_queries).map(|i| {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let seed = hasher.finish();

        let query_len = 1 + (seed % 4) as usize;
        let mut query = String::new();
        for j in 0..query_len {
            if j > 0 { query.push(' '); }
            let word_idx = ((seed.wrapping_mul(j as u64 + 1)) % all_words.len() as u64) as usize;
            query.push_str(all_words[word_idx]);
        }
        query
    }).collect()
}
