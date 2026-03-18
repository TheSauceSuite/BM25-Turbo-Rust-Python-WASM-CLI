//! BEIR Benchmark Harness for BM25 Turbo.
//!
//! Downloads BEIR datasets, builds BM25 indices, runs queries, measures
//! latency/QPS, computes nDCG@10, and outputs JSON results.
//!
//! # Usage
//! ```bash
//! cargo run -p benchmarks --release --bin beir_bench -- --datasets scifact
//! cargo run -p benchmarks --release --bin beir_bench -- --datasets scifact,fiqa --output results.json
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bm25_turbo_bench::dataset::{self, Dataset};
use bm25_turbo_bench::eval;
use bm25_turbo_bench::results::{BenchmarkRun, DatasetResult};

/// Maximum time allowed for a single dataset download before skipping.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// Per-query result: ranked doc IDs and optional relevance judgments.
type QueryResult<'a> = (Vec<String>, Option<&'a HashMap<String, u32>>);

fn main() {
    let args = parse_args();

    let mut dataset_results = Vec::new();

    for ds in &args.datasets {
        eprintln!("--- Evaluating {} ---", ds.name());

        match run_dataset(ds, &args) {
            Ok(result) => {
                eprintln!(
                    "  {} docs, {} queries, {:.1} ms index, {:.0} QPS, nDCG@10={:.4}",
                    result.num_docs,
                    result.num_queries,
                    result.index_time_ms,
                    result.queries_per_sec,
                    result.ndcg_at_10,
                );
                dataset_results.push(result);
            }
            Err(e) => {
                eprintln!("  SKIPPED: {}", e);
            }
        }
    }

    if dataset_results.is_empty() {
        eprintln!("No datasets were successfully evaluated.");
        std::process::exit(1);
    }

    let run = BenchmarkRun {
        timestamp: chrono_timestamp(),
        system_info: system_info(),
        rust_version: rust_version(),
        datasets: dataset_results,
    };

    // Print JSON to stdout
    let json = run.to_json().expect("failed to serialize results");
    println!("{json}");

    // Optionally write to file
    if let Some(ref path) = args.output {
        std::fs::write(path, &json).expect("failed to write output file");
        eprintln!("Results written to {}", path.display());
    }

    // Print human-readable summary to stderr
    eprintln!();
    eprintln!("{run}");
}

/// Parsed command-line arguments.
struct Args {
    datasets: Vec<Dataset>,
    cache_dir: PathBuf,
    output: Option<PathBuf>,
    k1: f32,
    b: f32,
    method: bm25_turbo::types::Method,
    max_queries: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut datasets_str = String::from("scifact");
    let mut cache_dir = PathBuf::from("./beir_cache");
    let mut output: Option<PathBuf> = None;
    let mut k1: f32 = 1.5;
    let mut b: f32 = 0.75;
    let mut method_str = String::from("lucene");
    let mut max_queries: usize = 5000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--datasets" => {
                i += 1;
                if i < args.len() {
                    datasets_str = args[i].clone();
                }
            }
            "--cache-dir" => {
                i += 1;
                if i < args.len() {
                    cache_dir = PathBuf::from(&args[i]);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = Some(PathBuf::from(&args[i]));
                }
            }
            "--k1" => {
                i += 1;
                if i < args.len() {
                    k1 = args[i].parse().expect("invalid --k1 value");
                }
            }
            "--b" => {
                i += 1;
                if i < args.len() {
                    b = args[i].parse().expect("invalid --b value");
                }
            }
            "--method" => {
                i += 1;
                if i < args.len() {
                    method_str = args[i].to_lowercase();
                }
            }
            "--max-queries" => {
                i += 1;
                if i < args.len() {
                    max_queries = args[i].parse().expect("invalid --max-queries value");
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let datasets: Vec<Dataset> = datasets_str
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| parse_dataset(s.trim()))
        .collect();

    let method = parse_method(&method_str);

    Args {
        datasets,
        cache_dir,
        output,
        k1,
        b,
        method,
        max_queries,
    }
}

fn print_usage() {
    eprintln!("Usage: beir_bench [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --datasets <list>   Comma-separated dataset names (default: scifact)");
    eprintln!("                      Available: nq, msmarco, scifact, fiqa");
    eprintln!("  --cache-dir <path>  Directory for cached datasets (default: ./beir_cache)");
    eprintln!("  --output <path>     Write JSON results to file (in addition to stdout)");
    eprintln!("  --k1 <float>        BM25 k1 parameter (default: 1.5)");
    eprintln!("  --b <float>         BM25 b parameter (default: 0.75)");
    eprintln!("  --method <name>     Scoring method: robertson, lucene, atire, bm25l, bm25plus");
    eprintln!("                      (default: lucene)");
    eprintln!("  --help, -h          Print this help message");
}

fn parse_dataset(name: &str) -> Dataset {
    match name.to_lowercase().as_str() {
        "nq" => Dataset::NQ,
        "msmarco" => Dataset::MSMARCO,
        "scifact" => Dataset::SciFact,
        "fiqa" => Dataset::FiQA,
        other => {
            eprintln!("Unknown dataset: {other}. Available: nq, msmarco, scifact, fiqa");
            std::process::exit(1);
        }
    }
}

fn parse_method(name: &str) -> bm25_turbo::types::Method {
    use bm25_turbo::types::Method;
    match name {
        "robertson" => Method::Robertson,
        "lucene" => Method::Lucene,
        "atire" => Method::Atire,
        "bm25l" => Method::Bm25l,
        "bm25plus" | "bm25+" => Method::Bm25Plus,
        other => {
            eprintln!("Unknown method: {other}. Available: robertson, lucene, atire, bm25l, bm25plus");
            std::process::exit(1);
        }
    }
}

/// Run a full evaluation on a single dataset.
fn run_dataset(ds: &Dataset, args: &Args) -> Result<DatasetResult, String> {
    // Download with timeout guard
    let download_start = Instant::now();
    let files = dataset::download_dataset(*ds, &args.cache_dir)?;
    let download_elapsed = download_start.elapsed();

    if download_elapsed > DOWNLOAD_TIMEOUT {
        return Err(format!(
            "download took {:.0}s, exceeding {:.0}s timeout",
            download_elapsed.as_secs_f64(),
            DOWNLOAD_TIMEOUT.as_secs_f64(),
        ));
    }

    // Parse corpus
    eprintln!("  Parsing corpus...");
    let corpus = dataset::parse_corpus(&files.corpus_path)?;
    let num_docs = corpus.texts.len() as u64;
    eprintln!("  {} documents loaded", num_docs);

    // Parse queries
    let queries = dataset::parse_queries(&files.queries_path)?;
    let num_queries = queries.texts.len() as u64;
    eprintln!("  {} queries loaded", num_queries);

    // Parse qrels
    let qrels = dataset::parse_qrels(&files.qrels_path)?;

    // Build corpus refs for BM25Builder
    let corpus_refs: Vec<&str> = corpus.texts.iter().map(|s| s.as_str()).collect();

    // Build BM25 index and measure time
    eprintln!("  Building index (method={}, k1={}, b={})...", args.method, args.k1, args.b);
    let index_start = Instant::now();
    let index = bm25_turbo::BM25Builder::new()
        .method(args.method)
        .k1(args.k1)
        .b(args.b)
        .build_from_corpus(&corpus_refs)
        .map_err(|e| format!("index build failed: {e}"))?;
    let index_time_ms = index_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Index built in {index_time_ms:.1} ms");

    // Build doc_id -> index mapping for result lookup
    let doc_id_to_idx: HashMap<&str, u32> = corpus
        .doc_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i as u32))
        .collect();

    // Reverse mapping: index -> doc_id
    let idx_to_doc_id: &[String] = &corpus.doc_ids;

    // Run queries, measuring per-query latency (capped by max_queries)
    let run_count = (num_queries as usize).min(args.max_queries);
    eprintln!("  Running {} queries (k=10, max={})...", run_count, args.max_queries);
    let mut query_latencies_us = Vec::with_capacity(run_count);
    let mut per_query_results: Vec<QueryResult<'_>> =
        Vec::with_capacity(run_count);

    for (qi, query_text) in queries.texts.iter().enumerate().take(run_count) {
        let query_start = Instant::now();
        let results = index
            .search(query_text, 10)
            .map_err(|e| format!("query {} failed: {e}", qi))?;
        let latency_us = query_start.elapsed().as_secs_f64() * 1_000_000.0;
        query_latencies_us.push(latency_us);

        // Map doc indices back to string IDs for nDCG evaluation
        let ranked_doc_ids: Vec<String> = results
            .doc_ids
            .iter()
            .map(|&idx| idx_to_doc_id[idx as usize].clone())
            .collect();

        let query_id = &queries.query_ids[qi];
        let query_qrels = qrels.0.get(query_id);

        per_query_results.push((ranked_doc_ids, query_qrels));
    }

    // Compute nDCG@10
    let eval_pairs: Vec<(Vec<String>, &HashMap<String, u32>)> = per_query_results
        .iter()
        .filter_map(|(ranked, qrels_opt)| {
            qrels_opt.map(|q| (ranked.clone(), q))
        })
        .collect();
    let ndcg_at_10 = eval::mean_ndcg(&eval_pairs, 10);

    // Compute latency percentiles
    query_latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = query_latencies_us.len();
    let latency_p50_us = if n > 0 { query_latencies_us[n / 2] } else { 0.0 };
    let latency_p95_us = if n > 0 { query_latencies_us[n * 95 / 100] } else { 0.0 };
    let latency_p99_us = if n > 0 { query_latencies_us[n * 99 / 100] } else { 0.0 };

    // Compute QPS
    let total_query_time_s: f64 = query_latencies_us.iter().sum::<f64>() / 1_000_000.0;
    let queries_per_sec = if total_query_time_s > 0.0 {
        num_queries as f64 / total_query_time_s
    } else {
        0.0
    };

    // We don't measure peak memory in this simple harness; report 0
    let _ = &doc_id_to_idx; // suppress unused warning

    Ok(DatasetResult {
        name: ds.name().to_string(),
        num_docs,
        num_queries: run_count as u64,
        index_time_ms,
        queries_per_sec,
        latency_p50_us,
        latency_p95_us,
        latency_p99_us,
        ndcg_at_10,
        peak_memory_bytes: 0,
    })
}

/// Generate an ISO 8601 timestamp without external dependencies.
fn chrono_timestamp() -> String {
    // Simple UTC timestamp using SystemTime
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    // Convert to approximate date-time (no leap seconds, good enough for benchmarks)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Compute year/month/day from days since epoch (1970-01-01)
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let year_days = if is_leap_year(y) { 366 } else { 365 };
        if remaining_days < year_days {
            break;
        }
        remaining_days -= year_days;
        y += 1;
    }
    let leap = is_leap_year(y);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut m = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md {
            m = i;
            break;
        }
        remaining_days -= md;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m + 1,
        remaining_days + 1,
        hours,
        minutes,
        seconds,
    )
}

fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

/// Collect basic system info string.
fn system_info() -> String {
    format!(
        "{} {} ({})",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::consts::FAMILY,
    )
}

/// Get the Rust version from the compiler that built this binary.
fn rust_version() -> String {
    // Embed at compile time
    format!(
        "rustc {} (edition {})",
        env!("CARGO_PKG_RUST_VERSION", "unknown"),
        "2024"
    )
}
