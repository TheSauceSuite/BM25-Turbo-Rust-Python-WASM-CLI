//! Phase 6 integration tests for the BM25 Turbo CLI.
//!
//! Tests cover: index command (JSONL, CSV, TXT), search command,
//! serve command (health endpoint), output formatting, tracing,
//! and error cases.

use std::io::Write;
use std::time::Duration;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

/// Helper: create the CLI command pointing at the built binary.
fn bm25_cmd() -> Command {
    Command::cargo_bin("bm25-turbo").expect("binary should be buildable")
}

/// Helper: write a JSONL file with N string documents.
fn write_jsonl(dir: &TempDir, name: &str, docs: &[&str]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    for doc in docs {
        writeln!(f, "\"{}\"", doc).unwrap();
    }
    path
}

/// Helper: write a JSONL file with objects (requires --field).
fn write_jsonl_objects(
    dir: &TempDir,
    name: &str,
    field: &str,
    docs: &[&str],
) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    for doc in docs {
        writeln!(f, "{{\"{}\": \"{}\"}}", field, doc).unwrap();
    }
    path
}

/// Helper: write a CSV file.
fn write_csv(dir: &TempDir, name: &str, header: &str, rows: &[&str]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "{}", header).unwrap();
    for row in rows {
        writeln!(f, "{}", row).unwrap();
    }
    path
}

/// Helper: write a plain text file (one doc per line).
fn write_txt(dir: &TempDir, name: &str, lines: &[&str]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    for line in lines {
        writeln!(f, "{}", line).unwrap();
    }
    path
}

/// Small test corpus for reuse.
const SMALL_CORPUS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "a fast red car drives on the highway",
    "brown fox sleeps in the sun",
    "the lazy dog sits on a mat",
    "quick quick quick fox fox fox",
    "highway car drives fast and red",
    "sun moon stars galaxy universe",
    "the quick red fox jumps high",
    "lazy dog lazy dog lazy dog",
    "brown bear eats honey in forest",
];

/// Helper: build an index from the small corpus, return the index file path.
fn build_test_index(dir: &TempDir) -> std::path::PathBuf {
    let input = write_jsonl(dir, "corpus.jsonl", SMALL_CORPUS);
    let output = dir.path().join("test.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .assert()
        .success();

    assert!(output.exists(), "Index file should exist after build");
    output
}

// =========================================================================
// TEST-P6-001: CLI Index JSONL
// =========================================================================

#[test]
fn test_p6_001_index_jsonl() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(&dir, "corpus.jsonl", SMALL_CORPUS);
    let output = dir.path().join("index_jsonl.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .assert()
        .success()
        .stdout(predicate::str::contains("Index built successfully"))
        .stdout(predicate::str::contains("Documents:  10"));

    // Verify the file exists and is loadable by the search command.
    assert!(output.exists());
    let metadata = std::fs::metadata(&output).unwrap();
    assert!(metadata.len() > 0, "Index file should not be empty");
}

/// Test JSONL with objects and --field flag.
#[test]
fn test_p6_001b_index_jsonl_objects() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl_objects(
        &dir,
        "corpus_obj.jsonl",
        "text",
        &["hello world", "foo bar baz", "hello foo bar"],
    );
    let output = dir.path().join("index_jsonl_obj.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .arg("--field")
        .arg("text")
        .assert()
        .success()
        .stdout(predicate::str::contains("Documents:  3"));
}

// =========================================================================
// TEST-P6-002: CLI Index CSV
// =========================================================================

#[test]
fn test_p6_002_index_csv() {
    let dir = TempDir::new().unwrap();
    let input = write_csv(
        &dir,
        "corpus.csv",
        "id,text,category",
        &[
            "1,the quick brown fox,animal",
            "2,the lazy dog,animal",
            "3,brown fox jumps over the lazy dog,animal",
            "4,quick quick quick fox,animal",
            "5,the dog sat on the mat,furniture",
        ],
    );
    let output = dir.path().join("index_csv.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .arg("--field")
        .arg("text")
        .assert()
        .success()
        .stdout(predicate::str::contains("Documents:  5"));

    assert!(output.exists());
}

// =========================================================================
// TEST-P6-001c: CLI Index Plain Text
// =========================================================================

#[test]
fn test_p6_001c_index_plain_text() {
    let dir = TempDir::new().unwrap();
    let input = write_txt(
        &dir,
        "corpus.txt",
        &[
            "the quick brown fox",
            "the lazy dog",
            "brown fox jumps over the lazy dog",
        ],
    );
    let output = dir.path().join("index_txt.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .assert()
        .success()
        .stdout(predicate::str::contains("Documents:  3"));

    assert!(output.exists());
}

// =========================================================================
// TEST-P6-003: CLI Search
// =========================================================================

#[test]
fn test_p6_003_search_basic() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    // Search for "quick" -- should find results and show latency.
    bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg(&index_path)
        .arg("--query")
        .arg("quick")
        .arg("--k")
        .arg("3")
        .assert()
        .success()
        .stdout(predicate::str::contains("Rank"))
        .stdout(predicate::str::contains("Doc ID"))
        .stdout(predicate::str::contains("Score"))
        .stdout(predicate::str::contains("Latency"));
}

// =========================================================================
// TEST-P6-004: CLI Search JSON Output
// =========================================================================

#[test]
fn test_p6_004_search_json_output() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    let output = bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg(&index_path)
        .arg("--query")
        .arg("quick fox")
        .arg("--k")
        .arg("5")
        .arg("--format")
        .arg("json")
        .output()
        .expect("command should run");

    assert!(output.status.success(), "Search command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("Output should be valid JSON");

    // Verify schema: query, results, count, latency_ms.
    assert!(json.get("query").is_some(), "JSON should have 'query' field");
    assert!(
        json.get("results").is_some(),
        "JSON should have 'results' field"
    );
    assert!(json.get("count").is_some(), "JSON should have 'count' field");
    assert!(
        json.get("latency_ms").is_some(),
        "JSON should have 'latency_ms' field"
    );

    assert_eq!(json["query"].as_str().unwrap(), "quick fox");
    assert!(json["count"].as_u64().unwrap() > 0);
    assert!(json["latency_ms"].as_f64().unwrap() >= 0.0);

    // Verify results array structure.
    let results = json["results"].as_array().unwrap();
    assert!(!results.is_empty());
    for result in results {
        assert!(result.get("rank").is_some(), "result should have 'rank'");
        assert!(
            result.get("doc_id").is_some(),
            "result should have 'doc_id'"
        );
        assert!(result.get("score").is_some(), "result should have 'score'");
    }
}

/// Table output format verification.
#[test]
fn test_p6_004b_search_table_output() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg(&index_path)
        .arg("--query")
        .arg("brown fox")
        .arg("--format")
        .arg("table")
        .assert()
        .success()
        .stdout(predicate::str::contains("Rank"))
        .stdout(predicate::str::contains("Doc ID"))
        .stdout(predicate::str::contains("Score"))
        .stdout(predicate::str::contains("Latency"));
}

// =========================================================================
// TEST-P6-005: CLI Serve Health Check
// =========================================================================

#[tokio::test]
async fn test_p6_005_serve_health_check() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    // Use a random-ish port to avoid conflicts.
    let port = 17720 + (std::process::id() % 1000) as u16;

    // Start the serve command in the background.
    let mut child = std::process::Command::new(
        assert_cmd::cargo::cargo_bin("bm25-turbo"),
    )
    .arg("serve")
    .arg("--index")
    .arg(&index_path)
    .arg("--port")
    .arg(port.to_string())
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .spawn()
    .expect("Failed to start serve command");

    // Wait for the server to be ready by polling the health endpoint.
    let client = reqwest::Client::new();
    let health_url = format!("http://127.0.0.1:{}/health", port);

    let mut ready = false;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                ready = true;
                break;
            }
        }
    }

    // Clean up: kill the server process regardless of test outcome.
    let _ = child.kill();
    let _ = child.wait();

    assert!(ready, "Server should respond to /health within 6 seconds");
}

// =========================================================================
// TEST-P6-006: Tracing Spans Emitted (RUST_LOG respected)
// =========================================================================

#[test]
fn test_p6_006_tracing_rust_log() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(
        &dir,
        "corpus.jsonl",
        &["hello world", "foo bar baz", "hello foo"],
    );
    let output = dir.path().join("trace_test.bm25");

    // Run with RUST_LOG=debug to verify tracing output appears on stderr.
    // Use std::process::Command directly because assert_cmd may not
    // forward env vars the same way on all platforms.
    let bin_path = assert_cmd::cargo::cargo_bin("bm25-turbo");
    let cmd_output = std::process::Command::new(&bin_path)
        .env("RUST_LOG", "debug")
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("command should run");

    assert!(cmd_output.status.success());

    let stderr = String::from_utf8_lossy(&cmd_output.stderr);
    // tracing-subscriber outputs spans and events to stderr.
    // With debug level, we expect to see tracing output containing
    // key operations like "build_from_corpus" or "Index built".
    assert!(
        stderr.contains("build_from_corpus") || stderr.contains("Index built") || stderr.contains("DEBUG") || stderr.contains("INFO"),
        "RUST_LOG=debug should produce tracing output on stderr, got: {}",
        stderr
    );
}

/// Verify that RUST_LOG=error suppresses info-level output.
#[test]
fn test_p6_006b_tracing_rust_log_suppresses() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(
        &dir,
        "corpus.jsonl",
        &["hello world", "foo bar baz"],
    );
    let output = dir.path().join("trace_suppress.bm25");

    let cmd_output = bm25_cmd()
        .env("RUST_LOG", "error")
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .expect("command should run");

    assert!(cmd_output.status.success());

    let stderr = String::from_utf8_lossy(&cmd_output.stderr);
    // With RUST_LOG=error, no INFO or DEBUG lines should appear.
    assert!(
        !stderr.contains("INFO") && !stderr.contains("DEBUG"),
        "RUST_LOG=error should suppress info/debug output, got: {}",
        stderr
    );
}

// =========================================================================
// TEST-P6-007: Empty Query Rejection
// =========================================================================

#[test]
fn test_p6_007_empty_query_rejection() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg(&index_path)
        .arg("--query")
        .arg("")
        .assert()
        .failure()
        .stderr(predicate::str::contains("query string must not be empty").or(
            predicate::str::contains("empty"),
        ));
}

// =========================================================================
// Error Cases
// =========================================================================

/// Missing input file for index command.
#[test]
fn test_error_missing_input_file() {
    let dir = TempDir::new().unwrap();
    let output = dir.path().join("missing.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg("/nonexistent/path/corpus.jsonl")
        .arg("--output")
        .arg(&output)
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not exist").or(
            predicate::str::contains("not exist"),
        ));
}

/// Invalid format string for output.
#[test]
fn test_error_invalid_output_format() {
    let dir = TempDir::new().unwrap();
    let index_path = build_test_index(&dir);

    bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg(&index_path)
        .arg("--query")
        .arg("test")
        .arg("--format")
        .arg("xml")
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown format").or(
            predicate::str::contains("Expected"),
        ));
}

/// Missing index file for search command.
#[test]
fn test_error_missing_index_file() {
    bm25_cmd()
        .arg("search")
        .arg("--index")
        .arg("/nonexistent/index.bm25")
        .arg("--query")
        .arg("test")
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not exist").or(
            predicate::str::contains("not exist"),
        ));
}

/// Empty corpus file produces error.
#[test]
fn test_error_empty_corpus() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("empty.jsonl");
    std::fs::write(&input, "").unwrap();
    let output = dir.path().join("empty.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .assert()
        .failure()
        .stderr(predicate::str::contains("no documents").or(
            predicate::str::contains("empty"),
        ));
}

/// Invalid method string produces error.
#[test]
fn test_error_invalid_method() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(&dir, "corpus.jsonl", &["hello world"]);
    let output = dir.path().join("badmethod.bm25");

    bm25_cmd()
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .arg("--method")
        .arg("invalid_method")
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown method"));
}

/// Search with all BM25 variants: index with each variant and search.
#[test]
fn test_all_variants_index_and_search() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(&dir, "corpus.jsonl", SMALL_CORPUS);

    for method in &["robertson", "lucene", "atire", "bm25l", "bm25plus"] {
        let output = dir.path().join(format!("index_{}.bm25", method));

        bm25_cmd()
            .arg("index")
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .arg("--method")
            .arg(method)
            .assert()
            .success();

        bm25_cmd()
            .arg("search")
            .arg("--index")
            .arg(&output)
            .arg("--query")
            .arg("quick fox")
            .arg("--k")
            .arg("3")
            .assert()
            .success()
            .stdout(predicate::str::contains("Rank"));
    }
}

/// Verify --verbose flag enables debug output.
#[test]
fn test_verbose_flag() {
    let dir = TempDir::new().unwrap();
    let input = write_jsonl(&dir, "corpus.jsonl", &["hello world", "foo bar"]);
    let output = dir.path().join("verbose.bm25");

    // Use std::process::Command directly for reliable env/stderr capture.
    let bin_path = assert_cmd::cargo::cargo_bin("bm25-turbo");
    let cmd_output = std::process::Command::new(&bin_path)
        .arg("--verbose")
        .arg("index")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("command should run");

    assert!(cmd_output.status.success());

    let stderr = String::from_utf8_lossy(&cmd_output.stderr);
    // --verbose sets log level to debug, so we should see DEBUG or trace output.
    assert!(
        stderr.contains("DEBUG") || stderr.contains("build_from_corpus") || stderr.contains("Index built"),
        "--verbose should produce debug-level tracing output, got: {}",
        stderr
    );
}
