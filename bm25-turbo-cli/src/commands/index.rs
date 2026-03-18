//! `bm25-turbo index` subcommand -- build a BM25 index from a corpus file.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};

use bm25_turbo::{BM25Builder, Method};

#[derive(Args)]
pub struct IndexArgs {
    /// Path to the input corpus file (CSV, JSONL, JSON, or plain text).
    #[arg(short, long)]
    pub input: PathBuf,

    /// Path to write the output index file.
    #[arg(short, long)]
    pub output: PathBuf,

    /// BM25 variant to use.
    #[arg(long, default_value = "lucene")]
    pub method: String,

    /// k1 parameter (term frequency saturation).
    #[arg(long, default_value = "1.5")]
    pub k1: f32,

    /// b parameter (document length normalization).
    #[arg(long, default_value = "0.75")]
    pub b: f32,

    /// Field name to extract from CSV or JSON objects.
    #[arg(long)]
    pub field: Option<String>,

    /// Output format: json or table.
    #[arg(long, default_value = "table")]
    pub format: String,
}

/// Parse the method string into a Method enum value.
fn parse_method(s: &str) -> anyhow::Result<Method> {
    match s.to_lowercase().as_str() {
        "robertson" => Ok(Method::Robertson),
        "lucene" => Ok(Method::Lucene),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::Bm25l),
        "bm25plus" | "bm25+" => Ok(Method::Bm25Plus),
        _ => bail!(
            "unknown method: '{}'. Expected: robertson, lucene, atire, bm25l, bm25plus",
            s
        ),
    }
}

/// Detect input format from file extension and content.
enum InputFormat {
    Json,
    Jsonl,
    Csv,
    PlainText,
}

fn detect_format(path: &Path) -> InputFormat {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("json") => InputFormat::Json,
        Some("jsonl") | Some("ndjson") => InputFormat::Jsonl,
        Some("csv") | Some("tsv") => InputFormat::Csv,
        _ => InputFormat::PlainText,
    }
}

/// Read documents from a JSON array file.
/// Supports: array of strings, or array of objects (requires --field).
fn read_json(path: &Path, field: &Option<String>) -> anyhow::Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&content).context("failed to parse JSON")?;

    let arr = value
        .as_array()
        .context("expected a JSON array at top level")?;

    let mut docs = Vec::with_capacity(arr.len());
    for item in arr {
        if let Some(s) = item.as_str() {
            docs.push(s.to_string());
        } else if let Some(obj) = item.as_object() {
            let field_name = field
                .as_deref()
                .context("JSON contains objects; use --field to specify the text column")?;
            let val = obj
                .get(field_name)
                .with_context(|| format!("field '{}' not found in JSON object", field_name))?;
            docs.push(
                val.as_str()
                    .with_context(|| format!("field '{}' is not a string", field_name))?
                    .to_string(),
            );
        } else {
            bail!("JSON array elements must be strings or objects");
        }
    }
    Ok(docs)
}

/// Read documents from a JSONL file (one JSON value per line).
fn read_jsonl(path: &Path, field: &Option<String>) -> anyhow::Result<Vec<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", i + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .with_context(|| format!("failed to parse JSON on line {}", i + 1))?;

        if let Some(s) = value.as_str() {
            docs.push(s.to_string());
        } else if let Some(obj) = value.as_object() {
            let field_name = field
                .as_deref()
                .context("JSONL contains objects; use --field to specify the text column")?;
            let val = obj
                .get(field_name)
                .with_context(|| format!("field '{}' not found on line {}", field_name, i + 1))?;
            docs.push(
                val.as_str()
                    .with_context(|| {
                        format!("field '{}' is not a string on line {}", field_name, i + 1)
                    })?
                    .to_string(),
            );
        } else {
            bail!("JSONL line {} is not a string or object", i + 1);
        }
    }
    Ok(docs)
}

/// Read documents from a CSV file using the specified field/column.
fn read_csv(path: &Path, field: &Option<String>) -> anyhow::Result<Vec<String>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open CSV {}", path.display()))?;

    // Determine which column to use.
    let headers = reader.headers()?.clone();
    let col_idx = if let Some(field_name) = field.as_deref() {
        headers
            .iter()
            .position(|h| h == field_name)
            .with_context(|| {
                format!(
                    "field '{}' not found in CSV headers: {:?}",
                    field_name,
                    headers.iter().collect::<Vec<_>>()
                )
            })?
    } else {
        // Default to first column.
        0
    };

    let mut docs = Vec::new();
    for result in reader.records() {
        let record = result.context("failed to read CSV record")?;
        if let Some(val) = record.get(col_idx) {
            if !val.is_empty() {
                docs.push(val.to_string());
            }
        }
    }
    Ok(docs)
}

/// Read documents from a plain text file (one document per line).
fn read_plain_text(path: &Path) -> anyhow::Result<Vec<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut docs = Vec::new();
    for line in reader.lines() {
        let line = line.context("failed to read line")?;
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            docs.push(trimmed);
        }
    }
    Ok(docs)
}

pub async fn run(args: IndexArgs) -> anyhow::Result<()> {
    let start = Instant::now();

    // Validate input file exists.
    if !args.input.exists() {
        bail!("input file does not exist: {}", args.input.display());
    }

    // Parse method.
    let method = parse_method(&args.method)?;

    // Set up Ctrl-C handler to clean up partial output.
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();
    let output_path = args.output.clone();
    ctrlc::set_handler(move || {
        interrupted_clone.store(true, Ordering::SeqCst);
        // Attempt to remove partial output file.
        let _ = std::fs::remove_file(&output_path);
        eprintln!("\nInterrupted. Partial output cleaned up.");
        std::process::exit(130);
    })
    .context("failed to set Ctrl-C handler")?;

    // Detect format and read documents.
    let format = detect_format(&args.input);
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message("Reading input file...");

    let docs = match format {
        InputFormat::Json => read_json(&args.input, &args.field)?,
        InputFormat::Jsonl => read_jsonl(&args.input, &args.field)?,
        InputFormat::Csv => read_csv(&args.input, &args.field)?,
        InputFormat::PlainText => read_plain_text(&args.input)?,
    };

    if docs.is_empty() {
        bail!("no documents found in input file");
    }

    pb.set_message(format!("Read {} documents. Building index...", docs.len()));

    // Check for interruption.
    if interrupted.load(Ordering::SeqCst) {
        bail!("interrupted");
    }

    // Build index.
    let corpus_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let index = BM25Builder::new()
        .method(method)
        .k1(args.k1)
        .b(args.b)
        .build_from_corpus(&corpus_refs)
        .context("failed to build index")?;

    pb.set_message("Saving index...");

    // Ensure output directory exists.
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }

    // Save index.
    bm25_turbo::persistence::save(&index, &args.output)
        .context("failed to save index")?;

    // Read-back verification.
    pb.set_message("Verifying index...");
    let loaded = bm25_turbo::persistence::load(&args.output)
        .context("failed to verify saved index")?;

    if loaded.num_docs() != index.num_docs() {
        bail!(
            "verification failed: saved {} docs but loaded {}",
            index.num_docs(),
            loaded.num_docs()
        );
    }

    pb.finish_and_clear();

    // Summary stats.
    let elapsed = start.elapsed();
    let file_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("Index built successfully!");
    println!("  Documents:  {}", index.num_docs());
    println!("  Vocab size: {}", index.vocab_size());
    println!("  Method:     {}", method);
    println!("  k1={}, b={}", args.k1, args.b);
    println!("  File size:  {}", format_bytes(file_size));
    println!("  Time:       {:.2?}", elapsed);
    println!("  Output:     {}", args.output.display());

    Ok(())
}

/// Format a byte count into a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
