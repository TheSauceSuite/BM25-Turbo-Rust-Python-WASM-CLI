//! Shared output formatting utilities.
//!
//! Provides helpers for rendering search results, index statistics,
//! and progress indicators in both table and JSON formats.

use std::time::Duration;

use bm25_turbo::Results;
use serde::Serialize;

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Table,
    Json,
}

impl Format {
    /// Parse a format string ("json" or "table").
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "table" => Ok(Self::Table),
            _ => anyhow::bail!("unknown format: {s}. Expected 'json' or 'table'"),
        }
    }
}

/// JSON output schema for search results with latency.
#[derive(Serialize)]
struct JsonSearchOutput {
    query: String,
    results: Vec<JsonResult>,
    count: usize,
    latency_ms: f64,
}

#[derive(Serialize)]
struct JsonResult {
    rank: usize,
    doc_id: u32,
    score: f32,
}

/// Print search results in the requested format (without latency info).
#[allow(dead_code)]
pub fn print_results(results: &Results, format: Format) -> anyhow::Result<()> {
    match format {
        Format::Json => {
            println!("{}", serde_json::to_string_pretty(results)?);
        }
        Format::Table => {
            let mut table = comfy_table::Table::new();
            table.set_header(vec!["Rank", "Doc ID", "Score"]);

            for (i, (doc_id, score)) in
                results.doc_ids.iter().zip(results.scores.iter()).enumerate()
            {
                table.add_row(vec![
                    (i + 1).to_string(),
                    doc_id.to_string(),
                    format!("{score:.6}"),
                ]);
            }

            println!("{table}");
        }
    }

    Ok(())
}

/// Print search results with latency information.
pub fn print_results_with_latency(
    results: &Results,
    query: &str,
    latency: Duration,
    format: Format,
) -> anyhow::Result<()> {
    let latency_ms = latency.as_secs_f64() * 1000.0;

    match format {
        Format::Json => {
            let output = JsonSearchOutput {
                query: query.to_string(),
                results: results
                    .doc_ids
                    .iter()
                    .zip(results.scores.iter())
                    .enumerate()
                    .map(|(i, (&doc_id, &score))| JsonResult {
                        rank: i + 1,
                        doc_id,
                        score,
                    })
                    .collect(),
                count: results.doc_ids.len(),
                latency_ms,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Format::Table => {
            let mut table = comfy_table::Table::new();
            table.set_header(vec!["Rank", "Doc ID", "Score"]);

            for (i, (doc_id, score)) in
                results.doc_ids.iter().zip(results.scores.iter()).enumerate()
            {
                table.add_row(vec![
                    (i + 1).to_string(),
                    doc_id.to_string(),
                    format!("{score:.6}"),
                ]);
            }

            println!("{table}");
            println!();
            println!(
                "Query: \"{}\" | Results: {} | Latency: {:.3} ms",
                query,
                results.doc_ids.len(),
                latency_ms
            );
        }
    }

    Ok(())
}
