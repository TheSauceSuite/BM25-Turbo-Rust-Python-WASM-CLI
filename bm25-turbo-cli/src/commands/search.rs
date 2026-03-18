//! `bm25-turbo search` subcommand -- query an existing index.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context};
use clap::Args;

use crate::output;

#[derive(Args)]
pub struct SearchArgs {
    /// Path to the index file.
    #[arg(short, long)]
    pub index: PathBuf,

    /// Search query string.
    #[arg(short, long)]
    pub query: String,

    /// Number of results to return.
    #[arg(short, long, default_value = "10")]
    pub k: usize,

    /// Output format: json or table.
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Disable colored output.
    #[arg(long)]
    pub no_color: bool,

    /// Use approximate (BMW) search instead of exact search.
    #[arg(short, long)]
    pub approximate: bool,
}

pub async fn run(args: SearchArgs) -> anyhow::Result<()> {
    // Validate query is not empty.
    if args.query.trim().is_empty() {
        bail!("query string must not be empty");
    }

    // Validate index file exists.
    if !args.index.exists() {
        bail!("index file does not exist: {}", args.index.display());
    }

    // Parse output format.
    let format = output::Format::from_str(&args.format)?;

    // Load index (try mmap first, fall back to regular load).
    let mut index = bm25_turbo::persistence::mmap_or_load(&args.index)
        .with_context(|| format!("failed to load index from {}", args.index.display()))?;

    // Build BMW index if approximate search is requested.
    if args.approximate {
        index
            .build_bmw_index()
            .context("failed to build BMW index for approximate search")?;
    }

    // Execute query with latency measurement.
    let start = Instant::now();
    let results = if args.approximate {
        index
            .search_approximate(&args.query, args.k)
            .context("approximate search failed")?
    } else {
        index
            .search(&args.query, args.k)
            .context("search failed")?
    };
    let latency = start.elapsed();

    // Print results with latency.
    output::print_results_with_latency(&results, &args.query, latency, format)?;

    Ok(())
}
