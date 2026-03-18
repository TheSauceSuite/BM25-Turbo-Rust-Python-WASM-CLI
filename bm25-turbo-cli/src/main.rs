//! BM25 Turbo CLI -- command-line interface and HTTP server.
//!
//! Subcommands:
//! - `index`  — Build a BM25 index from a corpus file
//! - `search` — Query an existing index
//! - `serve`  — Start an HTTP (+ optional MCP) server
//! - `push`   — Upload an index to HuggingFace Hub
//! - `pull`   — Download an index from HuggingFace Hub
//! - `bench`  — Run built-in benchmarks

mod commands;
mod output;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bm25-turbo",
    about = "The fastest BM25 information retrieval engine",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging (set RUST_LOG level).
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a BM25 index from a corpus file.
    Index(commands::index::IndexArgs),
    /// Search an existing BM25 index.
    Search(commands::search::SearchArgs),
    /// Start an HTTP server for querying indexes.
    Serve(commands::serve::ServeArgs),
    /// Start a standalone MCP server.
    Mcp(commands::mcp::McpArgs),
    /// Upload an index to HuggingFace Hub.
    Push(commands::push::PushArgs),
    /// Download an index from HuggingFace Hub.
    Pull(commands::pull::PullArgs),
    /// Run built-in benchmarks.
    Bench(commands::bench::BenchArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing with RUST_LOG env var support.
    // Default level: info (or debug with --verbose).
    // Serve mode uses JSON format for structured logging.
    let filter = if cli.verbose { "debug" } else { "info" };
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter));

    let is_serve = matches!(cli.command, Commands::Serve(_) | Commands::Mcp(_));
    if is_serve {
        // JSON format for serve mode (structured logging for production).
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(std::io::stderr)
            .json()
            .init();
    } else {
        // Human-readable format for interactive CLI commands.
        // Write to stderr so tracing output does not mix with program output.
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(std::io::stderr)
            .init();
    }

    match cli.command {
        Commands::Index(args) => commands::index::run(args).await,
        Commands::Search(args) => commands::search::run(args).await,
        Commands::Serve(args) => commands::serve::run(args).await,
        Commands::Mcp(args) => commands::mcp::run(args).await,
        Commands::Push(args) => commands::push::run(args).await,
        Commands::Pull(args) => commands::pull::run(args).await,
        Commands::Bench(args) => commands::bench::run(args).await,
    }
}
