//! `bm25-turbo mcp` subcommand -- start a standalone MCP server.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use tracing;

use bm25_turbo::persistence;

#[derive(Args)]
pub struct McpArgs {
    /// Path to the index file to serve.
    #[arg(short, long)]
    pub index: PathBuf,

    /// Port to bind to.
    #[arg(short, long, default_value = "3001")]
    pub port: u16,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
}

pub async fn run(args: McpArgs) -> anyhow::Result<()> {
    // Validate index file exists.
    if !args.index.exists() {
        anyhow::bail!("index file does not exist: {}", args.index.display());
    }

    // Load the index from disk.
    tracing::info!(path = %args.index.display(), "Loading index");
    let index = persistence::load(&args.index)?;
    let num_docs = index.num_docs();
    let vocab_size = index.vocab_size();
    tracing::info!(num_docs, vocab_size, "Index loaded");

    let index_arc = Arc::new(index);

    // Create MCP router.
    let app = bm25_turbo::mcp::mcp_router(index_arc);

    // Bind with port-in-use detection.
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "port {} is already in use. Try a different port with --port.",
                args.port
            )
        } else {
            anyhow::anyhow!("failed to bind to {}: {}", addr, e)
        }
    })?;

    eprintln!();
    eprintln!("BM25 Turbo MCP server running on http://{}", addr);
    eprintln!("  Documents: {}", num_docs);
    eprintln!("  Vocab:     {}", vocab_size);
    eprintln!();
    eprintln!("Endpoints:");
    eprintln!("  POST /mcp  - MCP (Model Context Protocol)");
    eprintln!();
    eprintln!("Press Ctrl-C to stop.");

    tracing::info!(addr = %addr, "MCP server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    eprintln!("\nServer stopped.");

    Ok(())
}

/// Wait for Ctrl-C for graceful shutdown.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for Ctrl-C");
    eprintln!("\nShutting down gracefully...");
}
