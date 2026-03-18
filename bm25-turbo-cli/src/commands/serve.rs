//! `bm25-turbo serve` subcommand -- start an HTTP server.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Args;
use tokio::sync::RwLock;
use tracing;

use bm25_turbo::persistence;
use bm25_turbo::server::{self, AppState};

#[derive(Args)]
pub struct ServeArgs {
    /// Path to the index file to serve.
    #[arg(short, long)]
    pub index: PathBuf,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to bind to.
    #[arg(short, long, default_value = "7720")]
    pub port: u16,

    /// Enable MCP server alongside HTTP.
    #[arg(long)]
    pub mcp: bool,

    /// Enable approximate (BMW) search mode.
    #[arg(short, long)]
    pub approximate: bool,
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    // Validate index file exists.
    if !args.index.exists() {
        anyhow::bail!("index file does not exist: {}", args.index.display());
    }

    // 1. Load the index from disk.
    tracing::info!(path = %args.index.display(), "Loading index");
    let mut index = persistence::load(&args.index)?;
    let num_docs = index.num_docs();
    let vocab_size = index.vocab_size();
    tracing::info!(num_docs, vocab_size, "Index loaded");

    // Build BMW index if approximate search is requested.
    if args.approximate {
        tracing::info!("Building BMW index for approximate search");
        index.build_bmw_index()?;
        tracing::info!("BMW index built");
    }

    // 2. Build shared state.
    let index_arc = Arc::new(index);
    let state = Arc::new(AppState {
        index: RwLock::new(Arc::clone(&index_arc)),
        started_at: Instant::now(),
        approximate: args.approximate,
    });

    // 3. Create the router.
    let mut app = server::router(state);

    // 3b. Mount MCP endpoint if --mcp flag is set.
    if args.mcp {
        tracing::info!("Mounting MCP endpoint at /mcp");
        let mcp_app = bm25_turbo::mcp::mcp_router(index_arc);
        app = app.merge(mcp_app);
    }

    // 4. Bind with port-in-use detection.
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
    eprintln!("BM25 Turbo server running on http://{}", addr);
    eprintln!("  Documents: {}", num_docs);
    eprintln!("  Vocab:     {}", vocab_size);
    eprintln!();
    eprintln!("Endpoints:");
    eprintln!("  POST /search       - Query the index");
    eprintln!("  POST /batch        - Batch query the index");
    eprintln!("  GET  /health       - Health check");
    eprintln!("  GET  /stats        - Index statistics");
    eprintln!("  POST /admin/reload - Reload index (placeholder)");
    if args.approximate {
        eprintln!("  Mode: approximate (BMW)");
    }
    if args.mcp {
        eprintln!("  POST /mcp          - MCP (Model Context Protocol)");
    }
    eprintln!();
    eprintln!("Press Ctrl-C to stop.");

    tracing::info!(addr = %addr, "Server listening");

    // 5. Serve with graceful shutdown on Ctrl-C / SIGTERM.
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
