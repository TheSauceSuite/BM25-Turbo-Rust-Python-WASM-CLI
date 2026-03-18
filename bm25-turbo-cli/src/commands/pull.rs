//! `bm25-turbo pull` subcommand — download an index from HuggingFace Hub.

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};

use bm25_turbo::huggingface::{self, HfClient};
use bm25_turbo::persistence;

#[derive(Args)]
pub struct PullArgs {
    /// HuggingFace repository ID (e.g., "username/repo-name").
    #[arg(short, long)]
    pub repo: String,

    /// Directory to save the downloaded index.
    #[arg(short, long)]
    pub output: PathBuf,

    /// HuggingFace API token (or set HF_TOKEN env var).
    #[arg(long)]
    pub token: Option<String>,

    /// Git revision (commit hash, tag, or branch) to download from.
    #[arg(long, default_value = "main")]
    pub revision: String,
}

pub async fn run(args: PullArgs) -> anyhow::Result<()> {
    // 1. Resolve token.
    let (token, source) = huggingface::resolve_token(args.token.as_deref())
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    tracing::info!("Using HuggingFace token from {:?}", source);

    // 2. Download index.
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message(format!(
        "Downloading index from {} (revision: {})...",
        args.repo, args.revision
    ));

    let client = HfClient::new(token)?;
    let index_path = client
        .pull_index(&args.repo, &args.output, &args.revision)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    pb.finish_with_message("Download complete.");

    // 3. Validate the downloaded index.
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message("Validating index integrity...");
    let index = persistence::load(&index_path)
        .map_err(|e| anyhow::anyhow!("downloaded index is invalid: {}", e))?;
    pb.finish_with_message("Index validated.");

    eprintln!();
    eprintln!("Index downloaded to: {}", index_path.display());
    eprintln!("  Documents: {}", index.num_docs());
    eprintln!("  Vocab:     {}", index.vocab_size());
    eprintln!("  Method:    {}", index.params().method);
    eprintln!();
    eprintln!("To search:");
    eprintln!(
        "  bm25-turbo search --index {} --query \"your query\"",
        index_path.display()
    );

    Ok(())
}
