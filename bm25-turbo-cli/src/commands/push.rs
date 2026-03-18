//! `bm25-turbo push` subcommand — upload an index to HuggingFace Hub.

use std::path::PathBuf;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};

use bm25_turbo::huggingface::{self, HfClient, IndexMetadata};
use bm25_turbo::persistence;

#[derive(Args)]
pub struct PushArgs {
    /// Path to the index file to upload.
    #[arg(short, long)]
    pub index: PathBuf,

    /// HuggingFace repository ID (e.g., "username/repo-name").
    #[arg(short, long)]
    pub repo: String,

    /// HuggingFace API token (or set HF_TOKEN env var).
    #[arg(long)]
    pub token: Option<String>,

    /// Force upload without confirmation.
    #[arg(long)]
    pub force: bool,
}

pub async fn run(args: PushArgs) -> anyhow::Result<()> {
    // 1. Validate index file exists.
    if !args.index.exists() {
        anyhow::bail!("index file does not exist: {}", args.index.display());
    }

    // 2. Resolve token.
    let (token, source) = huggingface::resolve_token(args.token.as_deref())
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    tracing::info!("Using HuggingFace token from {:?}", source);

    // 3. Load index to get metadata.
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message("Loading index for metadata...");
    let index = persistence::load(&args.index)?;
    let file_size = std::fs::metadata(&args.index)?.len();
    let params = index.params();

    let metadata = IndexMetadata {
        num_docs: index.num_docs(),
        vocab_size: index.vocab_size(),
        method: params.method.to_string(),
        file_size,
        k1: params.k1,
        b: params.b,
        delta: params.delta,
    };
    pb.finish_with_message("Index metadata loaded.");

    // 4. Confirm upload (unless --force).
    if !args.force {
        eprintln!();
        eprintln!("About to push to: {}", args.repo);
        eprintln!("  Documents: {}", metadata.num_docs);
        eprintln!("  Vocab:     {}", metadata.vocab_size);
        eprintln!("  Method:    {}", metadata.method);
        eprintln!(
            "  File size: {:.2} MB",
            metadata.file_size as f64 / (1024.0 * 1024.0)
        );
        eprintln!();
    }

    // 5. Push to HuggingFace Hub.
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    pb.set_message("Creating repository...");
    let client = HfClient::new(token)?;

    pb.set_message("Uploading index file...");
    let repo_url = client
        .push_index(&args.index, &args.repo, &metadata)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    pb.finish_with_message("Upload complete!");

    eprintln!();
    eprintln!("Index published to: {}", repo_url);
    eprintln!();
    eprintln!("To download:");
    eprintln!("  bm25-turbo pull --repo {} --output ./index/", args.repo);

    Ok(())
}
