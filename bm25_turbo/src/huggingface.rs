//! HuggingFace Hub client module (feature-gated: `huggingface`).
//!
//! Provides push/pull operations for BM25 indexes to/from HuggingFace Hub:
//! - `push_index` -- upload an index file and model card to a HF repo
//! - `pull_index` -- download an index file from a HF repo with validation
//!
//! Uses reqwest with rustls TLS for HTTP operations.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// HuggingFace Hub API base URL.
const HF_API_BASE: &str = "https://huggingface.co/api";

/// HuggingFace Hub file download base URL.
const HF_RESOLVE_BASE: &str = "https://huggingface.co";

/// Default index filename used in HF repos.
const INDEX_FILENAME: &str = "index.bm25";

/// Model card filename.
const MODEL_CARD_FILENAME: &str = "README.md";

/// HuggingFace Hub client for push/pull operations.
#[derive(Debug, Clone)]
pub struct HfClient {
    /// API token for authentication.
    token: String,
    /// HTTP client.
    client: reqwest::Client,
}

/// Token resolution result.
#[derive(Debug)]
pub enum TokenSource {
    /// Token was provided via CLI flag.
    Flag,
    /// Token was found in environment variable.
    EnvVar,
    /// Token was found in cached credentials file.
    CachedFile,
}

/// Metadata about an index for the model card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Number of documents in the index.
    pub num_docs: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Scoring method name.
    pub method: String,
    /// Index file size in bytes.
    pub file_size: u64,
    /// k1 parameter.
    pub k1: f32,
    /// b parameter.
    pub b: f32,
    /// delta parameter.
    pub delta: f32,
}

/// Response from the HuggingFace create repo API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CreateRepoResponse {
    url: Option<String>,
}

/// Response from the HuggingFace file info API.
#[derive(Debug, Deserialize)]
struct FileInfo {
    #[allow(dead_code)]
    #[serde(rename = "rfilename")]
    filename: String,
    size: u64,
}

/// Resolve HuggingFace API token using a 4-level precedence:
///
/// 1. Explicit token from CLI flag (if Some)
/// 2. `HF_TOKEN` environment variable
/// 3. Cached token file at `~/.cache/huggingface/token`
/// 4. Error with instructions
pub fn resolve_token(explicit: Option<&str>) -> Result<(String, TokenSource)> {
    // Level 1: Explicit token from CLI flag.
    if let Some(token) = explicit {
        if !token.is_empty() {
            return Ok((token.to_string(), TokenSource::Flag));
        }
    }

    // Level 2: HF_TOKEN environment variable.
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Ok((token, TokenSource::EnvVar));
        }
    }

    // Level 3: Cached token file.
    if let Some(home) = home_dir() {
        let token_path = home.join(".cache").join("huggingface").join("token");
        if token_path.exists() {
            if let Ok(token) = std::fs::read_to_string(&token_path) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Ok((token, TokenSource::CachedFile));
                }
            }
        }
    }

    // Level 4: Error with instructions.
    Err(Error::HuggingFaceError(
        "No HuggingFace token found. Provide one via:\n\
         1. --token <TOKEN> flag\n\
         2. HF_TOKEN environment variable\n\
         3. huggingface-cli login (stores token at ~/.cache/huggingface/token)"
            .to_string(),
    ))
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Generate a model card (README.md) with YAML frontmatter for a BM25 index.
pub fn generate_model_card(metadata: &IndexMetadata) -> String {
    let size_mb = metadata.file_size as f64 / (1024.0 * 1024.0);
    format!(
        r#"---
tags:
  - bm25
  - information-retrieval
  - bm25-turbo
library_name: bm25-turbo
---

# BM25 Turbo Index

This repository contains a BM25 index built with [BM25 Turbo](https://github.com/bm25-turbo/bm25-turbo), the fastest BM25 information retrieval engine in any language.

## Index Statistics

| Property | Value |
|----------|-------|
| Documents | {num_docs} |
| Vocabulary | {vocab_size} terms |
| Scoring | {method} |
| k1 | {k1} |
| b | {b} |
| delta | {delta} |
| File size | {size_mb:.2} MB |

## Usage

```bash
# Download the index
bm25-turbo pull --repo <this-repo> --output ./my-index.bm25

# Search the index
bm25-turbo search --index ./my-index.bm25 --query "your search query"

# Serve via HTTP + MCP
bm25-turbo serve --index ./my-index.bm25 --mcp
```

## Built with

[BM25 Turbo](https://github.com/bm25-turbo/bm25-turbo) — sub-millisecond BM25 search on 100M+ documents.
"#,
        num_docs = metadata.num_docs,
        vocab_size = metadata.vocab_size,
        method = metadata.method,
        k1 = metadata.k1,
        b = metadata.b,
        delta = metadata.delta,
        size_mb = size_mb,
    )
}

impl HfClient {
    /// Create a new HuggingFace Hub client with the given API token.
    pub fn new(token: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("bm25-turbo")
            .build()
            .map_err(|e| Error::HuggingFaceError(format!("failed to create HTTP client: {}", e)))?;
        Ok(Self { token, client })
    }

    /// Create a HuggingFace repository if it does not already exist.
    ///
    /// Creates a "dataset" type repo (appropriate for index data).
    pub async fn create_repo_if_needed(&self, repo_id: &str) -> Result<()> {
        let url = format!("{}/repos/create", HF_API_BASE);
        let body = serde_json::json!({
            "name": repo_id.rsplit('/').next().unwrap_or(repo_id),
            "type": "dataset",
            "private": false,
        });

        // If the user has an org prefix, include organization.
        let body = if let Some((org, _name)) = repo_id.split_once('/') {
            let mut b = body;
            b["organization"] = serde_json::json!(org);
            b
        } else {
            body
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("create repo request failed: {}", e)))?;

        let status = resp.status();
        if status.is_success() || status.as_u16() == 409 {
            // 409 Conflict means repo already exists -- that's fine.
            Ok(())
        } else {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            Err(Error::HuggingFaceError(format!(
                "failed to create repo '{}': {} - {}",
                repo_id, status, text
            )))
        }
    }

    /// Upload a file to a HuggingFace repository.
    pub async fn upload_file(
        &self,
        repo_id: &str,
        local_path: &Path,
        remote_filename: &str,
    ) -> Result<()> {
        let url = format!(
            "{}/{}/upload/main/{}",
            HF_API_BASE,
            repo_id,
            remote_filename
        );

        let file_bytes = std::fs::read(local_path).map_err(|e| {
            Error::HuggingFaceError(format!(
                "failed to read file '{}': {}",
                local_path.display(),
                e
            ))
        })?;

        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/octet-stream")
            .body(file_bytes)
            .send()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("upload request failed: {}", e)))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            Err(Error::HuggingFaceError(format!(
                "failed to upload '{}' to '{}': {}",
                remote_filename, repo_id, text
            )))
        }
    }

    /// Upload a string as a file to a HuggingFace repository.
    pub async fn upload_string(
        &self,
        repo_id: &str,
        content: &str,
        remote_filename: &str,
    ) -> Result<()> {
        let url = format!(
            "{}/{}/upload/main/{}",
            HF_API_BASE,
            repo_id,
            remote_filename
        );

        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "text/plain")
            .body(content.to_string())
            .send()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("upload request failed: {}", e)))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            Err(Error::HuggingFaceError(format!(
                "failed to upload '{}' to '{}': {}",
                remote_filename, repo_id, text
            )))
        }
    }

    /// Push a BM25 index to HuggingFace Hub.
    ///
    /// Creates the repo if needed, uploads the index file, and generates a model card.
    pub async fn push_index(
        &self,
        index_path: &Path,
        repo_id: &str,
        metadata: &IndexMetadata,
    ) -> Result<String> {
        // 1. Create repo if needed.
        self.create_repo_if_needed(repo_id).await?;

        // 2. Upload index file.
        self.upload_file(repo_id, index_path, INDEX_FILENAME)
            .await?;

        // 3. Generate and upload model card.
        let model_card = generate_model_card(metadata);
        self.upload_string(repo_id, &model_card, MODEL_CARD_FILENAME)
            .await?;

        let repo_url = format!("{}/{}", HF_RESOLVE_BASE, repo_id);
        Ok(repo_url)
    }

    /// Get file info (size) for a file in a HuggingFace repository.
    async fn get_file_info(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
    ) -> Result<FileInfo> {
        // Use the tree API to list files at the specified revision.
        let url = format!("{}/datasets/{}/tree/{}", HF_API_BASE, repo_id, revision);
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("file info request failed: {}", e)))?;

        if resp.status().as_u16() == 404 {
            return Err(Error::HuggingFaceError(format!(
                "repository '{}' not found or revision '{}' does not exist",
                repo_id, revision
            )));
        }

        if !resp.status().is_success() {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(Error::HuggingFaceError(format!(
                "failed to get file info for '{}': {}",
                repo_id, text
            )));
        }

        let files: Vec<FileInfo> = resp
            .json()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("failed to parse file listing: {}", e)))?;

        files
            .into_iter()
            .find(|f| f.filename == filename)
            .ok_or_else(|| {
                Error::HuggingFaceError(format!(
                    "file '{}' not found in repository '{}'",
                    filename, repo_id
                ))
            })
    }

    /// Download a file from a HuggingFace repository.
    ///
    /// Supports resumable downloads via Range headers. If a `.partial` file exists
    /// and its size is less than the expected file size, the download resumes
    /// from where it left off.
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        output_path: &Path,
        revision: &str,
    ) -> Result<()> {
        let url = format!(
            "{}/datasets/{}/resolve/{}/{}",
            HF_RESOLVE_BASE, repo_id, revision, filename
        );

        let partial_path = output_path.with_extension("bm25.partial");

        // Check if we can resume a partial download.
        let resume_from = if partial_path.exists() {
            std::fs::metadata(&partial_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        let mut req = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token));

        if resume_from > 0 {
            req = req.header("Range", format!("bytes={}-", resume_from));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("download request failed: {}", e)))?;

        if resp.status().as_u16() == 404 {
            return Err(Error::HuggingFaceError(format!(
                "file '{}' not found in repository '{}' at revision '{}'",
                filename, repo_id, revision
            )));
        }

        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(Error::HuggingFaceError(format!(
                "download failed: {}",
                text
            )));
        }

        let is_partial = resp.status().as_u16() == 206;
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::HuggingFaceError(format!("failed to read response body: {}", e)))?;

        // Write to partial file first, then rename to final path.
        if is_partial && resume_from > 0 {
            // Append to existing partial file.
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&partial_path)
                .map_err(|e| {
                    Error::HuggingFaceError(format!("failed to open partial file: {}", e))
                })?;
            file.write_all(&bytes).map_err(|e| {
                Error::HuggingFaceError(format!("failed to write to partial file: {}", e))
            })?;
        } else {
            // Write fresh (overwriting any previous partial).
            std::fs::write(&partial_path, &bytes).map_err(|e| {
                Error::HuggingFaceError(format!("failed to write partial file: {}", e))
            })?;
        }

        // Rename partial to final.
        std::fs::rename(&partial_path, output_path).map_err(|e| {
            Error::HuggingFaceError(format!("failed to rename partial file to output: {}", e))
        })?;

        Ok(())
    }

    /// Pull a BM25 index from HuggingFace Hub.
    ///
    /// Downloads the index file and validates it by loading.
    pub async fn pull_index(
        &self,
        repo_id: &str,
        output_dir: &Path,
        revision: &str,
    ) -> Result<PathBuf> {
        // Ensure output directory exists.
        std::fs::create_dir_all(output_dir)?;

        let output_path = output_dir.join(INDEX_FILENAME);

        // Check available disk space before downloading.
        if let Ok(file_info) = self.get_file_info(repo_id, INDEX_FILENAME, revision).await {
            check_disk_space(output_dir, file_info.size)?;

            // Resume: skip if file already exists and matches expected size.
            if output_path.exists() {
                let existing_size = std::fs::metadata(&output_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                if existing_size == file_info.size {
                    return Ok(output_path);
                }
            }
        }

        // Download the index file.
        self.download_file(repo_id, INDEX_FILENAME, &output_path, revision)
            .await?;

        Ok(output_path)
    }
}

/// Check available disk space on the volume containing `path`.
///
/// Returns an error if the available space is less than `required_bytes`.
pub fn check_disk_space(path: &Path, required_bytes: u64) -> Result<()> {
    let available = get_available_space(path);
    if let Some(avail) = available {
        if avail < required_bytes {
            let required_mb = required_bytes as f64 / (1024.0 * 1024.0);
            let available_mb = avail as f64 / (1024.0 * 1024.0);
            return Err(Error::HuggingFaceError(format!(
                "insufficient disk space: need {:.1} MB but only {:.1} MB available",
                required_mb, available_mb
            )));
        }
    }
    Ok(())
}

/// Get available disk space in bytes for the volume containing `path`.
///
/// Uses platform-specific APIs:
/// - Windows: `GetDiskFreeSpaceExW` via raw FFI
/// - Unix: `statvfs` via raw FFI
///
/// Returns `None` if the check fails (non-fatal).
fn get_available_space(path: &Path) -> Option<u64> {
    get_available_space_impl(path)
}

#[cfg(target_os = "windows")]
fn get_available_space_impl(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;

    // Resolve to an absolute path to ensure the volume can be determined.
    let abs_path = std::fs::canonicalize(path).ok()?;
    let path_wide: Vec<u16> = abs_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut free_bytes_available: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut total_free_bytes: u64 = 0;

    // SAFETY: Calling GetDiskFreeSpaceExW with a valid null-terminated UTF-16
    // path and valid mutable pointers to u64 values.
    let result = unsafe {
        // Declare the FFI binding inline to avoid adding a crate dependency.
        unsafe extern "system" {
            fn GetDiskFreeSpaceExW(
                lpDirectoryName: *const u16,
                lpFreeBytesAvailableToCaller: *mut u64,
                lpTotalNumberOfBytes: *mut u64,
                lpTotalNumberOfFreeBytes: *mut u64,
            ) -> i32;
        }
        GetDiskFreeSpaceExW(
            path_wide.as_ptr(),
            &mut free_bytes_available,
            &mut total_bytes,
            &mut total_free_bytes,
        )
    };

    if result != 0 {
        Some(free_bytes_available)
    } else {
        None
    }
}

#[cfg(all(unix, feature = "huggingface"))]
fn get_available_space_impl(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    let c_path = CString::new(path.to_string_lossy().as_bytes()).ok()?;

    // SAFETY: Calling POSIX statvfs with a valid null-terminated C string path
    // and a properly initialized output buffer.
    unsafe {
        let mut stat = MaybeUninit::<libc::statvfs>::zeroed();
        let result = libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr());
        if result == 0 {
            let stat = stat.assume_init();
            #[allow(clippy::unnecessary_cast)]
            Some(stat.f_bavail as u64 * stat.f_frsize as u64)
        } else {
            None
        }
    }
}

#[cfg(all(unix, not(feature = "huggingface")))]
fn get_available_space_impl(_path: &Path) -> Option<u64> {
    None
}

#[cfg(not(any(target_os = "windows", unix)))]
fn get_available_space_impl(_path: &Path) -> Option<u64> {
    // No disk space check available on this platform.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that manipulate environment variables.
    /// Prevents race conditions when tests run in parallel (default).
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn token_resolution_explicit_flag() {
        let (token, source) = resolve_token(Some("my-test-token")).unwrap();
        assert_eq!(token, "my-test-token");
        assert!(matches!(source, TokenSource::Flag));
    }

    #[test]
    fn token_resolution_empty_flag_falls_through() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Set env var so level 2 catches it.
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::set_var("HF_TOKEN", "env-token-fallthrough") };
        let (token, source) = resolve_token(Some("")).unwrap();
        assert_eq!(token, "env-token-fallthrough");
        assert!(matches!(source, TokenSource::EnvVar));
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::remove_var("HF_TOKEN") };
    }

    #[test]
    fn token_resolution_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::set_var("HF_TOKEN", "my-env-token") };
        let (token, source) = resolve_token(None).unwrap();
        assert_eq!(token, "my-env-token");
        assert!(matches!(source, TokenSource::EnvVar));
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::remove_var("HF_TOKEN") };
    }

    #[test]
    fn model_card_generation() {
        let metadata = IndexMetadata {
            num_docs: 1000,
            vocab_size: 5000,
            method: "Lucene".to_string(),
            file_size: 2 * 1024 * 1024, // 2 MB
            k1: 1.5,
            b: 0.75,
            delta: 0.5,
        };
        let card = generate_model_card(&metadata);

        // Verify YAML frontmatter.
        assert!(card.starts_with("---\n"));
        assert!(card.contains("tags:"));
        assert!(card.contains("bm25"));
        assert!(card.contains("information-retrieval"));
        assert!(card.contains("bm25-turbo"));
        assert!(card.contains("library_name: bm25-turbo"));

        // Verify metadata values.
        assert!(card.contains("1000"));
        assert!(card.contains("5000"));
        assert!(card.contains("Lucene"));
        assert!(card.contains("2.00 MB"));
    }

    #[test]
    fn disk_space_check_sufficient() {
        // Check current directory -- should have space.
        let result = check_disk_space(Path::new("."), 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn disk_space_check_insufficient() {
        // Request an impossibly large amount of space.
        let result = check_disk_space(Path::new("."), u64::MAX);
        // This should fail unless the system has exabytes of storage.
        // On most systems this will be an error.
        if let Some(avail) = get_available_space(Path::new(".")) {
            if avail < u64::MAX {
                assert!(result.is_err());
                let err_msg = result.unwrap_err().to_string();
                assert!(err_msg.contains("insufficient disk space"));
                assert!(err_msg.contains("MB"));
            }
        }
    }

    #[test]
    fn token_resolution_cached_file() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Create a temp directory simulating ~/.cache/huggingface/token.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let cache_dir = tmp.path().join(".cache").join("huggingface");
        std::fs::create_dir_all(&cache_dir).expect("create cache dir");
        std::fs::write(cache_dir.join("token"), "cached-token-123\n").expect("write token");

        // Remove any HF_TOKEN env var so level 2 doesn't interfere.
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::remove_var("HF_TOKEN") };

        // Point HOME/USERPROFILE to our temp dir so resolve_token finds it.
        #[cfg(target_os = "windows")]
        let home_key = "USERPROFILE";
        #[cfg(not(target_os = "windows"))]
        let home_key = "HOME";

        let original_home = std::env::var(home_key).ok();
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::set_var(home_key, tmp.path()) };

        let result = resolve_token(None);
        assert!(result.is_ok(), "should resolve cached token");
        let (token, source) = result.unwrap();
        assert_eq!(token, "cached-token-123");
        assert!(matches!(source, TokenSource::CachedFile));

        // Restore original home.
        if let Some(home) = original_home {
            // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
            unsafe { std::env::set_var(home_key, home) };
        }
    }

    #[test]
    fn token_resolution_no_token_returns_error() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Remove all possible token sources.
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::remove_var("HF_TOKEN") };

        // Point HOME to a non-existent directory so cached file isn't found.
        #[cfg(target_os = "windows")]
        let home_key = "USERPROFILE";
        #[cfg(not(target_os = "windows"))]
        let home_key = "HOME";

        let original_home = std::env::var(home_key).ok();
        // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
        unsafe { std::env::set_var(home_key, "/nonexistent/path/that/does/not/exist") };

        let result = resolve_token(None);
        assert!(result.is_err(), "should error when no token found");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("No HuggingFace token found"));

        // Restore original home.
        if let Some(home) = original_home {
            // SAFETY: Guarded by ENV_MUTEX, no concurrent env access.
            unsafe { std::env::set_var(home_key, home) };
        }
    }

    #[test]
    fn model_card_contains_yaml_frontmatter_tags() {
        let metadata = IndexMetadata {
            num_docs: 50000,
            vocab_size: 12345,
            method: "Robertson".to_string(),
            file_size: 10 * 1024 * 1024,
            k1: 1.2,
            b: 0.8,
            delta: 1.0,
        };
        let card = generate_model_card(&metadata);

        // Verify YAML frontmatter delimiters.
        assert!(card.starts_with("---\n"));
        assert!(card.contains("\n---\n"));

        // Verify all required tags in YAML.
        assert!(card.contains("  - bm25\n"));
        assert!(card.contains("  - information-retrieval\n"));
        assert!(card.contains("  - bm25-turbo\n"));
        assert!(card.contains("library_name: bm25-turbo"));

        // Verify metadata values rendered.
        assert!(card.contains("50000"));
        assert!(card.contains("12345"));
        assert!(card.contains("Robertson"));
        assert!(card.contains("10.00 MB"));
        assert!(card.contains("| k1 | 1.2 |"));
        assert!(card.contains("| b | 0.8 |"));
        assert!(card.contains("| delta | 1 |"));
    }
}
