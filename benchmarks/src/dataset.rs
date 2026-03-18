use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

/// Supported BEIR evaluation datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    NQ,
    MSMARCO,
    SciFact,
    FiQA,
}

impl Dataset {
    /// Returns the BEIR download URL for this dataset.
    pub fn url(&self) -> &str {
        match self {
            Dataset::NQ => {
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
            }
            Dataset::MSMARCO => {
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
            }
            Dataset::SciFact => {
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
            }
            Dataset::FiQA => {
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
            }
        }
    }

    /// Returns the dataset name used as the directory name after extraction.
    pub fn name(&self) -> &str {
        match self {
            Dataset::NQ => "nq",
            Dataset::MSMARCO => "msmarco",
            Dataset::SciFact => "scifact",
            Dataset::FiQA => "fiqa",
        }
    }
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Paths to the extracted dataset files.
#[derive(Debug, Clone)]
pub struct DatasetFiles {
    pub corpus_path: PathBuf,
    pub queries_path: PathBuf,
    pub qrels_path: PathBuf,
}

/// Parsed corpus: parallel arrays of document IDs and texts.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub doc_ids: Vec<String>,
    pub texts: Vec<String>,
}

/// Parsed queries: parallel arrays of query IDs and texts.
#[derive(Debug, Clone)]
pub struct Queries {
    pub query_ids: Vec<String>,
    pub texts: Vec<String>,
}

/// Relevance judgments: query_id -> (doc_id -> relevance_grade).
#[derive(Debug, Clone)]
pub struct Qrels(pub HashMap<String, HashMap<String, u32>>);

/// BEIR corpus.jsonl record format.
#[derive(Deserialize)]
struct CorpusRecord {
    _id: String,
    title: Option<String>,
    text: String,
}

/// BEIR queries.jsonl record format.
#[derive(Deserialize)]
struct QueryRecord {
    _id: String,
    text: String,
}

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_RETRIES: u32 = 3;

/// Downloads a BEIR dataset tarball to `cache_dir`, extracts it, and returns
/// paths to the extracted corpus, queries, and qrels files.
///
/// If the extracted directory already exists in `cache_dir`, the download is
/// skipped (cache hit).
pub fn download_dataset(dataset: Dataset, cache_dir: &Path) -> Result<DatasetFiles, String> {
    let dataset_dir = cache_dir.join(dataset.name());

    // Check cache hit
    if !dataset_dir.exists() {
        fs::create_dir_all(cache_dir).map_err(|e| format!("failed to create cache dir: {e}"))?;

        let zip_bytes = download_with_retry(dataset.url(), MAX_RETRIES)?;

        let cursor = std::io::Cursor::new(&zip_bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| format!("failed to open zip archive: {e}"))?;
        archive
            .extract(cache_dir)
            .map_err(|e| format!("failed to extract zip archive: {e}"))?;

        if !dataset_dir.exists() {
            return Err(format!(
                "extraction did not produce expected directory: {}",
                dataset_dir.display()
            ));
        }
    }

    let corpus_path = dataset_dir.join("corpus.jsonl");
    let queries_path = dataset_dir.join("queries.jsonl");
    let qrels_dir = dataset_dir.join("qrels");
    let qrels_path = qrels_dir.join("test.tsv");

    if !corpus_path.exists() {
        return Err(format!("corpus.jsonl not found at {}", corpus_path.display()));
    }
    if !queries_path.exists() {
        return Err(format!(
            "queries.jsonl not found at {}",
            queries_path.display()
        ));
    }
    if !qrels_path.exists() {
        return Err(format!("qrels/test.tsv not found at {}", qrels_path.display()));
    }

    Ok(DatasetFiles {
        corpus_path,
        queries_path,
        qrels_path,
    })
}

/// Downloads a URL with retry logic. Returns the response bytes.
fn download_with_retry(url: &str, max_retries: u32) -> Result<Vec<u8>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    let mut last_err = String::new();
    for attempt in 1..=max_retries {
        match client.get(url).send() {
            Ok(resp) => {
                if resp.status().is_success() {
                    return resp
                        .bytes()
                        .map(|b| b.to_vec())
                        .map_err(|e| format!("failed to read response bytes: {e}"));
                }
                last_err = format!("HTTP {}", resp.status());
            }
            Err(e) => {
                last_err = format!("{e}");
            }
        }
        if attempt < max_retries {
            std::thread::sleep(Duration::from_secs(u64::from(attempt)));
        }
    }
    Err(format!(
        "download failed after {max_retries} attempts: {last_err}"
    ))
}

/// Parses a BEIR corpus.jsonl file. Each line is a JSON object with `_id`,
/// `title` (optional), and `text` fields. Text is `title + " " + text` when
/// title is present.
///
/// Uses streaming line-by-line parsing to handle large files.
pub fn parse_corpus(path: &Path) -> Result<Corpus, String> {
    let file =
        fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);

    let mut doc_ids = Vec::new();
    let mut texts = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read line {} of {}: {e}", line_num + 1, path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: CorpusRecord = serde_json::from_str(line)
            .map_err(|e| format!("failed to parse line {} of {}: {e}", line_num + 1, path.display()))?;

        let text = match record.title {
            Some(ref title) if !title.is_empty() => format!("{} {}", title, record.text),
            _ => record.text,
        };

        doc_ids.push(record._id);
        texts.push(text);
    }

    Ok(Corpus { doc_ids, texts })
}

/// Parses a BEIR queries.jsonl file. Each line is a JSON object with `_id`
/// and `text` fields.
///
/// Uses streaming line-by-line parsing.
pub fn parse_queries(path: &Path) -> Result<Queries, String> {
    let file =
        fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);

    let mut query_ids = Vec::new();
    let mut texts = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read line {} of {}: {e}", line_num + 1, path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: QueryRecord = serde_json::from_str(line)
            .map_err(|e| format!("failed to parse line {} of {}: {e}", line_num + 1, path.display()))?;

        query_ids.push(record._id);
        texts.push(record.text);
    }

    Ok(Queries { query_ids, texts })
}

/// Parses a BEIR qrels TSV file. Format: `query_id\tdoc_id\trelevance`.
/// The first line is a header and is skipped.
///
/// Returns a nested HashMap: query_id -> (doc_id -> relevance_grade).
pub fn parse_qrels(path: &Path) -> Result<Qrels, String> {
    let file =
        fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(true)
        .from_reader(BufReader::new(file));

    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();

    for (row_num, result) in reader.records().enumerate() {
        let record = result.map_err(|e| {
            format!(
                "failed to parse row {} of {}: {e}",
                row_num + 2,
                path.display()
            )
        })?;

        // BEIR qrels format: query-id, corpus-id, score
        if record.len() < 3 {
            return Err(format!(
                "row {} of {} has {} fields, expected at least 3",
                row_num + 2,
                path.display(),
                record.len()
            ));
        }

        let query_id = record[0].to_string();
        let doc_id = record[1].to_string();
        let relevance: u32 = record[2]
            .parse()
            .map_err(|e| format!("invalid relevance at row {}: {e}", row_num + 2))?;

        qrels
            .entry(query_id)
            .or_default()
            .insert(doc_id, relevance);
    }

    Ok(Qrels(qrels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_urls_are_nonempty() {
        for ds in [Dataset::NQ, Dataset::MSMARCO, Dataset::SciFact, Dataset::FiQA] {
            assert!(!ds.url().is_empty());
            assert!(ds.url().starts_with("https://"));
            assert!(ds.url().ends_with(".zip") || ds.url().ends_with(".tar.gz"));
        }
    }

    #[test]
    fn dataset_names() {
        assert_eq!(Dataset::NQ.name(), "nq");
        assert_eq!(Dataset::MSMARCO.name(), "msmarco");
        assert_eq!(Dataset::SciFact.name(), "scifact");
        assert_eq!(Dataset::FiQA.name(), "fiqa");
    }

    #[test]
    fn dataset_display() {
        assert_eq!(format!("{}", Dataset::NQ), "nq");
    }
}
