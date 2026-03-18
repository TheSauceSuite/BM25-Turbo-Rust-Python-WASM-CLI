use serde::{Deserialize, Serialize};
use std::fmt;

/// Top-level benchmark run containing results for all evaluated datasets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    pub timestamp: String,
    pub system_info: String,
    pub rust_version: String,
    pub datasets: Vec<DatasetResult>,
}

/// Results for a single dataset evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetResult {
    pub name: String,
    pub num_docs: u64,
    pub num_queries: u64,
    pub index_time_ms: f64,
    pub queries_per_sec: f64,
    pub latency_p50_us: f64,
    pub latency_p95_us: f64,
    pub latency_p99_us: f64,
    pub ndcg_at_10: f64,
    pub peak_memory_bytes: u64,
}

impl BenchmarkRun {
    /// Serializes the benchmark run to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a benchmark run from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for BenchmarkRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BM25 Turbo Benchmark Results")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Timestamp:    {}", self.timestamp)?;
        writeln!(f, "System:       {}", self.system_info)?;
        writeln!(f, "Rust:         {}", self.rust_version)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "Dataset", "Docs", "Queries", "Idx(ms)", "QPS", "P50(us)", "P95(us)", "P99(us)", "nDCG@10"
        )?;
        writeln!(f, "{}", "-".repeat(100))?;
        for ds in &self.datasets {
            writeln!(
                f,
                "{:<12} {:>10} {:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>8.4}",
                ds.name,
                ds.num_docs,
                ds.num_queries,
                ds.index_time_ms,
                ds.queries_per_sec,
                ds.latency_p50_us,
                ds.latency_p95_us,
                ds.latency_p99_us,
                ds.ndcg_at_10,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_run() -> BenchmarkRun {
        BenchmarkRun {
            timestamp: "2026-03-17T00:00:00Z".to_string(),
            system_info: "test-machine".to_string(),
            rust_version: "1.85.0".to_string(),
            datasets: vec![DatasetResult {
                name: "scifact".to_string(),
                num_docs: 5183,
                num_queries: 300,
                index_time_ms: 120.5,
                queries_per_sec: 15000.0,
                latency_p50_us: 45.2,
                latency_p95_us: 89.1,
                latency_p99_us: 112.3,
                ndcg_at_10: 0.6823,
                peak_memory_bytes: 10_000_000,
            }],
        }
    }

    #[test]
    fn json_round_trip() {
        let run = sample_run();
        let json = run.to_json().expect("serialize");
        let deserialized = BenchmarkRun::from_json(&json).expect("deserialize");
        assert_eq!(run, deserialized);
    }

    #[test]
    fn display_produces_table() {
        let run = sample_run();
        let output = format!("{run}");
        assert!(output.contains("BM25 Turbo Benchmark Results"));
        assert!(output.contains("scifact"));
        assert!(output.contains("0.6823"));
    }
}
