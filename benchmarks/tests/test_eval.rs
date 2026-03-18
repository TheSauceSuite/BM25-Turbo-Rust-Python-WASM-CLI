use std::collections::HashMap;

use bm25_turbo_bench::{ndcg_at_k, BenchmarkRun, DatasetResult};

/// TEST-P2-001: nDCG Hand-Computed Example
/// Ranked: doc1(rel=3), doc2(rel=2), doc3(rel=0), doc4(rel=1), doc5(rel=0)
/// Expected nDCG@5 = 0.9854
#[test]
fn ndcg_hand_computed_example() {
    let ranked = vec![
        "doc1".to_string(),
        "doc2".to_string(),
        "doc3".to_string(),
        "doc4".to_string(),
        "doc5".to_string(),
    ];
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 3u32);
    qrels.insert("doc2".to_string(), 2);
    qrels.insert("doc4".to_string(), 1);

    // DCG@5 = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
    //       = 3.0 + 1.26186 + 0.0 + 0.43067 + 0.0 = 4.69253
    // IDCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0 + 0
    //        = 3.0 + 1.26186 + 0.5 = 4.76186
    // nDCG@5 = 4.69253 / 4.76186 = 0.9854
    let ndcg = ndcg_at_k(&ranked, &qrels, 5);
    assert!(
        (ndcg - 0.9854).abs() < 0.001,
        "nDCG@5 = {ndcg}, expected ~0.9854"
    );
}

/// TEST-P2-002: nDCG No Relevant Documents
#[test]
fn ndcg_no_relevant_documents() {
    let ranked = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];

    // Empty qrels -> 0.0
    let empty_qrels = HashMap::new();
    assert_eq!(ndcg_at_k(&ranked, &empty_qrels, 10), 0.0);

    // All-zero relevance -> 0.0
    let mut zero_qrels = HashMap::new();
    zero_qrels.insert("doc1".to_string(), 0u32);
    zero_qrels.insert("doc2".to_string(), 0);
    assert_eq!(ndcg_at_k(&ranked, &zero_qrels, 10), 0.0);
}

/// TEST-P2-003: nDCG Fewer Than k Results
#[test]
fn ndcg_fewer_than_k_results() {
    let ranked = vec!["doc1".to_string(), "doc2".to_string()];
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 3u32);
    qrels.insert("doc2".to_string(), 2);

    // k=10 but only 2 results, perfect ranking
    let ndcg = ndcg_at_k(&ranked, &qrels, 10);
    assert!(
        (ndcg - 1.0).abs() < 0.001,
        "nDCG should be 1.0 for perfect ranking with fewer results, got {ndcg}"
    );
}

/// TEST-P2-004: nDCG Empty Input
#[test]
fn ndcg_empty_input() {
    let empty_ranked: Vec<String> = vec![];
    let qrels = HashMap::new();
    assert_eq!(ndcg_at_k(&empty_ranked, &qrels, 10), 0.0);

    let mut nonempty_qrels = HashMap::new();
    nonempty_qrels.insert("doc1".to_string(), 3u32);
    assert_eq!(ndcg_at_k(&empty_ranked, &nonempty_qrels, 10), 0.0);
}

/// TEST-P2-008: Result Struct JSON Round-Trip
#[test]
fn result_struct_json_round_trip() {
    let run = BenchmarkRun {
        timestamp: "2026-03-17T12:00:00Z".to_string(),
        system_info: "test-machine x86_64".to_string(),
        rust_version: "1.85.0".to_string(),
        datasets: vec![
            DatasetResult {
                name: "nq".to_string(),
                num_docs: 2_681_468,
                num_queries: 3452,
                index_time_ms: 45000.0,
                queries_per_sec: 8500.0,
                latency_p50_us: 95.2,
                latency_p95_us: 210.5,
                latency_p99_us: 450.8,
                ndcg_at_10: 0.3210,
                peak_memory_bytes: 500_000_000,
            },
            DatasetResult {
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
            },
        ],
    };

    let json = run.to_json().expect("serialization should succeed");
    let deserialized = BenchmarkRun::from_json(&json).expect("deserialization should succeed");
    assert_eq!(run, deserialized);

    // Verify key fields survive the round-trip
    assert_eq!(deserialized.datasets.len(), 2);
    assert_eq!(deserialized.datasets[0].name, "nq");
    assert_eq!(deserialized.datasets[1].num_docs, 5183);
    assert!((deserialized.datasets[1].ndcg_at_10 - 0.6823).abs() < f64::EPSILON);
}
