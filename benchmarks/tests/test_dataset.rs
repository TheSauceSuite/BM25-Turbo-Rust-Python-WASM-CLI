use std::path::Path;

use bm25_turbo_bench::dataset::{parse_corpus, parse_qrels, parse_queries};

fn fixtures_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

/// TEST-P2-005: Corpus JSONL Parsing
#[test]
fn parse_corpus_fixture() {
    let corpus = parse_corpus(&fixtures_dir().join("corpus.jsonl")).expect("parse corpus");

    assert_eq!(corpus.doc_ids.len(), 10, "expected 10 documents");
    assert_eq!(corpus.texts.len(), 10);

    // Verify IDs
    for i in 0..10 {
        assert_eq!(corpus.doc_ids[i], format!("doc{i}"));
    }

    // Verify title + " " + text concatenation
    assert!(
        corpus.texts[0].starts_with("Introduction to Information Retrieval "),
        "text should start with title: {}",
        &corpus.texts[0]
    );
    assert!(corpus.texts[0].contains("Information retrieval is the activity"));

    // Verify doc1
    assert!(corpus.texts[1].starts_with("BM25 Scoring "));
    assert!(corpus.texts[1].contains("ranking function"));
}

/// TEST-P2-006: Queries JSONL Parsing
#[test]
fn parse_queries_fixture() {
    let queries = parse_queries(&fixtures_dir().join("queries.jsonl")).expect("parse queries");

    assert_eq!(queries.query_ids.len(), 3, "expected 3 queries");
    assert_eq!(queries.texts.len(), 3);

    assert_eq!(queries.query_ids[0], "q0");
    assert_eq!(queries.texts[0], "what is BM25 scoring");

    assert_eq!(queries.query_ids[1], "q1");
    assert_eq!(queries.texts[1], "how does an inverted index work");

    assert_eq!(queries.query_ids[2], "q2");
    assert_eq!(queries.texts[2], "information retrieval evaluation metrics");
}

/// TEST-P2-007: Qrels TSV Parsing
#[test]
fn parse_qrels_fixture() {
    let qrels = parse_qrels(&fixtures_dir().join("qrels.tsv")).expect("parse qrels");
    let map = &qrels.0;

    // 3 queries
    assert_eq!(map.len(), 3, "expected 3 queries in qrels");

    // q0: doc1=3, doc9=2, doc5=1
    let q0 = map.get("q0").expect("q0 should exist");
    assert_eq!(q0.len(), 3);
    assert_eq!(q0["doc1"], 3);
    assert_eq!(q0["doc9"], 2);
    assert_eq!(q0["doc5"], 1);

    // q1: doc2=3, doc0=1
    let q1 = map.get("q1").expect("q1 should exist");
    assert_eq!(q1.len(), 2);
    assert_eq!(q1["doc2"], 3);
    assert_eq!(q1["doc0"], 1);

    // q2: doc7=3, doc6=2, doc8=1
    let q2 = map.get("q2").expect("q2 should exist");
    assert_eq!(q2.len(), 3);
    assert_eq!(q2["doc7"], 3);
    assert_eq!(q2["doc6"], 2);
    assert_eq!(q2["doc8"], 1);
}
