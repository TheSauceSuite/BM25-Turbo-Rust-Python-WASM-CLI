use std::collections::HashMap;

/// Computes Discounted Cumulative Gain for the first `k` items.
///
/// `gains[i]` is the relevance gain at rank i (0-indexed).
/// DCG = sum_{i=0}^{k-1} gain_i / log2(i + 2)
pub fn dcg(gains: &[f64], k: usize) -> f64 {
    gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &gain)| gain / (i as f64 + 2.0).log2())
        .sum()
}

/// Computes the Ideal DCG: the DCG of the best possible ranking.
///
/// Sorts relevances in descending order, converts to f64, then computes DCG@k.
pub fn ideal_dcg(relevances: &mut [u32], k: usize) -> f64 {
    relevances.sort_unstable_by(|a, b| b.cmp(a));
    let gains: Vec<f64> = relevances.iter().map(|&r| r as f64).collect();
    dcg(&gains, k)
}

/// Computes nDCG@k for a single query.
///
/// `ranked_doc_ids` is the list of retrieved document IDs in rank order.
/// `qrels` maps doc_id -> relevance_grade for the query.
/// Returns 0.0 if there are no relevant documents (IDCG = 0).
pub fn ndcg_at_k(ranked_doc_ids: &[String], qrels: &HashMap<String, u32>, k: usize) -> f64 {
    if ranked_doc_ids.is_empty() || qrels.is_empty() || k == 0 {
        return 0.0;
    }

    // Build gains from the ranked results
    let gains: Vec<f64> = ranked_doc_ids
        .iter()
        .take(k)
        .map(|doc_id| *qrels.get(doc_id).unwrap_or(&0) as f64)
        .collect();

    let actual_dcg = dcg(&gains, k);

    // Compute ideal DCG from all relevance grades
    let mut all_relevances: Vec<u32> = qrels.values().copied().collect();
    let idcg = ideal_dcg(&mut all_relevances, k);

    if idcg == 0.0 {
        return 0.0;
    }

    actual_dcg / idcg
}

/// Computes mean nDCG@k across multiple queries.
///
/// Each element is (ranked_doc_ids, qrels_for_query).
/// Returns 0.0 if the input is empty.
pub fn mean_ndcg(per_query_results: &[(Vec<String>, &HashMap<String, u32>)], k: usize) -> f64 {
    if per_query_results.is_empty() {
        return 0.0;
    }

    let total: f64 = per_query_results
        .iter()
        .map(|(ranked, qrels)| ndcg_at_k(ranked, qrels, k))
        .sum();

    total / per_query_results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcg_basic() {
        // gains = [3, 2, 3, 0, 1, 2]
        // DCG@6 = 3/log2(2) + 2/log2(3) + 3/log2(4) + 0/log2(5) + 1/log2(6) + 2/log2(7)
        //       = 3.0 + 1.2618... + 1.5 + 0.0 + 0.3868... + 0.7124...
        //       = 6.8611...
        let gains = vec![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let result = dcg(&gains, 6);
        assert!((result - 6.8611).abs() < 0.01, "DCG@6 = {result}");
    }

    #[test]
    fn dcg_truncated_at_k() {
        let gains = vec![3.0, 2.0, 1.0, 0.0];
        let at_2 = dcg(&gains, 2);
        // DCG@2 = 3/log2(2) + 2/log2(3) = 3.0 + 1.2618 = 4.2618
        assert!((at_2 - 4.2618).abs() < 0.01, "DCG@2 = {at_2}");
    }

    #[test]
    fn ideal_dcg_sorts_descending() {
        let mut rels = vec![1, 3, 2, 0];
        let idcg = ideal_dcg(&mut rels, 4);
        // Sorted: [3, 2, 1, 0]
        // IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
        //      = 3.0 + 1.2618 + 0.5 + 0.0 = 4.7618
        assert!((idcg - 4.7618).abs() < 0.01, "IDCG = {idcg}");
    }

    #[test]
    fn ndcg_hand_computed() {
        // Ranked results: doc1(rel=3), doc2(rel=2), doc3(rel=0), doc4(rel=1), doc5(rel=0)
        let ranked = vec![
            "doc1".to_string(),
            "doc2".to_string(),
            "doc3".to_string(),
            "doc4".to_string(),
            "doc5".to_string(),
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 3);
        qrels.insert("doc2".to_string(), 2);
        qrels.insert("doc4".to_string(), 1);

        // Actual gains: [3, 2, 0, 1, 0]
        // DCG@5 = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
        //       = 3.0 + 1.26186 + 0.0 + 0.43067 + 0.0 = 4.69253

        // Ideal: sorted [3, 2, 1] (only non-zero) padded to 5: [3, 2, 1, 0, 0]
        // IDCG@5 = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0 + 0
        //        = 3.0 + 1.26186 + 0.5 = 4.76186

        // nDCG@5 = 4.69253 / 4.76186 = 0.9854
        let ndcg = ndcg_at_k(&ranked, &qrels, 5);
        assert!(
            (ndcg - 0.9854).abs() < 0.001,
            "nDCG@5 = {ndcg}, expected ~0.9854"
        );
    }

    #[test]
    fn ndcg_no_relevant_docs() {
        let ranked = vec!["doc1".to_string(), "doc2".to_string()];
        let qrels = HashMap::new();
        assert_eq!(ndcg_at_k(&ranked, &qrels, 10), 0.0);
    }

    #[test]
    fn ndcg_all_zero_relevance() {
        let ranked = vec!["doc1".to_string(), "doc2".to_string()];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 0);
        qrels.insert("doc2".to_string(), 0);
        assert_eq!(ndcg_at_k(&ranked, &qrels, 10), 0.0);
    }

    #[test]
    fn ndcg_fewer_than_k_results() {
        // Only 2 results but k=10
        let ranked = vec!["doc1".to_string(), "doc2".to_string()];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 3);
        qrels.insert("doc2".to_string(), 2);

        let ndcg = ndcg_at_k(&ranked, &qrels, 10);
        // Should still compute correctly: DCG has only 2 terms
        // DCG = 3/log2(2) + 2/log2(3) = 3.0 + 1.26186 = 4.26186
        // IDCG@10 with rels [3,2]: same = 4.26186
        // nDCG = 1.0
        assert!(
            (ndcg - 1.0).abs() < 0.001,
            "nDCG should be 1.0 for perfect ranking, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_empty_input() {
        let empty_ranked: Vec<String> = vec![];
        let qrels = HashMap::new();
        assert_eq!(ndcg_at_k(&empty_ranked, &qrels, 10), 0.0);
    }

    #[test]
    fn ndcg_k_zero() {
        let ranked = vec!["doc1".to_string()];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 3);
        assert_eq!(ndcg_at_k(&ranked, &qrels, 0), 0.0);
    }

    #[test]
    fn mean_ndcg_basic() {
        let qrels1: HashMap<String, u32> =
            [("d1".to_string(), 3), ("d2".to_string(), 1)].into();
        let qrels2: HashMap<String, u32> =
            [("d3".to_string(), 2)].into();

        let results: Vec<(Vec<String>, &HashMap<String, u32>)> = vec![
            (vec!["d1".to_string(), "d2".to_string()], &qrels1),
            (vec!["d3".to_string()], &qrels2),
        ];

        let mean = mean_ndcg(&results, 10);
        // Both queries have perfect ranking -> both nDCG = 1.0 -> mean = 1.0
        assert!(
            (mean - 1.0).abs() < 0.001,
            "mean nDCG = {mean}, expected 1.0"
        );
    }

    #[test]
    fn mean_ndcg_empty() {
        let results: Vec<(Vec<String>, &HashMap<String, u32>)> = vec![];
        assert_eq!(mean_ndcg(&results, 10), 0.0);
    }
}
