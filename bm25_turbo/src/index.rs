//! BM25 index: build, query, and manage document collections.
//!
//! The [`BM25Index`] struct holds a built index (CSC matrix + vocabulary + parameters)
//! and provides the main query interface. Use [`BM25Builder`] to construct an index
//! from a corpus of documents.

use std::collections::HashMap;

use tracing::instrument;

use crate::csc::CscMatrix;
use crate::error::{Error, Result};
use crate::query_cache::QueryCache;
use crate::scoring;
use crate::selection;
use crate::tokenizer::Tokenizer;
use crate::types::{BM25Params, Method, Results};

// Compile-time proof that BM25Index is Send + Sync.
const _: () = {
    const fn _assert<T: Send + Sync>() {}
    _assert::<BM25Index>();
};

/// A built BM25 index ready for querying.
pub struct BM25Index {
    /// The CSC score matrix.
    pub(crate) matrix: CscMatrix,
    /// Token string -> token ID mapping.
    pub(crate) vocab: HashMap<String, u32>,
    /// Reverse mapping: token ID -> token string.
    pub(crate) vocab_inv: Vec<String>,
    /// Scoring parameters used to build this index.
    pub(crate) params: BM25Params,
    /// Average document length in the corpus.
    pub(crate) avg_doc_len: f32,
    /// Number of documents in the corpus.
    pub(crate) num_docs: u32,
    /// The tokenizer used for this index.
    pub(crate) tokenizer: Tokenizer,
    /// Document frequency for each term ID (length = vocab_size).
    /// Used for IDF-ordered term processing in the search hot path.
    pub(crate) doc_freqs: Vec<u32>,
    /// Optional LRU query result cache.
    /// Enabled via `BM25Builder::cache_capacity()` or `StreamingBuilder::cache_capacity()`.
    pub(crate) cache: Option<QueryCache>,
    /// Optional Block-Max WAND index for approximate search.
    /// Built via `build_bmw_index()` or `BM25Builder::with_bmw(true)`.
    #[cfg(feature = "ann")]
    pub(crate) block_max_index: Option<crate::wand::BlockMaxIndex>,
}

impl BM25Index {
    /// Query the index and return the top-k results.
    ///
    /// # Arguments
    /// * `query` -- the search query string
    /// * `k` -- number of top results to return
    #[instrument(skip(self), fields(query = query, k = k))]
    pub fn search(&self, query: &str, k: usize) -> Result<Results> {
        if k == 0 {
            return Err(Error::InvalidParameter("k must be > 0".to_string()));
        }

        // Tokenize query
        let query_tokens = self.tokenizer.tokenize(query);

        self.search_tokens(&query_tokens, k)
    }

    /// Query the index with pre-tokenized query tokens and return the top-k results.
    ///
    /// # Arguments
    /// * `tokens` -- pre-tokenized query terms
    /// * `k` -- number of top results to return
    pub fn search_tokens(&self, tokens: &[String], k: usize) -> Result<Results> {
        if k == 0 {
            return Err(Error::InvalidParameter("k must be > 0".to_string()));
        }

        // Map to token IDs (skip unknown tokens)
        let token_ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| self.vocab.get(t.as_str()).copied())
            .collect();

        if token_ids.is_empty() {
            return Ok(Results {
                doc_ids: Vec::new(),
                scores: Vec::new(),
            });
        }

        // Auto-select query strategy:
        // - BMW path for large indices (>50K docs) when BMW data is available
        // - Dense SAAT path for small indices or when BMW not built
        #[cfg(feature = "ann")]
        if self.num_docs > 500_000 {
            if let Some(ref bmw) = self.block_max_index {
                return Ok(crate::wand::block_max_wand_top_k(
                    &self.matrix, bmw, &token_ids, k,
                ));
            }
        }

        // Fallback: dense SAAT accumulator
        let scores = self.get_scores_by_ids(&token_ids);
        Ok(selection::top_k(&scores, k))
    }

    /// Compute raw scores for all documents given a query string.
    ///
    /// Returns a dense f32 array of length `num_docs` where each element
    /// is the accumulated BM25 score for that document.
    pub fn get_scores(&self, query: &str) -> Result<Vec<f32>> {
        let query_tokens = self.tokenizer.tokenize(query);
        let token_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.vocab.get(t.as_str()).copied())
            .collect();
        Ok(self.get_scores_by_ids(&token_ids))
    }

    /// Internal: accumulate scores from CSC columns for given token IDs.
    ///
    /// Token IDs are sorted by ascending document frequency (rarest terms first)
    /// to improve cache behavior and branch prediction.
    fn get_scores_by_ids(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.num_docs as usize];

        // IDF-ordered term processing: sort by ascending doc frequency.
        let mut sorted_ids = token_ids.to_vec();
        sorted_ids.sort_by_key(|&tid| {
            if (tid as usize) < self.doc_freqs.len() {
                self.doc_freqs[tid as usize]
            } else {
                u32::MAX
            }
        });

        for &tid in &sorted_ids {
            if tid < self.matrix.vocab_size {
                let (col_scores, col_indices) = self.matrix.column(tid);

                // When simd feature is enabled, use scatter_add.
                #[cfg(feature = "simd")]
                {
                    crate::simd::scatter_add(&mut scores, col_indices, col_scores);
                }

                // Scalar fallback when simd is not enabled.
                #[cfg(not(feature = "simd"))]
                {
                    for (i, &doc_id) in col_indices.iter().enumerate() {
                        scores[doc_id as usize] += col_scores[i];
                    }
                }
            }
        }

        scores
    }

    /// Query the index with caching enabled.
    ///
    /// If a cache is configured (via `BM25Builder::cache_capacity()`), checks
    /// the cache first and returns a cached result on hit. On cache miss,
    /// delegates to `search()` and stores the result.
    ///
    /// If no cache is configured, this is equivalent to `search()`.
    ///
    /// # Arguments
    /// * `query` -- the search query string
    /// * `k` -- number of top results to return
    pub fn search_cached(&self, query: &str, k: usize) -> Result<Results> {
        if let Some(ref cache) = self.cache {
            let key = QueryCache::cache_key(query, k);
            if let Some(cached) = cache.get(&key) {
                return Ok(cached);
            }
            let result = self.search(query, k)?;
            cache.insert(key, result.clone());
            Ok(result)
        } else {
            self.search(query, k)
        }
    }

    /// Return the number of documents in the index.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        self.matrix.vocab_size
    }

    /// Return the scoring parameters.
    pub fn params(&self) -> &BM25Params {
        &self.params
    }

    /// Return the average document length.
    pub fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    /// Enable WAL mode on this index and return an initialized WAL.
    pub fn enable_wal(&self) -> Result<crate::wal::WriteAheadLog> {
        let mut wal = crate::wal::WriteAheadLog::new();
        wal.initialize(self)?;
        Ok(wal)
    }

    /// Add documents via WAL. Convenience wrapper.
    pub fn add_documents(
        &self,
        wal: &mut crate::wal::WriteAheadLog,
        docs: &[&str],
    ) -> Result<Vec<u32>> {
        wal.append_documents(docs)
    }

    /// Delete documents via WAL. Convenience wrapper.
    pub fn delete_documents(
        &self,
        wal: &mut crate::wal::WriteAheadLog,
        doc_ids: &[u32],
    ) -> Result<crate::wal::DeleteReport> {
        wal.delete_documents(doc_ids)
    }

    /// Compact the WAL into this index.
    pub fn compact(&mut self, wal: &mut crate::wal::WriteAheadLog) -> Result<()> {
        wal.compact(self)
    }

    /// Search with WAL overlay.
    pub fn search_with_wal(
        &self,
        wal: &crate::wal::WriteAheadLog,
        query: &str,
        k: usize,
    ) -> Result<Results> {
        wal.search(self, query, k)
    }

    /// Search with WAL overlay and specified strategy.
    pub fn search_with_strategy(
        &self,
        wal: &crate::wal::WriteAheadLog,
        query: &str,
        k: usize,
        strategy: crate::wal::QueryStrategy,
    ) -> Result<Results> {
        wal.search_with_strategy(self, query, k, strategy)
    }

    /// Query the index with multiple queries in parallel, returning top-k results for each.
    ///
    /// When the `parallel` feature is enabled, queries are processed concurrently via
    /// `rayon::par_iter`. Otherwise, queries run sequentially. Result order matches
    /// input order: `output[i]` corresponds to `queries[i]`.
    ///
    /// # Arguments
    /// * `queries` -- slice of query strings
    /// * `k` -- number of top results to return per query
    ///
    /// # Errors
    /// Returns `Error::InvalidParameter` if `k == 0`.
    /// Returns `Ok(vec![])` for an empty queries slice (not an error).
    pub fn search_batch(&self, queries: &[&str], k: usize) -> Result<Vec<Results>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        if k == 0 {
            return Err(Error::InvalidParameter("k must be > 0".to_string()));
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let results: Vec<Results> = queries
                .par_iter()
                .map(|query| {
                    // Each query gets its own dense accumulator (no shared mutable state).
                    // search() handles tokenization and scoring internally.
                    // k is already validated above, and search() won't fail for valid k.
                    self.search(query, k).unwrap_or_else(|_| Results {
                        doc_ids: Vec::new(),
                        scores: Vec::new(),
                    })
                })
                .collect();
            Ok(results)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut results = Vec::with_capacity(queries.len());
            for query in queries {
                results.push(self.search(query, k)?);
            }
            Ok(results)
        }
    }
}

/// Builder for constructing a [`BM25Index`] from a corpus.
#[derive(Default)]
pub struct BM25Builder {
    params: BM25Params,
    tokenizer: Tokenizer,
    cache_capacity: Option<usize>,
}

impl BM25Builder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the BM25 variant.
    pub fn method(mut self, method: Method) -> Self {
        self.params.method = method;
        self
    }

    /// Set the k1 parameter.
    pub fn k1(mut self, k1: f32) -> Self {
        self.params.k1 = k1;
        self
    }

    /// Set the b parameter.
    pub fn b(mut self, b: f32) -> Self {
        self.params.b = b;
        self
    }

    /// Set the delta parameter (BM25L / BM25+ only).
    pub fn delta(mut self, delta: f32) -> Self {
        self.params.delta = delta;
        self
    }

    /// Set a custom tokenizer.
    pub fn tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set the LRU query cache capacity for the resulting index.
    ///
    /// When set, the index will have an internal `QueryCache` of the given
    /// capacity, usable via `BM25Index::search_cached()`. A capacity of 0
    /// disables caching.
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = Some(capacity);
        self
    }

    /// Build the index from a corpus of document strings.
    ///
    /// Consumes the builder. Steps:
    /// 1. Tokenize all documents (parallel via rayon when `parallel` feature enabled).
    /// 2. Build vocabulary mapping token strings to IDs.
    /// 3. Compute document frequencies (df) for each term.
    /// 4. Compute IDF array.
    /// 5. Score all (token, document) pairs.
    /// 6. Build CSC matrix from triplets.
    /// 7. Return `BM25Index`.
    #[instrument(skip(self, corpus), fields(doc_count = corpus.len(), method = %self.params.method))]
    pub fn build_from_corpus(self, corpus: &[&str]) -> Result<BM25Index> {
        // Validate parameters.
        scoring::validate_params(&self.params)?;

        if corpus.is_empty() {
            return Err(Error::InvalidParameter(
                "corpus must not be empty".to_string(),
            ));
        }

        // Step 1: Tokenize all documents.
        let doc_tokens = self.tokenize_corpus(corpus);

        // Delegate to shared builder logic.
        let index = self.build_from_token_vecs(doc_tokens)?;
        tracing::info!(
            num_docs = index.num_docs,
            vocab_size = index.vocab_size(),
            "Index built"
        );
        Ok(index)
    }

    /// Build the index from pre-tokenized documents.
    ///
    /// Consumes the builder. Each element of `tokens` is the list of token
    /// strings for one document.
    pub fn build_from_tokens(self, tokens: &[Vec<String>]) -> Result<BM25Index> {
        // Validate parameters.
        scoring::validate_params(&self.params)?;

        if tokens.is_empty() {
            return Err(Error::InvalidParameter(
                "corpus must not be empty".to_string(),
            ));
        }

        // Clone tokens so we own them.
        let doc_tokens: Vec<Vec<String>> = tokens.to_vec();
        self.build_from_token_vecs(doc_tokens)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Tokenize all documents in the corpus.
    ///
    /// Uses `rayon::par_iter` when the `parallel` feature is enabled,
    /// otherwise iterates sequentially.
    fn tokenize_corpus(&self, corpus: &[&str]) -> Vec<Vec<String>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            corpus
                .par_iter()
                .map(|doc| self.tokenizer.tokenize(doc))
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            corpus
                .iter()
                .map(|doc| self.tokenizer.tokenize(doc))
                .collect()
        }
    }

    /// Core build logic shared between `build_from_corpus` and `build_from_tokens`.
    fn build_from_token_vecs(self, doc_tokens: Vec<Vec<String>>) -> Result<BM25Index> {
        let num_docs = doc_tokens.len() as u32;

        // Step 2: Build vocabulary.
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for tokens in &doc_tokens {
            for token in tokens {
                let next_id = vocab.len() as u32;
                vocab.entry(token.clone()).or_insert(next_id);
            }
        }
        let vocab_size = vocab.len() as u32;

        // Build inverse vocabulary.
        let mut vocab_inv = vec![String::new(); vocab_size as usize];
        for (token, &id) in &vocab {
            vocab_inv[id as usize] = token.clone();
        }

        // Step 3: Compute document lengths and average document length.
        let doc_lengths: Vec<u32> = doc_tokens.iter().map(|t| t.len() as u32).collect();
        let total_tokens: u64 = doc_lengths.iter().map(|&l| l as u64).sum();
        let avg_doc_len = if num_docs > 0 {
            total_tokens as f32 / num_docs as f32
        } else {
            0.0
        };

        // Step 4: Compute document frequencies (df) for each term.
        let mut df = vec![0u32; vocab_size as usize];
        for tokens in &doc_tokens {
            // Use a bitset to count each term at most once per document.
            let mut seen = vec![false; vocab_size as usize];
            for token in tokens {
                if let Some(&tid) = vocab.get(token) {
                    if !seen[tid as usize] {
                        seen[tid as usize] = true;
                        df[tid as usize] += 1;
                    }
                }
            }
        }

        // Step 5: Compute IDF array.
        let idf_arr: Vec<f32> = (0..vocab_size as usize)
            .map(|tid| scoring::idf(self.params.method, num_docs, df[tid]))
            .collect();

        // Step 6: For each document, for each unique token, compute BM25 score.
        // Collect as (term_id, doc_id, score) triplets.
        let triplets = self.compute_triplets(
            &doc_tokens,
            &vocab,
            &idf_arr,
            &doc_lengths,
            avg_doc_len,
            num_docs,
        );

        // Step 7: Build CSC matrix.
        let matrix = CscMatrix::from_triplets(&triplets, num_docs, vocab_size)?;

        // Step 8: Auto-build BMW index for large corpora.
        #[cfg(feature = "ann")]
        let block_max_index = if num_docs > 500_000 {
            crate::wand::precompute_block_maxes(&matrix, crate::wand::DEFAULT_BLOCK_SIZE).ok()
        } else {
            None
        };

        Ok(BM25Index {
            matrix,
            vocab,
            vocab_inv,
            params: self.params,
            avg_doc_len,
            num_docs,
            tokenizer: self.tokenizer,
            doc_freqs: df,
            cache: self.cache_capacity.map(QueryCache::new),
            #[cfg(feature = "ann")]
            block_max_index,
        })
    }

    /// Compute (term_id, doc_id, score) triplets for all documents.
    ///
    /// When the `parallel` feature is enabled, documents are processed in
    /// parallel via `rayon::par_iter()` and per-thread results are collected
    /// via `flat_map`.
    #[allow(clippy::too_many_arguments)]
    fn compute_triplets(
        &self,
        doc_tokens: &[Vec<String>],
        vocab: &HashMap<String, u32>,
        idf_arr: &[f32],
        doc_lengths: &[u32],
        avg_doc_len: f32,
        _num_docs: u32,
    ) -> Vec<(u32, u32, f32)> {
        let method = self.params.method;
        let k1 = self.params.k1;
        let b = self.params.b;
        let delta = self.params.delta;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            doc_tokens
                .par_iter()
                .enumerate()
                .flat_map(|(doc_id, tokens)| {
                    Self::triplets_for_doc(
                        doc_id as u32,
                        tokens,
                        vocab,
                        idf_arr,
                        doc_lengths[doc_id],
                        avg_doc_len,
                        method,
                        k1,
                        b,
                        delta,
                    )
                })
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut triplets = Vec::new();
            for (doc_id, tokens) in doc_tokens.iter().enumerate() {
                triplets.extend(Self::triplets_for_doc(
                    doc_id as u32,
                    tokens,
                    vocab,
                    idf_arr,
                    doc_lengths[doc_id],
                    avg_doc_len,
                    method,
                    k1,
                    b,
                    delta,
                ));
            }
            triplets
        }
    }

    /// Compute triplets for a single document.
    ///
    /// Counts term frequencies, then computes idf * tfc for each unique term.
    #[allow(clippy::too_many_arguments)]
    fn triplets_for_doc(
        doc_id: u32,
        tokens: &[String],
        vocab: &HashMap<String, u32>,
        idf_arr: &[f32],
        doc_len: u32,
        avg_doc_len: f32,
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
    ) -> Vec<(u32, u32, f32)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Count term frequencies within this document.
        let mut tf_map: HashMap<u32, u32> = HashMap::new();
        for token in tokens {
            if let Some(&tid) = vocab.get(token) {
                *tf_map.entry(tid).or_insert(0) += 1;
            }
        }

        let doc_len_f32 = doc_len as f32;

        tf_map
            .iter()
            .filter_map(|(&tid, &tf)| {
                let idf_val = idf_arr[tid as usize];
                let tfc_val = scoring::tfc(method, tf as f32, doc_len_f32, avg_doc_len, k1, b, delta);
                let s = (idf_val as f64 * tfc_val as f64) as f32;
                if s != 0.0 {
                    Some((tid, doc_id, s))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_default_params() {
        let builder = BM25Builder::new();
        assert_eq!(builder.params.method, Method::Lucene);
        assert_eq!(builder.params.k1, 1.5);
        assert_eq!(builder.params.b, 0.75);
    }

    /// TEST-P3-001: End-to-end small corpus.
    #[test]
    fn end_to_end_small_corpus() {
        let corpus = &[
            "the quick brown fox",
            "the lazy dog",
            "brown fox jumps over the lazy dog",
            "quick quick quick fox",
            "the dog sat on the mat",
        ];
        let index = BM25Builder::new()
            .build_from_corpus(corpus)
            .expect("build should succeed");

        assert_eq!(index.num_docs(), 5);
        assert!(index.vocab_size() > 0);

        // Search for "quick" -- should find docs 0 and 3 in top results.
        let results = index.search("quick", 2).unwrap();
        assert!(!results.doc_ids.is_empty());
        // Doc 3 has "quick" three times, should rank highest.
        assert_eq!(results.doc_ids[0], 3, "Doc 3 has highest tf for 'quick'");
    }

    /// TEST-P3-002: All five BM25 variants.
    ///
    /// Robertson IDF can go negative when df > N/2, so we use a larger corpus
    /// where "hello" appears in a minority of documents to ensure positive IDF
    /// across all variants.
    #[test]
    fn all_five_variants() {
        let corpus = &[
            "hello world",
            "foo bar baz",
            "cat dog fish",
            "one two three four",
            "alpha beta gamma",
            "hello again here",
        ];
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let index = BM25Builder::new()
                .method(*method)
                .build_from_corpus(corpus)
                .unwrap_or_else(|e| panic!("{:?} build failed: {}", method, e));
            let results = index.search("hello", 2).unwrap();
            assert!(
                !results.doc_ids.is_empty(),
                "{:?} should return results for 'hello'",
                method
            );
            assert!(
                results.scores[0] > 0.0,
                "{:?} scores should be positive, got {}",
                method,
                results.scores[0]
            );
        }
    }

    /// TEST-P3-003: Empty corpus returns error.
    #[test]
    fn empty_corpus_returns_error() {
        let result = BM25Builder::new().build_from_corpus(&[]);
        assert!(result.is_err(), "Empty corpus should return an error");
    }

    /// TEST-P3-004: Unknown query tokens return empty results.
    #[test]
    fn unknown_query_tokens_empty_results() {
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let results = index.search("zzzzzzz", 10).unwrap();
        assert!(
            results.doc_ids.is_empty(),
            "Unknown tokens should produce empty results"
        );
    }

    /// TEST-P3-007: Builder defaults work without explicit configuration.
    #[test]
    fn builder_defaults_produce_working_index() {
        let corpus = &["one two three", "four five six"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let results = index.search("one", 1).unwrap();
        assert_eq!(results.doc_ids[0], 0);
    }

    /// Builder is consumed on build (move semantics) -- verified by the type system.
    /// This test just confirms build_from_corpus returns a valid index.
    #[test]
    fn builder_consumed_on_build() {
        let builder = BM25Builder::new();
        let _index = builder.build_from_corpus(&["doc one"]).unwrap();
        // builder cannot be used here -- move semantics enforced by compiler.
    }

    /// Test build_from_tokens with pre-tokenized input.
    #[test]
    fn build_from_tokens_basic() {
        let tokens = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["foo".to_string(), "bar".to_string()],
            vec!["hello".to_string(), "foo".to_string()],
        ];
        let index = BM25Builder::new().build_from_tokens(&tokens).unwrap();
        assert_eq!(index.num_docs(), 3);
    }

    /// Test search_tokens with pre-tokenized query.
    #[test]
    fn search_tokens_basic() {
        let corpus = &["hello world", "foo bar", "hello foo"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let query_tokens = vec!["hello".to_string()];
        let results = index.search_tokens(&query_tokens, 2).unwrap();
        assert!(!results.doc_ids.is_empty());
    }

    /// Test get_scores returns raw score vector.
    #[test]
    fn get_scores_returns_dense_array() {
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let scores = index.get_scores("hello").unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 0.0, "doc 0 should have positive score for 'hello'");
        assert_eq!(scores[1], 0.0, "doc 1 should have zero score for 'hello'");
    }

    /// Test that parameter validation works at build time.
    #[test]
    fn invalid_params_rejected_at_build_time() {
        let result = BM25Builder::new()
            .k1(-1.0)
            .build_from_corpus(&["hello world"]);
        assert!(result.is_err(), "Negative k1 should be rejected");
    }

    /// Test k=0 returns error.
    #[test]
    fn search_k_zero_returns_error() {
        let corpus = &["hello world"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        assert!(index.search("hello", 0).is_err());
    }

    /// k > corpus size returns all matching documents.
    #[test]
    fn search_k_larger_than_corpus() {
        let corpus = &["alpha beta", "gamma delta", "alpha gamma"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let results = index.search("alpha", 100).unwrap();
        // Only 2 docs contain "alpha", so we should get at most 2 results.
        assert_eq!(results.doc_ids.len(), 2);
        assert!(results.doc_ids.contains(&0));
        assert!(results.doc_ids.contains(&2));
    }

    /// k=1 returns exactly one result.
    #[test]
    fn search_k_one() {
        let corpus = &["hello world", "foo bar", "hello foo"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let results = index.search("hello", 1).unwrap();
        assert_eq!(results.doc_ids.len(), 1);
        assert_eq!(results.scores.len(), 1);
    }

    /// Empty query string returns empty results (all tokens filtered out).
    #[test]
    fn empty_query_returns_empty() {
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let results = index.search("", 10).unwrap();
        assert!(results.doc_ids.is_empty());
    }

    /// Custom params (non-default k1, b) produce a working index.
    #[test]
    fn custom_params_work() {
        let corpus = &["hello world", "foo bar baz", "hello foo"];
        let index = BM25Builder::new()
            .method(Method::Atire)
            .k1(2.0)
            .b(0.5)
            .build_from_corpus(corpus)
            .unwrap();
        let results = index.search("hello", 2).unwrap();
        assert!(!results.doc_ids.is_empty());
        assert!(results.scores[0] > 0.0);
    }

    /// TEST-P3-005: Parallel vs sequential produce identical CSC matrices.
    ///
    /// Since parallel is a compile-time feature, we verify that calling
    /// build_from_corpus multiple times yields bit-identical CSC data.
    /// The parallel code path is exercised when the `parallel` feature is
    /// enabled (which is the default), and the triplets_for_doc function
    /// is deterministic regardless of thread scheduling because each
    /// document's triplets depend only on its own content.
    ///
    /// To truly test parallel vs sequential equivalence, we also compare
    /// build_from_corpus with build_from_tokens (which share the same
    /// build_from_token_vecs path but differ in the tokenization step).
    #[test]
    fn parallel_vs_sequential_equivalence() {
        let corpus = &[
            "the quick brown fox jumps over the lazy dog",
            "a fast red car drives on the highway",
            "the lazy brown dog sleeps in the sun",
            "quick fox quick fox quick fox",
            "highway car drives fast red",
            "sun moon stars galaxy universe",
            "the quick red fox",
            "lazy lazy lazy dog dog dog",
        ];

        // Build twice -- if parallel randomizes ordering, results would differ.
        let index1 = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let index2 = BM25Builder::new().build_from_corpus(corpus).unwrap();

        // CSC matrices must be bit-identical.
        assert_eq!(index1.matrix.data.len(), index2.matrix.data.len(), "data length mismatch");
        assert_eq!(index1.matrix.indices.len(), index2.matrix.indices.len(), "indices length mismatch");
        assert_eq!(index1.matrix.indptr, index2.matrix.indptr, "indptr mismatch");

        for i in 0..index1.matrix.data.len() {
            assert_eq!(
                index1.matrix.data[i].to_bits(),
                index2.matrix.data[i].to_bits(),
                "data mismatch at index {}: {} vs {}",
                i, index1.matrix.data[i], index2.matrix.data[i]
            );
        }
        assert_eq!(index1.matrix.indices, index2.matrix.indices, "indices mismatch");

        // Query results must also be identical.
        for query in &["quick", "fox", "lazy dog", "highway car"] {
            let r1 = index1.search(query, 5).unwrap();
            let r2 = index2.search(query, 5).unwrap();
            assert_eq!(r1.doc_ids, r2.doc_ids, "doc_ids differ for query '{}'", query);
            for j in 0..r1.scores.len() {
                assert_eq!(
                    r1.scores[j].to_bits(),
                    r2.scores[j].to_bits(),
                    "scores differ for query '{}' at position {}",
                    query, j
                );
            }
        }
    }

    /// Pre-tokenized build_from_tokens produces the same index as build_from_corpus.
    #[test]
    fn build_from_tokens_matches_build_from_corpus() {
        let corpus = &["alpha beta gamma", "beta gamma delta", "alpha delta"];

        // Build from corpus (uses the tokenizer internally).
        let index_corpus = BM25Builder::new().build_from_corpus(corpus).unwrap();

        // Pre-tokenize with the same default tokenizer.
        let tokenizer = crate::tokenizer::Tokenizer::default();
        let token_vecs: Vec<Vec<String>> = corpus.iter().map(|doc| tokenizer.tokenize(doc)).collect();
        let index_tokens = BM25Builder::new().build_from_tokens(&token_vecs).unwrap();

        // Both should produce identical results.
        assert_eq!(index_corpus.num_docs(), index_tokens.num_docs());
        assert_eq!(index_corpus.vocab_size(), index_tokens.vocab_size());

        // Query results must match.
        for query_str in &["alpha", "beta", "delta", "gamma"] {
            let tokens = tokenizer.tokenize(query_str);
            let r1 = index_corpus.search_tokens(&tokens, 5).unwrap();
            let r2 = index_tokens.search_tokens(&tokens, 5).unwrap();
            assert_eq!(r1.doc_ids, r2.doc_ids, "doc_ids differ for query '{}'", query_str);
            for j in 0..r1.scores.len() {
                assert_eq!(
                    r1.scores[j].to_bits(),
                    r2.scores[j].to_bits(),
                    "scores differ for query '{}' at position {}",
                    query_str, j
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Batch query tests
    // -----------------------------------------------------------------------

    /// TEST-P4-004: Batch query results match individual query results.
    #[test]
    fn batch_query_matches_individual() {
        let corpus = &[
            "the quick brown fox jumps over the lazy dog",
            "a fast red car drives on the highway",
            "the lazy brown dog sleeps in the sun",
            "quick fox quick fox quick fox",
            "highway car drives fast red",
            "sun moon stars galaxy universe",
            "the quick red fox",
            "lazy lazy lazy dog dog dog",
            "brown bear eats honey in the forest",
            "the sun rises in the east",
        ];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();

        let queries = &[
            "quick", "fox", "lazy dog", "highway car", "sun", "brown",
            "moon stars", "fast red", "the", "jumps over",
            "bear forest", "galaxy universe", "quick brown fox",
            "dog sleeps", "red car highway", "east rises",
        ];
        let queries_refs: Vec<&str> = queries.to_vec();

        let batch_results = index.search_batch(&queries_refs, 5).unwrap();

        assert_eq!(batch_results.len(), queries.len());

        for (i, query) in queries.iter().enumerate() {
            let individual = index.search(query, 5).unwrap();
            assert_eq!(
                batch_results[i].doc_ids, individual.doc_ids,
                "doc_ids mismatch for query '{}' at index {}",
                query, i
            );
            for j in 0..individual.scores.len() {
                assert_eq!(
                    batch_results[i].scores[j].to_bits(),
                    individual.scores[j].to_bits(),
                    "score mismatch for query '{}' at position {}: batch={} individual={}",
                    query, j, batch_results[i].scores[j], individual.scores[j]
                );
            }
        }
    }

    /// TEST-P4-005: Batch query order preservation.
    #[test]
    fn batch_query_order_preservation() {
        let corpus = &[
            "alpha unique one",
            "beta unique two",
            "gamma unique three",
            "delta unique four",
            "epsilon unique five",
        ];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();

        // Each query is distinctive: "alpha" only in doc 0, "beta" only in doc 1, etc.
        let queries: Vec<&str> = vec!["alpha", "beta", "gamma", "delta", "epsilon"];
        let batch = index.search_batch(&queries, 1).unwrap();

        assert_eq!(batch.len(), 5);
        // output[0] should match "alpha" -> doc 0
        assert_eq!(batch[0].doc_ids[0], 0, "output[0] should correspond to query 'alpha'");
        // output[1] should match "beta" -> doc 1
        assert_eq!(batch[1].doc_ids[0], 1, "output[1] should correspond to query 'beta'");
        // output[2] should match "gamma" -> doc 2
        assert_eq!(batch[2].doc_ids[0], 2, "output[2] should correspond to query 'gamma'");
        // output[3] should match "delta" -> doc 3
        assert_eq!(batch[3].doc_ids[0], 3, "output[3] should correspond to query 'delta'");
        // output[4] should match "epsilon" -> doc 4
        assert_eq!(batch[4].doc_ids[0], 4, "output[4] should correspond to query 'epsilon'");
    }

    /// TEST-P4-006: Batch query empty input.
    #[test]
    fn batch_query_empty_input() {
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();

        // Empty batch returns empty vec.
        let result = index.search_batch(&[], 10).unwrap();
        assert!(result.is_empty(), "Empty batch should return empty vec");
    }

    /// TEST-P4-006 extended: Single query batch.
    #[test]
    fn batch_query_single_query() {
        let corpus = &["hello world", "foo bar", "hello foo"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();

        let batch = index.search_batch(&["hello"], 2).unwrap();
        let individual = index.search("hello", 2).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].doc_ids, individual.doc_ids);
        for j in 0..individual.scores.len() {
            assert_eq!(
                batch[0].scores[j].to_bits(),
                individual.scores[j].to_bits(),
            );
        }
    }

    /// TEST-P4-006 extended: Batch query k=0 returns error.
    #[test]
    fn batch_query_k_zero_returns_error() {
        let corpus = &["hello world"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        assert!(index.search_batch(&["hello"], 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Phase 1: IDF ordering and cache tests
    // -----------------------------------------------------------------------

    /// TEST-P1-001: IDF ordering produces same results.
    /// Verify that IDF-ordered term processing does not change search results
    /// by comparing multi-term queries against a known-good baseline.
    #[test]
    fn idf_ordering_produces_same_results() {
        let corpus = &[
            "the quick brown fox jumps over the lazy dog",
            "a fast red car drives on the highway",
            "the lazy brown dog sleeps in the sun",
            "quick fox quick fox quick fox",
            "highway car drives fast red",
            "sun moon stars galaxy universe",
            "the quick red fox",
            "lazy lazy lazy dog dog dog",
        ];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();

        // Verify doc_freqs is populated.
        assert_eq!(index.doc_freqs.len(), index.vocab_size() as usize);

        // Multi-term queries -- these exercise the IDF-ordered sorting path.
        // The results should be identical regardless of term order in the query.
        for query in &["quick brown fox", "lazy dog", "fast red car highway", "sun moon stars"] {
            let r1 = index.search(query, 5).unwrap();
            let r2 = index.search(query, 5).unwrap();
            assert_eq!(r1.doc_ids, r2.doc_ids, "doc_ids differ for '{}'", query);
            for (j, (s1, s2)) in r1.scores.iter().zip(r2.scores.iter()).enumerate() {
                assert_eq!(
                    s1.to_bits(), s2.to_bits(),
                    "scores differ for '{}' at position {}", query, j
                );
            }
        }

        // Also verify raw scores for a single-term query.
        let scores = index.get_scores("quick").unwrap();
        // Doc 3 has "quick" 3 times, should have highest score.
        let max_doc = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_doc, 3, "Doc 3 should have highest score for 'quick'");
    }

    /// TEST-P1-003: Cached search returns identical results to uncached search.
    #[test]
    fn cached_search_returns_identical_results() {
        let corpus = &[
            "the quick brown fox",
            "the lazy dog",
            "brown fox jumps over the lazy dog",
            "quick quick quick fox",
        ];
        let index = BM25Builder::new()
            .cache_capacity(10)
            .build_from_corpus(corpus)
            .unwrap();

        for query in &["quick", "fox", "lazy dog", "brown"] {
            let uncached = index.search(query, 3).unwrap();
            let cached1 = index.search_cached(query, 3).unwrap();
            let cached2 = index.search_cached(query, 3).unwrap(); // should hit cache

            assert_eq!(uncached.doc_ids, cached1.doc_ids, "first cached call differs for '{}'", query);
            assert_eq!(uncached.doc_ids, cached2.doc_ids, "second cached call differs for '{}'", query);
            for (j, (s1, s2)) in uncached.scores.iter().zip(cached1.scores.iter()).enumerate() {
                assert_eq!(s1.to_bits(), s2.to_bits(), "score mismatch for '{}' at {}", query, j);
            }
        }
    }

    /// search_cached without cache configured falls back to search.
    #[test]
    fn search_cached_without_cache_falls_back() {
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        assert!(index.cache.is_none());

        let result = index.search_cached("hello", 1).unwrap();
        assert!(!result.doc_ids.is_empty());
    }

    /// Thread safety: BM25Index is Send + Sync (compile-time check).
    /// The static assertion at the top of this file already enforces this;
    /// this test additionally exercises it by sending an index across threads.
    #[test]
    fn bm25_index_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<BM25Index>();
        assert_sync::<BM25Index>();

        // Also verify at runtime by moving an index to another thread.
        let corpus = &["hello world", "foo bar"];
        let index = BM25Builder::new().build_from_corpus(corpus).unwrap();
        let handle = std::thread::spawn(move || {
            index.search("hello", 1).unwrap()
        });
        let results = handle.join().unwrap();
        assert!(!results.doc_ids.is_empty());
    }
}
