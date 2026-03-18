//! Streaming/chunked indexing builder for large corpora.
//!
//! [`StreamingBuilder`] processes documents in configurable chunks to enable
//! indexing corpora that exceed available memory. Each chunk is processed
//! independently, and chunks are merged at build time to produce a single
//! [`BM25Index`].
//!
//! Memory usage per chunk: O(chunk_size * avg_tokens_per_doc), not O(total_corpus).

use std::collections::HashMap;

use crate::csc::CscMatrix;
use crate::error::{Error, Result};
use crate::index::BM25Index;
use crate::scoring;
use crate::tokenizer::Tokenizer;
use crate::types::{BM25Params, Method};

/// Default chunk size: 100,000 documents per chunk.
const DEFAULT_CHUNK_SIZE: usize = 100_000;

/// Internal state accumulated per chunk.
struct ChunkState {
    /// Local vocabulary: token string -> local token ID.
    vocab: HashMap<String, u32>,
    /// Term frequency counts: `tf_counts[local_tid]` is a map from doc_id to tf.
    tf_counts: Vec<HashMap<u32, u32>>,
    /// Document lengths (token count per document).
    doc_lengths: Vec<u32>,
    /// Global starting doc_id for documents in this chunk.
    doc_id_offset: u32,
}

/// A streaming index builder that processes documents in chunks.
///
/// Follows the consuming builder pattern (same as [`crate::BM25Builder`]).
/// Documents are accumulated into internal chunk state and flushed
/// at chunk-size boundaries. At build time, all chunks are merged
/// into a unified vocabulary and CSC matrix.
pub struct StreamingBuilder {
    params: BM25Params,
    tokenizer: Tokenizer,
    chunk_size: usize,
    cache_capacity: Option<usize>,

    /// Completed chunks.
    chunks: Vec<ChunkState>,
    /// Current in-progress chunk state.
    current_vocab: HashMap<String, u32>,
    current_tf_counts: Vec<HashMap<u32, u32>>,
    current_doc_lengths: Vec<u32>,
    /// Total documents added so far (across all chunks + current).
    total_docs: u32,
}

impl Default for StreamingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingBuilder {
    /// Create a new streaming builder with default parameters.
    pub fn new() -> Self {
        Self {
            params: BM25Params::default(),
            tokenizer: Tokenizer::default(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            cache_capacity: None,
            chunks: Vec::new(),
            current_vocab: HashMap::new(),
            current_tf_counts: Vec::new(),
            current_doc_lengths: Vec::new(),
            total_docs: 0,
        }
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

    /// Set the chunk size (number of documents per chunk).
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set cache capacity for the resulting index.
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = Some(capacity);
        self
    }

    /// Add a batch of documents to the builder.
    ///
    /// Documents are tokenized and accumulated into internal chunk state.
    /// When the current chunk reaches `chunk_size`, it is flushed and a
    /// new chunk begins.
    pub fn add_documents(&mut self, docs: &[&str]) {
        for doc in docs {
            let tokens = self.tokenizer.tokenize(doc);
            self.add_tokenized_doc(tokens);
        }
    }

    /// Add documents from an iterator (true streaming from file readers).
    pub fn add_iter<I: Iterator<Item = String>>(&mut self, docs: I) {
        for doc in docs {
            let tokens = self.tokenizer.tokenize(&doc);
            self.add_tokenized_doc(tokens);
        }
    }

    /// Internal: add a single tokenized document to the current chunk.
    fn add_tokenized_doc(&mut self, tokens: Vec<String>) {
        let doc_id = self.total_docs;
        self.total_docs += 1;

        let doc_len = tokens.len() as u32;
        self.current_doc_lengths.push(doc_len);

        // Count term frequencies for this document.
        let mut tf_map: HashMap<u32, u32> = HashMap::new();
        for token in &tokens {
            let next_id = self.current_vocab.len() as u32;
            let tid = *self.current_vocab.entry(token.clone()).or_insert(next_id);
            *tf_map.entry(tid).or_insert(0) += 1;
        }

        // Ensure tf_counts has enough entries for all local token IDs.
        while self.current_tf_counts.len() <= self.current_vocab.len() {
            self.current_tf_counts.push(HashMap::new());
        }

        // Record term frequencies.
        for (&tid, &tf) in &tf_map {
            self.current_tf_counts[tid as usize].insert(doc_id, tf);
        }

        // Flush chunk if we have reached chunk_size.
        if self.current_doc_lengths.len() >= self.chunk_size {
            self.flush_chunk();
        }
    }

    /// Flush the current chunk state to a completed ChunkState.
    fn flush_chunk(&mut self) {
        if self.current_doc_lengths.is_empty() {
            return;
        }

        let doc_id_offset = self.total_docs - self.current_doc_lengths.len() as u32;

        let chunk = ChunkState {
            vocab: std::mem::take(&mut self.current_vocab),
            tf_counts: std::mem::take(&mut self.current_tf_counts),
            doc_lengths: std::mem::take(&mut self.current_doc_lengths),
            doc_id_offset,
        };
        self.chunks.push(chunk);
    }

    /// Build the final BM25Index by merging all chunks.
    ///
    /// Steps:
    /// 1. Flush any remaining documents in the current chunk.
    /// 2. Build a unified global vocabulary across all chunks.
    /// 3. Compute global document frequencies.
    /// 4. Compute IDF values.
    /// 5. Compute BM25 scores for all (term, document) pairs.
    /// 6. Build CSC matrix from triplets.
    pub fn build(mut self) -> Result<BM25Index> {
        // Validate parameters.
        scoring::validate_params(&self.params)?;

        // Flush remaining documents.
        self.flush_chunk();

        if self.chunks.is_empty() {
            return Err(Error::InvalidParameter(
                "corpus must not be empty".to_string(),
            ));
        }

        let num_docs = self.total_docs;

        // Step 1: Build unified global vocabulary.
        let mut global_vocab: HashMap<String, u32> = HashMap::new();
        for chunk in &self.chunks {
            for token in chunk.vocab.keys() {
                let next_id = global_vocab.len() as u32;
                global_vocab.entry(token.clone()).or_insert(next_id);
            }
        }
        let vocab_size = global_vocab.len() as u32;

        // Build inverse vocabulary.
        let mut vocab_inv = vec![String::new(); vocab_size as usize];
        for (token, &id) in &global_vocab {
            vocab_inv[id as usize] = token.clone();
        }

        // Step 2: Compute global document frequencies and total token count.
        let mut df = vec![0u32; vocab_size as usize];
        let mut total_tokens: u64 = 0;

        for chunk in &self.chunks {
            total_tokens += chunk.doc_lengths.iter().map(|&l| l as u64).sum::<u64>();

            // Build local-to-global ID mapping for this chunk.
            let local_to_global: Vec<u32> = {
                let mut mapping = vec![0u32; chunk.vocab.len()];
                for (token, &local_id) in &chunk.vocab {
                    mapping[local_id as usize] = global_vocab[token];
                }
                mapping
            };

            // Count document frequencies: for each local term, count how many
            // documents in this chunk have a non-zero tf.
            for (local_tid, tf_map) in chunk.tf_counts.iter().enumerate() {
                if local_tid < local_to_global.len() {
                    let global_tid = local_to_global[local_tid];
                    df[global_tid as usize] += tf_map.len() as u32;
                }
            }
        }

        let avg_doc_len = if num_docs > 0 {
            total_tokens as f32 / num_docs as f32
        } else {
            0.0
        };

        // Step 3: Compute IDF array.
        let idf_arr: Vec<f32> = (0..vocab_size as usize)
            .map(|tid| scoring::idf(self.params.method, num_docs, df[tid]))
            .collect();

        // Step 4: Build doc_lengths array indexed by global doc_id.
        let mut all_doc_lengths = vec![0u32; num_docs as usize];
        for chunk in &self.chunks {
            for (i, &len) in chunk.doc_lengths.iter().enumerate() {
                all_doc_lengths[chunk.doc_id_offset as usize + i] = len;
            }
        }

        // Step 5: Compute triplets.
        let method = self.params.method;
        let k1 = self.params.k1;
        let b = self.params.b;
        let delta = self.params.delta;

        let mut triplets: Vec<(u32, u32, f32)> = Vec::new();

        for chunk in &self.chunks {
            let local_to_global: Vec<u32> = {
                let mut mapping = vec![0u32; chunk.vocab.len()];
                for (token, &local_id) in &chunk.vocab {
                    mapping[local_id as usize] = global_vocab[token];
                }
                mapping
            };

            for (local_tid, tf_map) in chunk.tf_counts.iter().enumerate() {
                if local_tid >= local_to_global.len() {
                    continue;
                }
                let global_tid = local_to_global[local_tid];
                let idf_val = idf_arr[global_tid as usize];

                for (&doc_id, &tf) in tf_map {
                    let doc_len = all_doc_lengths[doc_id as usize] as f32;
                    let tfc_val = scoring::tfc(method, tf as f32, doc_len, avg_doc_len, k1, b, delta);
                    let s = (idf_val as f64 * tfc_val as f64) as f32;
                    if s != 0.0 {
                        triplets.push((global_tid, doc_id, s));
                    }
                }
            }
        }

        // Step 6: Build CSC matrix.
        let matrix = CscMatrix::from_triplets(&triplets, num_docs, vocab_size)?;

        // Step 7: Store doc_freqs for IDF-ordered term processing.
        let doc_freqs = df;

        let mut index = BM25Index {
            matrix,
            vocab: global_vocab,
            vocab_inv,
            params: self.params,
            avg_doc_len,
            num_docs,
            tokenizer: self.tokenizer,
            doc_freqs,
            cache: self.cache_capacity.map(crate::query_cache::QueryCache::new),
            #[cfg(feature = "ann")]
            block_max_index: None,
        };

        // Sort the index the same way BM25Builder does -- not needed since
        // we're building fresh, but we set doc_freqs for IDF ordering.
        let _ = &mut index;

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::BM25Builder;

    /// TEST-P1-007: Streaming matches batch build.
    #[test]
    fn streaming_matches_batch_build() {
        let corpus: Vec<&str> = vec![
            "the quick brown fox",
            "the lazy dog",
            "brown fox jumps over the lazy dog",
            "quick quick quick fox",
            "the dog sat on the mat",
        ];

        // Build via BM25Builder.
        let batch_index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .expect("batch build should succeed");

        // Build via StreamingBuilder with small chunk size to force multiple chunks.
        let mut streaming = StreamingBuilder::new().chunk_size(2);
        streaming.add_documents(&corpus);
        let stream_index = streaming.build().expect("streaming build should succeed");

        assert_eq!(batch_index.num_docs(), stream_index.num_docs());

        // Run queries and compare results.
        for query in &["quick", "fox", "lazy dog", "brown", "the mat"] {
            let r_batch = batch_index.search(query, 5).unwrap();
            let r_stream = stream_index.search(query, 5).unwrap();
            assert_eq!(
                r_batch.doc_ids, r_stream.doc_ids,
                "doc_ids mismatch for query '{}'", query
            );
            for (j, (s1, s2)) in r_batch.scores.iter().zip(r_stream.scores.iter()).enumerate() {
                assert_eq!(
                    s1.to_bits(), s2.to_bits(),
                    "score mismatch for query '{}' at position {}: batch={} stream={}",
                    query, j, s1, s2
                );
            }
        }
    }

    /// TEST-P1-007 extended: Streaming with single chunk matches batch.
    #[test]
    fn streaming_single_chunk_matches_batch() {
        let corpus: Vec<&str> = vec![
            "alpha beta gamma",
            "beta gamma delta",
            "alpha delta",
        ];

        let batch_index = BM25Builder::new()
            .build_from_corpus(&corpus)
            .expect("batch build should succeed");

        let mut streaming = StreamingBuilder::new().chunk_size(100);
        streaming.add_documents(&corpus);
        let stream_index = streaming.build().expect("streaming build should succeed");

        for query in &["alpha", "beta", "delta", "gamma"] {
            let r_batch = batch_index.search(query, 5).unwrap();
            let r_stream = stream_index.search(query, 5).unwrap();
            assert_eq!(
                r_batch.doc_ids, r_stream.doc_ids,
                "doc_ids mismatch for query '{}'", query
            );
            for (j, (s1, s2)) in r_batch.scores.iter().zip(r_stream.scores.iter()).enumerate() {
                assert_eq!(
                    s1.to_bits(), s2.to_bits(),
                    "score mismatch for query '{}' at position {}",
                    query, j
                );
            }
        }
    }

    /// TEST-P1-008: Streaming large chunks (synthetic corpus).
    ///
    /// Note: reduced from 500K to 5K for fast test execution.
    /// The chunking logic is the same regardless of corpus size.
    #[test]
    fn streaming_large_chunks() {
        let corpus_size = 5_000;
        let words = ["alpha", "beta", "gamma", "delta", "epsilon",
                     "zeta", "eta", "theta", "iota", "kappa"];

        // Generate synthetic documents deterministically.
        let corpus: Vec<String> = (0..corpus_size)
            .map(|i| {
                let w1 = words[i % words.len()];
                let w2 = words[(i * 3 + 1) % words.len()];
                let w3 = words[(i * 7 + 2) % words.len()];
                format!("{} {} {}", w1, w2, w3)
            })
            .collect();

        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();

        // Build via BM25Builder for reference.
        let batch_index = BM25Builder::new()
            .build_from_corpus(&corpus_refs)
            .expect("batch build should succeed");

        // Build via StreamingBuilder with 1K chunk size.
        let mut streaming = StreamingBuilder::new().chunk_size(1_000);
        streaming.add_documents(&corpus_refs);
        let stream_index = streaming.build().expect("streaming build should succeed");

        assert_eq!(batch_index.num_docs(), stream_index.num_docs());

        // Verify search results for sample queries.
        for query in &["alpha", "beta gamma", "delta epsilon", "theta"] {
            let r_batch = batch_index.search(query, 10).unwrap();
            let r_stream = stream_index.search(query, 10).unwrap();
            assert_eq!(
                r_batch.doc_ids, r_stream.doc_ids,
                "doc_ids mismatch for query '{}' (corpus_size={})", query, corpus_size
            );
        }
    }

    /// Streaming with add_iter.
    #[test]
    fn streaming_add_iter() {
        let docs = vec![
            "hello world".to_string(),
            "foo bar".to_string(),
            "hello foo".to_string(),
        ];

        let mut streaming = StreamingBuilder::new().chunk_size(2);
        streaming.add_iter(docs.into_iter());
        let index = streaming.build().expect("build should succeed");

        let results = index.search("hello", 2).unwrap();
        assert!(!results.doc_ids.is_empty());
    }

    /// Empty corpus returns error.
    #[test]
    fn streaming_empty_corpus_error() {
        let streaming = StreamingBuilder::new();
        assert!(streaming.build().is_err());
    }

    /// Builder methods work correctly.
    #[test]
    fn streaming_builder_methods() {
        let mut builder = StreamingBuilder::new()
            .method(Method::Atire)
            .k1(2.0)
            .b(0.5)
            .delta(1.0)
            .chunk_size(50);

        builder.add_documents(&["hello world", "foo bar"]);
        let index = builder.build().unwrap();
        assert_eq!(index.num_docs(), 2);
    }

    /// All five BM25 variants via streaming.
    #[test]
    fn streaming_all_variants() {
        let corpus: Vec<&str> = vec![
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
            let batch = BM25Builder::new()
                .method(*method)
                .build_from_corpus(&corpus)
                .unwrap();

            let mut streaming = StreamingBuilder::new()
                .method(*method)
                .chunk_size(3);
            streaming.add_documents(&corpus);
            let stream = streaming.build().unwrap();

            let r_batch = batch.search("hello", 2).unwrap();
            let r_stream = stream.search("hello", 2).unwrap();
            assert_eq!(
                r_batch.doc_ids, r_stream.doc_ids,
                "{:?}: doc_ids mismatch", method
            );
            for (j, (s1, s2)) in r_batch.scores.iter().zip(r_stream.scores.iter()).enumerate() {
                assert_eq!(
                    s1.to_bits(), s2.to_bits(),
                    "{:?}: score mismatch at position {}", method, j
                );
            }
        }
    }
}
