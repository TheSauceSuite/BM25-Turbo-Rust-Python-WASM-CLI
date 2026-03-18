//! Thread-safe LRU query result cache.
//!
//! Provides an opt-in cache for BM25 search results, backed by a
//! [`LinkedHashMap`] wrapped in a [`parking_lot::Mutex`] for concurrent access.
//! Cache key format: `"{query}\0{k}"` to distinguish different k values.

use linked_hash_map::LinkedHashMap;
use parking_lot::Mutex;

use crate::types::Results;

/// Thread-safe LRU cache for query results.
///
/// The cache stores `Results` keyed by a composite string of query text and k.
/// When the cache exceeds its capacity, the least-recently-used entry is evicted.
///
/// A capacity of 0 disables caching entirely -- all operations become no-ops.
pub struct QueryCache {
    inner: Mutex<LruInner>,
}

struct LruInner {
    map: LinkedHashMap<String, Results>,
    capacity: usize,
}

// Compile-time proof that QueryCache is Send + Sync.
const _: () = {
    const fn _assert<T: Send + Sync>() {}
    _assert::<QueryCache>();
};

impl QueryCache {
    /// Create a new cache with the given maximum capacity.
    ///
    /// A capacity of 0 disables caching -- `insert()` will be a no-op and
    /// `get()` will always return `None`.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruInner {
                map: LinkedHashMap::new(),
                capacity,
            }),
        }
    }

    /// Build the cache key from a query string and k value.
    ///
    /// Format: `"{query}\0{k}"` -- the null byte separator ensures no
    /// ambiguity between query text and the k value.
    pub fn cache_key(query: &str, k: usize) -> String {
        format!("{}\0{}", query, k)
    }

    /// Look up a cached result by key.
    ///
    /// If the key exists, it is promoted to most-recently-used and a clone
    /// of the result is returned. Returns `None` on cache miss or if
    /// capacity is 0.
    pub fn get(&self, key: &str) -> Option<Results> {
        let mut inner = self.inner.lock();
        if inner.capacity == 0 {
            return None;
        }
        inner.map.get_refresh(key).cloned()
    }

    /// Insert a result into the cache.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted
    /// first. If capacity is 0, this is a no-op.
    pub fn insert(&self, key: String, value: Results) {
        let mut inner = self.inner.lock();
        if inner.capacity == 0 {
            return;
        }
        if inner.map.contains_key(&key) {
            // Update existing entry and refresh its position.
            inner.map.insert(key, value);
            return;
        }
        if inner.map.len() >= inner.capacity {
            // Evict LRU (front of the linked list).
            inner.map.pop_front();
        }
        inner.map.insert(key, value);
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.map.clear();
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        let inner = self.inner.lock();
        inner.map.len()
    }

    /// Return true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TEST-P1-002: Cache hit returns same result.
    #[test]
    fn cache_hit_returns_same_result() {
        let cache = QueryCache::new(10);
        let results = Results {
            doc_ids: vec![3, 1, 0],
            scores: vec![2.5, 1.8, 0.3],
        };
        let key = QueryCache::cache_key("hello world", 5);
        cache.insert(key.clone(), results.clone());

        let cached = cache.get(&key).expect("cache should return a hit");
        assert_eq!(cached.doc_ids, results.doc_ids);
        for (a, b) in cached.scores.iter().zip(results.scores.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// TEST-P1-003: Cache evicts LRU entry at capacity.
    #[test]
    fn cache_evicts_lru_entry() {
        let cache = QueryCache::new(2);

        let r1 = Results { doc_ids: vec![0], scores: vec![1.0] };
        let r2 = Results { doc_ids: vec![1], scores: vec![2.0] };
        let r3 = Results { doc_ids: vec![2], scores: vec![3.0] };

        cache.insert("a".to_string(), r1);
        cache.insert("b".to_string(), r2);
        assert_eq!(cache.len(), 2);

        // Inserting a third entry should evict "a" (LRU).
        cache.insert("c".to_string(), r3);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("a").is_none(), "LRU entry 'a' should be evicted");
        assert!(cache.get("b").is_some(), "'b' should still be present");
        assert!(cache.get("c").is_some(), "'c' should be present");
    }

    /// TEST-P1-003 extended: Accessing an entry refreshes its position.
    #[test]
    fn cache_access_refreshes_lru_position() {
        let cache = QueryCache::new(2);

        let r1 = Results { doc_ids: vec![0], scores: vec![1.0] };
        let r2 = Results { doc_ids: vec![1], scores: vec![2.0] };
        let r3 = Results { doc_ids: vec![2], scores: vec![3.0] };

        cache.insert("a".to_string(), r1);
        cache.insert("b".to_string(), r2);

        // Access "a" to make it most-recently-used.
        let _ = cache.get("a");

        // Insert "c" -- should evict "b" (now LRU), not "a".
        cache.insert("c".to_string(), r3);
        assert!(cache.get("a").is_some(), "'a' should still be present after refresh");
        assert!(cache.get("b").is_none(), "'b' should be evicted");
        assert!(cache.get("c").is_some(), "'c' should be present");
    }

    /// TEST-P1-004: Cache with capacity 0 disables caching.
    #[test]
    fn cache_capacity_zero_disables_caching() {
        let cache = QueryCache::new(0);
        let results = Results { doc_ids: vec![0], scores: vec![1.0] };
        cache.insert("test".to_string(), results);
        assert_eq!(cache.len(), 0);
        assert!(cache.get("test").is_none());
    }

    /// TEST-P1-005: Cache thread safety under concurrent access.
    #[test]
    fn cache_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(QueryCache::new(100));
        let mut handles = Vec::new();

        for t in 0..4 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("thread{}-query{}", t, i);
                    let results = Results {
                        doc_ids: vec![i as u32],
                        scores: vec![i as f32],
                    };
                    cache.insert(key.clone(), results);
                    let _ = cache.get(&key);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread should not panic");
        }

        // Cache should have entries and be in a consistent state.
        let len = cache.len();
        assert!(len > 0, "cache should have entries after concurrent access");
        assert!(len <= 100, "cache should not exceed capacity");
    }

    /// Clear removes all entries.
    #[test]
    fn cache_clear() {
        let cache = QueryCache::new(10);
        for i in 0..5 {
            cache.insert(format!("key{}", i), Results {
                doc_ids: vec![i],
                scores: vec![i as f32],
            });
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    /// Cache key format uses null byte separator.
    #[test]
    fn cache_key_format() {
        let key = QueryCache::cache_key("hello", 10);
        assert_eq!(key, "hello\x0010");
        // Different k values produce different keys.
        let key2 = QueryCache::cache_key("hello", 5);
        assert_ne!(key, key2);
    }
}
