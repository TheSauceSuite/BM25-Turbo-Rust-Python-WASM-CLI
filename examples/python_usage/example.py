"""BM25 Turbo Python usage example.

Prerequisites:
    cd bm25-turbo-python
    maturin develop --release

Run:
    python examples/python_usage/example.py
"""

from bm25_turbo_python import BM25

# --- Build an index -----------------------------------------------------------

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces the fox",
    "the lazy cat sleeps all day long",
    "brown bears eat fish in the river",
    "the fox and the dog are friends",
]

engine = BM25(method="lucene", k1=1.5, b=0.75)
engine.index(corpus)
stats = engine.stats()
print(f"Indexed {stats['num_docs']} documents, vocab size: {stats['vocab_size']}")

# --- Search -------------------------------------------------------------------

query = "quick brown fox"
doc_ids, scores = engine.search(query, k=3)
print(f"\nQuery: \"{query}\"")
for doc_id, score in zip(doc_ids, scores):
    print(f"  doc {doc_id}: {score:.4f}  \"{corpus[doc_id]}\"")

# --- NumPy interop ------------------------------------------------------------

try:
    import numpy as np

    np_doc_ids, np_scores = engine.search_numpy(query, k=3)
    print(f"\nNumPy results (dtype: {np_doc_ids.dtype}, {np_scores.dtype}):")
    for doc_id, score in zip(np_doc_ids, np_scores):
        print(f"  doc {doc_id}: {score:.4f}")
except ImportError:
    print("\nNumPy not installed -- skipping numpy interop demo.")

# --- Persistence --------------------------------------------------------------

engine.save("example_index.bm25")
print("\nIndex saved to example_index.bm25")

loaded = BM25.load("example_index.bm25")
doc_ids2, scores2 = loaded.search(query, k=3)
print("Loaded index and searched -- results match:", doc_ids == doc_ids2)

# Clean up.
import os
os.remove("example_index.bm25")
