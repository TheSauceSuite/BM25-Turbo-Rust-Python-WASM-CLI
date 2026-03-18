# Python Usage Example

Demonstrates BM25 Turbo's Python bindings via PyO3: indexing, searching, NumPy interop, and persistence.

## Prerequisites

- Python 3.8+
- Rust toolchain (rustup)
- maturin (`pip install maturin`)

## Install

```bash
cd bm25-turbo-python
maturin develop --release
```

## Run

```bash
python examples/python_usage/example.py
```

## What It Does

1. Creates a BM25 engine with Lucene scoring
2. Indexes 5 sample documents
3. Searches for "quick brown fox" and prints top-3 results
4. Demonstrates NumPy array returns (if numpy is installed)
5. Saves the index to disk, loads it back, and verifies results match
