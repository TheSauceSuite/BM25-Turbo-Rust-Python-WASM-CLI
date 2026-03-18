# BM25 Turbo Examples

Real-world examples demonstrating BM25 Turbo's key capabilities. Each Rust example is a standalone Cargo project with its own `Cargo.toml` -- just pick one and run it.

## Examples

| Example | Description | Features Used | Run Command |
|---------|-------------|---------------|-------------|
| `basic_search` | Minimal search: build index, query, print top-3 results for all 5 BM25 variants | core | `cargo run --manifest-path examples/basic_search/Cargo.toml` |
| `persistence` | Save index to disk, load it back, verify round-trip correctness | persistence | `cargo run --manifest-path examples/persistence/Cargo.toml` |
| `http_server` | Axum REST API with `/search` and `/health` endpoints | server | `cargo run --manifest-path examples/http_server/Cargo.toml` |
| `python_usage` | Python bindings: index, search, NumPy interop, save/load | python (PyO3) | `python examples/python_usage/example.py` |
| `streaming_index` | Chunked indexing for large corpora via `StreamingBuilder` | streaming | `cargo run --manifest-path examples/streaming_index/Cargo.toml` |

## Getting Started

1. Clone the repository
2. Pick an example from the table above
3. Run the command -- dependencies are resolved automatically via path references to the `bm25_turbo` crate

For the Python example, install the bindings first:

```bash
cd bm25-turbo-python
pip install maturin
maturin develop --release
```
