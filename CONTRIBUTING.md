# Contributing to BM25 Turbo

Thanks for your interest in contributing! This guide covers everything you need to get the project running locally.

## Prerequisites

- **Rust 1.85+** (edition 2024) — install via [rustup](https://rustup.rs)
- **Git**

Optional (for specific targets):
- **wasm-pack** + `wasm32-unknown-unknown` target — for WASM builds
- **Python 3.9-3.13** + **maturin** — for Python bindings
- **protoc** — only if modifying `.proto` files (not needed for normal development)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI-.git
cd BM25-Turbo-Rust-Python-WASM-CLI-

# Build the core library
cargo build -p bm25_turbo

# Build the CLI
cargo build --release -p bm25-turbo-cli

# Run the CLI
./target/release/bm25-turbo --help
```

## Running Tests

```bash
# Run all tests (excludes WASM and Python — they need extra tooling)
cargo test --workspace --exclude bm25-turbo-wasm --exclude bm25-turbo-python

# Run only core library tests (258 tests)
cargo test -p bm25_turbo

# Run only CLI integration tests (18 tests)
cargo test -p bm25-turbo-cli

# Run benchmark tests (16 tests)
cargo test -p bm25-turbo-bench

# Run clippy (must pass with zero warnings)
cargo clippy -p bm25_turbo -p bm25-turbo-cli --all-targets -- -D warnings
```

## Project Structure

```
bm25_turbo/           Core library (the engine)
bm25-turbo-cli/       CLI binary + HTTP server
bm25-turbo-python/    Python bindings (PyO3/maturin)
bm25-turbo-wasm/      WASM/npm package (wasm-bindgen)
benchmarks/           BEIR benchmark suite + Criterion benches
examples/             Runnable example programs
assets/               SVG charts for README
```

## Building Each Target

### Core Library (Rust)

```bash
cargo build -p bm25_turbo                    # Default features (parallel, persistence, simd)
cargo build -p bm25_turbo --features full    # All features enabled
cargo doc -p bm25_turbo --no-deps --open     # Generate and open API docs
```

### CLI + HTTP Server

```bash
cargo build --release -p bm25-turbo-cli

# Quick smoke test
echo '{"id":"1","text":"hello world"}' > /tmp/test.jsonl
./target/release/bm25-turbo index --input /tmp/test.jsonl --output /tmp/test.bin --field text
./target/release/bm25-turbo search --index /tmp/test.bin --query "hello" -k 5
./target/release/bm25-turbo serve --index /tmp/test.bin --port 8080
```

### WASM Package

```bash
# Install prerequisites (one-time)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Build
cd bm25-turbo-wasm
wasm-pack build --release --target web

# Output is in pkg/
ls pkg/
```

### Python Bindings

```bash
# Install prerequisites (one-time)
pip install maturin

# Build and install in development mode
cd bm25-turbo-python
maturin develop --release

# Test it
python -c "
from bm25_turbo_python import BM25
engine = BM25()
engine.index(['hello world', 'foo bar baz'])
results = engine.search('hello', k=1)
print(results)
"
```

**Note:** Python bindings require Python 3.9-3.13. Python 3.14+ is not yet supported by PyO3.

### Examples

```bash
# Each example is a standalone Cargo project
cargo run --manifest-path examples/basic_search/Cargo.toml
cargo run --manifest-path examples/persistence/Cargo.toml
cargo run --manifest-path examples/http_server/Cargo.toml
cargo run --manifest-path examples/streaming_index/Cargo.toml
```

## Running Benchmarks

### Quick benchmark (SciFact, 5K docs)

```bash
cargo run -p bm25-turbo-bench --release --bin beir_bench -- --datasets scifact
```

### Full benchmark (requires dataset download)

Download BEIR datasets from `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/` and extract to `beir_cache/`:

```bash
# Download MS MARCO (~1GB compressed, 8.8M docs)
mkdir -p beir_cache
# Extract msmarco.zip into beir_cache/msmarco/

# Run with query limit (recommended for development)
cargo run -p bm25-turbo-bench --release --bin beir_bench -- \
  --datasets msmarco --cache-dir ./beir_cache --max-queries 1000
```

### Criterion micro-benchmarks

```bash
cargo bench -p bm25-turbo-bench
# HTML report: target/criterion/report/index.html
```

## Feature Flags

The core library uses feature flags to keep the default build lightweight:

| Feature | Default | What it adds |
|---------|---------|-------------|
| `parallel` | yes | Rayon-based parallel indexing |
| `persistence` | yes | Save/load indexes to disk, memory-mapped I/O |
| `simd` | yes | SIMD-accelerated scoring and top-k selection |
| `server` | no | Axum HTTP server |
| `mcp` | no | MCP (Model Context Protocol) server |
| `huggingface` | no | HuggingFace Hub push/pull |
| `distributed` | no | gRPC distributed search (experimental) |
| `ann` | no | Approximate nearest neighbor (BMW pruning) |
| `wasm` | no | WebAssembly bindings |
| `full` | no | All features enabled |

## Code Style

- Run `cargo fmt` before committing (config in `rustfmt.toml`)
- All code must pass `cargo clippy -- -D warnings`
- Tests must pass on Linux, macOS, and Windows (CI checks all three)

## CI

GitHub Actions runs on every push and PR:
- Clippy (Linux)
- Tests (Linux, macOS, Windows)
- Benchmark compile check
- Python bindings (3.10, 3.11, 3.12, 3.13 x Linux, macOS, Windows)
- Release build

## Submitting Changes

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests (`cargo test --workspace --exclude bm25-turbo-wasm --exclude bm25-turbo-python`)
5. Run clippy (`cargo clippy -p bm25_turbo -p bm25-turbo-cli --all-targets -- -D warnings`)
6. Sign off every commit (see DCO below)
7. Push and open a pull request
8. Sign the CLA when prompted by the bot

### DCO (Developer Certificate of Origin)

All commits must include a `Signed-off-by` line certifying you wrote the code and have the right to submit it. Add it automatically with the `-s` flag:

```bash
git commit -s -m "Add new feature"
```

This produces:

```
Add new feature

Signed-off-by: Your Name <your@email.com>
```

If you forget, amend the commit:

```bash
git commit --amend -s
```

The DCO check will block your PR until all commits are signed off.

### CLA (Contributor License Agreement)

On your first PR, the CLA bot will ask you to sign by commenting:

> I have read the CLA Document and I hereby sign the CLA

This is a one-time signature that grants the maintainer the right to include your contribution under both the open source (AGPL-3.0) and commercial licenses. The full CLA text is in [CLA.md](CLA.md).

Both DCO and CLA are required for all pull requests.

## License

BM25 Turbo is dual-licensed under AGPL-3.0 (open source) and a commercial license. By contributing, you agree to the terms in [CLA.md](CLA.md), which grants the maintainer the right to distribute your contributions under both licenses.
