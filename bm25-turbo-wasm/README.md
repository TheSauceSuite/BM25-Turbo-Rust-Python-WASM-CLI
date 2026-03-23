# bm25-turbo

The fastest BM25 scoring engine, compiled to WebAssembly.

BM25 Turbo is a Rust-native BM25 information retrieval engine that supports 5 scoring variants (Robertson, Lucene, ATIRE, BM25L, BM25+), 17-language tokenization with Snowball stemming, and compressed sparse column storage. This package brings the full engine to the browser and Node.js via WebAssembly.

## Installation

```bash
npm install bm25-turbo
```

## Quick Start

```javascript
import init, { WasmBM25 } from 'bm25-turbo';

// Initialize the WASM module (required once before use)
await init();

// Build an index from an array of documents
const index = new WasmBM25([
  "The quick brown fox jumps over the lazy dog",
  "A fast red car drives on the highway",
  "The brown dog sleeps in the sun",
  "Quick foxes are surprisingly lazy animals",
]);

// Search for the top 2 results
const results = index.search("quick brown fox", 2);
console.log(results);
// { doc_ids: [0, 3], scores: [1.82, 0.94] }
```

## Usage

### Creating an Index

The `WasmBM25` constructor accepts an array of document strings and optional parameters:

```typescript
const index = new WasmBM25(
  documents,   // string[] — array of document texts
  method?,     // string — scoring variant (default: "lucene")
  k1?,         // number — term frequency saturation (default: 1.5)
  b?,          // number — document length normalization (default: 0.75)
);
```

**Supported methods:** `"robertson"`, `"lucene"`, `"atire"`, `"bm25l"`, `"bm25plus"`

### Searching

```typescript
const results = index.search(query, k);
// Returns: { doc_ids: number[], scores: number[] }
```

- `query` — search query string
- `k` — maximum number of results to return (must be > 0)

### Serialization

Save an index to bytes for storage (e.g., IndexedDB, localStorage):

```typescript
// Serialize
const bytes = index.to_bytes(); // Uint8Array

// Deserialize
const restored = WasmBM25.loadBytes(bytes);
```

### Index Statistics

```typescript
console.log(index.num_docs());    // number of documents
console.log(index.vocab_size());  // number of unique terms
console.log(index.stats());      // JSON string with full stats
```

## API Reference

### `new WasmBM25(documents, method?, k1?, b?)`

Construct a BM25 index from a corpus of documents.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents` | `string[]` | (required) | Array of document texts to index |
| `method` | `string` | `"lucene"` | BM25 scoring variant |
| `k1` | `number` | `1.5` | Term frequency saturation parameter |
| `b` | `number` | `0.75` | Document length normalization parameter |

### `.search(query, k)`

Search the index and return the top-k results.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `string` | Search query text |
| `k` | `number` | Maximum number of results (must be > 0) |

**Returns:** `{ doc_ids: number[], scores: number[] }`

### `.num_docs()`

Returns the number of documents in the index.

### `.vocab_size()`

Returns the number of unique terms in the vocabulary.

### `.stats()`

Returns a JSON string with index statistics including `num_docs`, `vocab_size`, `method`, `k1`, `b`, `delta`, and `avg_doc_len`.

### `.to_bytes()`

Serializes the index to a `Uint8Array` for storage or transfer.

### `WasmBM25.loadBytes(data)`

Static method. Deserializes an index from bytes produced by `to_bytes()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `Uint8Array` | Serialized index bytes |

**Returns:** `WasmBM25`

## Browser Support

Requires a browser with WebAssembly support (all modern browsers). The package targets the `web` platform by default, producing ES module output suitable for `<script type="module">` or bundlers.

## License

MIT OR Apache-2.0
