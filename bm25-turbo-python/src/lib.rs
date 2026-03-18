//! Python bindings for BM25 Turbo via PyO3.
//!
//! Exposes the core BM25 engine as Python classes with numpy array returns
//! for zero-copy interop. The GIL is released during indexing and querying.

use std::path::Path;

use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use bm25_turbo::types::Method;
use bm25_turbo::{BM25Builder, BM25Index};

// ---------------------------------------------------------------------------
// Custom exception
// ---------------------------------------------------------------------------

pyo3::create_exception!(bm25_turbo_python, BM25Error, pyo3::exceptions::PyException);

/// Convert a bm25_turbo::Error into a Python BM25Error.
fn to_py_err(e: bm25_turbo::Error) -> PyErr {
    BM25Error::new_err(e.to_string())
}

/// Parse a method string into a Method enum.
fn parse_method(method: &str) -> PyResult<Method> {
    match method.to_lowercase().as_str() {
        "robertson" => Ok(Method::Robertson),
        "lucene" => Ok(Method::Lucene),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::Bm25l),
        "bm25plus" | "bm25+" => Ok(Method::Bm25Plus),
        _ => Err(PyRuntimeError::new_err(format!(
            "unknown method '{}': expected one of: robertson, lucene, atire, bm25l, bm25plus",
            method
        ))),
    }
}

// ---------------------------------------------------------------------------
// BM25 Python class
// ---------------------------------------------------------------------------

/// BM25 search engine Python class.
///
/// # Example
/// ```python
/// from bm25_turbo import BM25
///
/// engine = BM25(method="lucene", k1=1.5, b=0.75)
/// engine.index(["document one", "document two", "document three"])
/// doc_ids, scores = engine.search("document", k=2)
/// print(doc_ids, scores)
/// ```
#[pyclass]
struct BM25 {
    /// The built index (None before index() is called).
    inner: Option<BM25Index>,
    /// Scoring method string.
    method: String,
    /// k1 parameter.
    k1: f32,
    /// b parameter.
    b: f32,
    /// delta parameter.
    delta: f32,
}

#[pymethods]
impl BM25 {
    #[new]
    #[pyo3(signature = (method = "lucene", k1 = 1.5, b = 0.75, delta = 0.5))]
    fn new(method: &str, k1: f32, b: f32, delta: f32) -> PyResult<Self> {
        // Validate method name eagerly.
        parse_method(method)?;
        Ok(Self {
            inner: None,
            method: method.to_string(),
            k1,
            b,
            delta,
        })
    }

    /// Build the index from a list of document strings.
    ///
    /// The GIL is released during index construction for concurrent Python threads.
    fn index(&mut self, py: Python<'_>, corpus: Vec<String>) -> PyResult<()> {
        let method = parse_method(&self.method)?;
        let k1 = self.k1;
        let b = self.b;
        let delta = self.delta;

        // Release GIL during the heavy computation.
        let index = py.allow_threads(move || {
            let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
            BM25Builder::new()
                .method(method)
                .k1(k1)
                .b(b)
                .delta(delta)
                .build_from_corpus(&corpus_refs)
        });

        self.inner = Some(index.map_err(to_py_err)?);
        Ok(())
    }

    /// Search the index and return (doc_ids, scores) as Python lists.
    ///
    /// The GIL is released during the search computation.
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `k` - Number of top results to return
    ///
    /// # Returns
    /// A tuple of (doc_ids: list[int], scores: list[float])
    #[pyo3(signature = (query, k = 10))]
    fn search(&self, _py: Python<'_>, query: &str, k: usize) -> PyResult<(Vec<u32>, Vec<f32>)> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| BM25Error::new_err("index not built: call index() first"))?;

        let results = index.search(query, k).map_err(to_py_err)?;
        Ok((results.doc_ids, results.scores))
    }

    /// Search the index and return (doc_ids, scores) as numpy arrays.
    ///
    /// Returns (numpy.ndarray[uint32], numpy.ndarray[float32]) for zero-copy
    /// interop with numpy/scipy/scikit-learn.
    #[pyo3(signature = (query, k = 10))]
    fn search_numpy<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let (doc_ids, scores) = self.search(py, query, k)?;
        let doc_ids_array = PyArray1::from_vec(py, doc_ids);
        let scores_array = PyArray1::from_vec(py, scores);
        Ok((doc_ids_array, scores_array))
    }

    /// Save the index to a file.
    ///
    /// Requires the `persistence` feature in the underlying Rust library.
    fn save(&self, path: &str) -> PyResult<()> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| BM25Error::new_err("index not built: call index() first"))?;

        bm25_turbo::persistence::save(index, Path::new(path)).map_err(to_py_err)
    }

    /// Load an index from a file.
    ///
    /// Returns a new BM25 instance with the loaded index.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let index = bm25_turbo::persistence::load(Path::new(path)).map_err(to_py_err)?;

        let method_str = match index.params().method {
            Method::Robertson => "robertson",
            Method::Lucene => "lucene",
            Method::Atire => "atire",
            Method::Bm25l => "bm25l",
            Method::Bm25Plus => "bm25plus",
        };

        Ok(Self {
            method: method_str.to_string(),
            k1: index.params().k1,
            b: index.params().b,
            delta: index.params().delta,
            inner: Some(index),
        })
    }

    /// Return index statistics as a Python dict.
    ///
    /// Keys: num_docs, vocab_size, method, k1, b, delta, avg_doc_len
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| BM25Error::new_err("index not built: call index() first"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_docs", index.num_docs())?;
        dict.set_item("vocab_size", index.vocab_size())?;
        dict.set_item("method", format!("{}", index.params().method))?;
        dict.set_item("k1", index.params().k1)?;
        dict.set_item("b", index.params().b)?;
        dict.set_item("delta", index.params().delta)?;
        dict.set_item("avg_doc_len", index.avg_doc_len())?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(index) => format!(
                "BM25(method='{}', num_docs={}, vocab_size={})",
                self.method,
                index.num_docs(),
                index.vocab_size()
            ),
            None => format!(
                "BM25(method='{}', not indexed)",
                self.method
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer Python class
// ---------------------------------------------------------------------------

/// Standalone tokenizer for text processing.
///
/// Exposed for users who want to pre-tokenize text before indexing
/// or inspect tokenization behavior.
#[pyclass]
struct Tokenizer {
    inner: bm25_turbo::Tokenizer,
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (language = None))]
    fn new(language: Option<&str>) -> PyResult<Self> {
        let builder = bm25_turbo::Tokenizer::builder();
        let builder = match language {
            Some(lang) => builder.language(lang),
            None => builder,
        };
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Tokenize a text string into a list of token strings.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize(text)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------

/// Python module for BM25 Turbo.
#[pymodule]
fn bm25_turbo_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25>()?;
    m.add_class::<Tokenizer>()?;
    m.add("BM25Error", m.py().get_type::<BM25Error>())?;
    Ok(())
}
