//! REST API server module (feature-gated: `server`).
//!
//! Provides an Axum router with:
//! - `POST /search` — query the index
//! - `GET /health` — health check
//! - `GET /stats` — index statistics
//! - `POST /admin/reload` — atomically swap the loaded index

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::index::BM25Index;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// POST /search request body.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// The query string to search for.
    pub query: String,
    /// Number of top results to return (default: 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Use approximate (BMW) search. Defaults to server's approximate mode.
    pub approximate: Option<bool>,
}

/// POST /batch request body.
#[derive(Debug, Deserialize)]
pub struct BatchSearchRequest {
    /// List of query strings to search.
    pub queries: Vec<String>,
    /// Number of top results to return per query (default: 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// POST /batch response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchSearchResponse {
    /// Results for each query, in the same order as the input.
    pub results: Vec<SearchResponse>,
    /// Total latency across all queries in milliseconds.
    pub total_latency_ms: f64,
}

fn default_top_k() -> usize {
    10
}

/// A single search result.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID in the corpus.
    pub doc_id: u32,
    /// BM25 relevance score.
    pub score: f32,
}

/// POST /search response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The original query string.
    pub query: String,
    /// Matching documents sorted by descending score.
    pub results: Vec<SearchResult>,
    /// Number of results returned.
    pub count: usize,
    /// Query latency in milliseconds.
    pub latency_ms: f64,
}

/// GET /health response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Always "ok" when the server is running.
    pub status: String,
    /// Seconds since server started.
    pub uptime_seconds: f64,
    /// Whether an index is currently loaded.
    pub index_loaded: bool,
}

/// GET /stats response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct StatsResponse {
    /// Number of documents in the index.
    pub num_docs: u32,
    /// Number of unique terms.
    pub vocab_size: u32,
    /// BM25 scoring variant name.
    pub variant: String,
    /// Scoring parameters.
    pub params: ParamsInfo,
    /// Average document length.
    pub avg_doc_length: f32,
}

/// Scoring parameter subset for the stats endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParamsInfo {
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
}

/// Error response body.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail within an error response.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

/// Shared application state passed to all handlers.
pub struct AppState {
    /// The loaded BM25 index. `RwLock` allows concurrent reads with exclusive
    /// write access for the reload endpoint.
    pub index: RwLock<Arc<BM25Index>>,
    /// Server start time for uptime reporting.
    pub started_at: Instant,
    /// Whether to use approximate (BMW) search by default.
    pub approximate: bool,
}

// ---------------------------------------------------------------------------
// Router constructor
// ---------------------------------------------------------------------------

/// Default request body size limit (1 MB).
const DEFAULT_BODY_LIMIT: usize = 1024 * 1024;

/// Create the Axum router with all routes wired to the given shared state.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/search", post(handle_search))
        .route("/batch", post(handle_batch))
        .route("/health", get(handle_health))
        .route("/stats", get(handle_stats))
        .route("/admin/reload", post(handle_reload))
        .layer(axum::extract::DefaultBodyLimit::max(DEFAULT_BODY_LIMIT))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /search — execute a BM25 query.
async fn handle_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    // Validate top_k.
    if req.top_k == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "invalid_parameter".to_string(),
                    message: "top_k must be > 0".to_string(),
                },
            }),
        ));
    }

    // Grab a read lock and clone the Arc (cheap) so we can release the lock
    // before doing CPU-bound work.
    let index = {
        let guard = state.index.read().await;
        Arc::clone(&guard)
    };

    let query = req.query.clone();
    let top_k = req.top_k;
    let use_approximate = req.approximate.unwrap_or(state.approximate);

    // Run scoring on a blocking thread to avoid starving the async runtime.
    let result = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let search_result = if use_approximate {
            #[cfg(feature = "ann")]
            { index.search_approximate(&query, top_k) }
            #[cfg(not(feature = "ann"))]
            { Err(crate::error::Error::FeatureNotEnabled("approximate search requires the 'ann' feature".to_string())) }
        } else {
            index.search(&query, top_k)
        };
        let elapsed = start.elapsed();
        (search_result, elapsed, query)
    })
    .await;

    match result {
        Ok((Ok(results), elapsed, query)) => {
            let search_results: Vec<SearchResult> = results
                .doc_ids
                .iter()
                .zip(results.scores.iter())
                .map(|(&doc_id, &score)| SearchResult { doc_id, score })
                .collect();
            let count = search_results.len();
            Ok((
                StatusCode::OK,
                Json(SearchResponse {
                    query,
                    results: search_results,
                    count,
                    latency_ms: elapsed.as_secs_f64() * 1000.0,
                }),
            ))
        }
        Ok((Err(e), _, _)) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "search_error".to_string(),
                    message: e.to_string(),
                },
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "internal_error".to_string(),
                    message: format!("task join error: {}", e),
                },
            }),
        )),
    }
}

/// POST /batch — execute multiple BM25 queries in one request.
async fn handle_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    if req.top_k == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "invalid_parameter".to_string(),
                    message: "top_k must be > 0".to_string(),
                },
            }),
        ));
    }

    let index = {
        let guard = state.index.read().await;
        Arc::clone(&guard)
    };

    let queries = req.queries;
    let top_k = req.top_k;

    let result = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let query_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();
        let batch_result = index.search_batch(&query_refs, top_k);
        let elapsed = start.elapsed();
        (batch_result, elapsed, queries)
    })
    .await;

    match result {
        Ok((Ok(batch_results), elapsed, queries)) => {
            let responses: Vec<SearchResponse> = batch_results
                .into_iter()
                .zip(queries.into_iter())
                .map(|(results, query)| {
                    let search_results: Vec<SearchResult> = results
                        .doc_ids
                        .iter()
                        .zip(results.scores.iter())
                        .map(|(&doc_id, &score)| SearchResult { doc_id, score })
                        .collect();
                    let count = search_results.len();
                    SearchResponse {
                        query,
                        results: search_results,
                        count,
                        latency_ms: 0.0, // individual latencies not tracked in batch
                    }
                })
                .collect();
            Ok((
                StatusCode::OK,
                Json(BatchSearchResponse {
                    results: responses,
                    total_latency_ms: elapsed.as_secs_f64() * 1000.0,
                }),
            ))
        }
        Ok((Err(e), _, _)) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "search_error".to_string(),
                    message: e.to_string(),
                },
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    code: "internal_error".to_string(),
                    message: format!("task join error: {}", e),
                },
            }),
        )),
    }
}

/// GET /health — server health check.
async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let uptime = state.started_at.elapsed().as_secs_f64();
    let index_loaded = {
        let guard = state.index.read().await;
        guard.num_docs() > 0
    };
    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok".to_string(),
            uptime_seconds: uptime,
            index_loaded,
        }),
    )
}

/// GET /stats — index statistics.
async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let index = {
        let guard = state.index.read().await;
        Arc::clone(&guard)
    };

    let params = index.params();
    (
        StatusCode::OK,
        Json(StatsResponse {
            num_docs: index.num_docs(),
            vocab_size: index.vocab_size(),
            variant: params.method.to_string(),
            params: ParamsInfo {
                k1: params.k1,
                b: params.b,
                delta: params.delta,
            },
            avg_doc_length: index.avg_doc_len,
        }),
    )
}

/// POST /admin/reload — placeholder for atomic index reload.
///
/// In a full implementation this would accept a path or index data and
/// atomically swap the loaded index behind the `RwLock`. For now it returns
/// a 501 Not Implemented.
async fn handle_reload() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: ErrorDetail {
                code: "not_implemented".to_string(),
                message: "Index reload is not yet implemented".to_string(),
            },
        }),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{self, Request};
    use tower::ServiceExt;

    /// Helper: build a test router with a small index.
    fn test_app() -> Router {
        let index = crate::BM25Builder::new()
            .build_from_corpus(&[
                "the quick brown fox",
                "the lazy dog",
                "brown fox jumps over the lazy dog",
            ])
            .expect("build test index");

        let state = Arc::new(AppState {
            index: RwLock::new(Arc::new(index)),
            started_at: Instant::now(),
            approximate: false,
        });
        router(state)
    }

    #[tokio::test]
    async fn test_search_valid() {
        let app = test_app();
        let body = serde_json::json!({ "query": "quick", "top_k": 2 });
        let req = Request::builder()
            .method(http::Method::POST)
            .uri("/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: SearchResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json.query, "quick");
        assert!(!json.results.is_empty());
        assert!(json.latency_ms >= 0.0);
        assert_eq!(json.count, json.results.len());
    }

    #[tokio::test]
    async fn test_health() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: HealthResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json.status, "ok");
        assert!(json.uptime_seconds >= 0.0);
        assert!(json.index_loaded);
    }

    #[tokio::test]
    async fn test_stats() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::GET)
            .uri("/stats")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: StatsResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json.num_docs, 3);
        assert!(json.vocab_size > 0);
        assert_eq!(json.variant, "Lucene");
    }

    #[tokio::test]
    async fn test_search_bad_json() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::POST)
            .uri("/search")
            .header("content-type", "application/json")
            .body(Body::from(b"not json".to_vec()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 422 for deserialization failures by default.
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::UNPROCESSABLE_ENTITY
        );
    }

    #[tokio::test]
    async fn test_search_method_not_allowed() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::GET)
            .uri("/search")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    /// TEST-P7-001b: Empty query returns results (per UC-002 behavior).
    /// An empty query string has no known tokens, so the index returns
    /// an empty results set with count=0 and valid schema.
    #[tokio::test]
    async fn test_search_empty_query() {
        let app = test_app();
        let body = serde_json::json!({ "query": "", "top_k": 5 });
        let req = Request::builder()
            .method(http::Method::POST)
            .uri("/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: SearchResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json.query, "");
        assert_eq!(json.count, json.results.len());
        assert!(json.latency_ms >= 0.0);
    }

    /// TEST-P7-004b: Missing query field in JSON body returns 400 or 422.
    #[tokio::test]
    async fn test_search_missing_query_field() {
        let app = test_app();
        // Valid JSON but missing the required "query" field.
        let body = serde_json::json!({ "top_k": 5 });
        let req = Request::builder()
            .method(http::Method::POST)
            .uri("/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum returns 422 for missing required fields by default.
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
            "Expected 400 or 422, got {}",
            resp.status()
        );
    }

    /// TEST-P7-002b: Health endpoint includes all required fields.
    #[tokio::test]
    async fn test_health_full_schema() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Verify all three required fields are present.
        assert!(json.get("status").is_some(), "missing status field");
        assert!(json.get("uptime_seconds").is_some(), "missing uptime_seconds field");
        assert!(json.get("index_loaded").is_some(), "missing index_loaded field");
    }

    /// TEST-P7-003b: Stats endpoint returns params sub-object.
    #[tokio::test]
    async fn test_stats_includes_params() {
        let app = test_app();
        let req = Request::builder()
            .method(http::Method::GET)
            .uri("/stats")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: StatsResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json.num_docs, 3);
        assert!(json.vocab_size > 0);
        assert_eq!(json.variant, "Lucene");
        // Verify params sub-object has correct default values.
        assert_eq!(json.params.k1, 1.5);
        assert_eq!(json.params.b, 0.75);
        assert!(json.avg_doc_length > 0.0);
    }

    /// TEST-P7-008: Concurrent queries -- spawn multiple requests in parallel
    /// and verify they all succeed without blocking.
    #[tokio::test]
    async fn test_concurrent_queries() {
        let index = crate::BM25Builder::new()
            .build_from_corpus(&[
                "the quick brown fox",
                "the lazy dog",
                "brown fox jumps over the lazy dog",
            ])
            .expect("build test index");

        let state = Arc::new(AppState {
            index: RwLock::new(Arc::new(index)),
            started_at: Instant::now(),
            approximate: false,
        });

        let mut handles = Vec::new();
        for i in 0..10 {
            let app = router(state.clone());
            handles.push(tokio::spawn(async move {
                let query = match i % 3 {
                    0 => "quick",
                    1 => "lazy",
                    _ => "brown fox",
                };
                let body = serde_json::json!({ "query": query, "top_k": 2 });
                let req = Request::builder()
                    .method(http::Method::POST)
                    .uri("/search")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);

                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let json: SearchResponse = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(json.count, json.results.len());
                json
            }));
        }

        // Wait for all concurrent requests to complete.
        let mut completed = 0;
        for handle in handles {
            handle.await.expect("task should not panic");
            completed += 1;
        }
        assert_eq!(completed, 10, "All 10 concurrent requests should complete");
    }

    /// TEST-P7-009: Request body size limit -- oversized body returns 413.
    #[tokio::test]
    async fn test_body_size_limit() {
        let app = test_app();
        // DEFAULT_BODY_LIMIT is 1 MB. Send a body larger than that.
        let oversized = vec![b'x'; 2 * 1024 * 1024]; // 2 MB
        let req = Request::builder()
            .method(http::Method::POST)
            .uri("/search")
            .header("content-type", "application/json")
            .body(Body::from(oversized))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "Oversized body should return 413"
        );
    }
}
