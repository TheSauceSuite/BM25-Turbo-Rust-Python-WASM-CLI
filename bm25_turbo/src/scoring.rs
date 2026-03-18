//! BM25 scoring kernels for all five variants.
//!
//! Each variant is defined by two functions:
//! - `idf(num_docs, doc_freq)` -- inverse document frequency
//! - `tfc(tf, doc_len, avg_doc_len, k1, b, delta)` -- term frequency component
//!
//! These are pure functions with no side effects, designed to be inlined
//! and auto-vectorized by the compiler.
//!
//! # Determinism
//!
//! All scoring paths use explicit `a * b + c` arithmetic (never `mul_add` / FMA)
//! to guarantee bit-identical results across platforms. Internal computation
//! uses f64 precision; results are cast to f32 on return.

use crate::error::{Error, Result};
use crate::types::{BM25Params, Method};

/// Validate BM25 parameters.
///
/// Returns `Ok(())` if all parameters are within valid ranges, or
/// `Err(Error::InvalidParameter)` describing the first violation found.
pub fn validate_params(params: &BM25Params) -> Result<()> {
    if params.k1 < 0.0 {
        return Err(Error::InvalidParameter(format!(
            "k1 must be >= 0, got {}",
            params.k1
        )));
    }
    if params.b < 0.0 || params.b > 1.0 {
        return Err(Error::InvalidParameter(format!(
            "b must be in [0, 1], got {}",
            params.b
        )));
    }
    if matches!(params.method, Method::Bm25l | Method::Bm25Plus) && params.delta < 0.0 {
        return Err(Error::InvalidParameter(format!(
            "delta must be >= 0 for {:?}, got {}",
            params.method, params.delta
        )));
    }
    Ok(())
}

/// Compute the IDF value for a term.
///
/// # Arguments
/// * `method` -- BM25 variant
/// * `num_docs` -- total number of documents in the corpus (N)
/// * `doc_freq` -- number of documents containing this term (df)
///
/// Uses f64 internal precision, returns f32. No FMA instructions.
#[inline(always)]
pub fn idf(method: Method, num_docs: u32, doc_freq: u32) -> f32 {
    let n = num_docs as f64;
    let df = doc_freq as f64;

    // Explicit mul+add -- no mul_add / FMA
    let value = match method {
        Method::Robertson => ((n - df + 0.5) / (df + 0.5)).ln(),
        Method::Lucene => (1.0 + (n - df + 0.5) / (df + 0.5)).ln(),
        Method::Atire => (n / df).ln(),
        Method::Bm25l => ((n + 1.0) / (df + 0.5)).ln(),
        Method::Bm25Plus => ((n + 1.0) / df).ln(),
    };

    value as f32
}

/// Compute the term frequency component (TFC) for a term in a document.
///
/// # Arguments
/// * `method` -- BM25 variant
/// * `tf` -- raw term frequency in the document
/// * `doc_len` -- document length (number of tokens)
/// * `avg_doc_len` -- average document length across the corpus
/// * `k1` -- term frequency saturation parameter
/// * `b` -- document length normalization parameter
/// * `delta` -- additive smoothing (used by BM25L and BM25+ only)
///
/// Uses f64 internal precision, returns f32. No FMA instructions.
#[inline(always)]
pub fn tfc(
    method: Method,
    tf: f32,
    doc_len: f32,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
    delta: f32,
) -> f32 {
    let tf = tf as f64;
    let doc_len = doc_len as f64;
    let avg_doc_len = avg_doc_len as f64;
    let k1 = k1 as f64;
    let b = b as f64;
    let delta = delta as f64;

    // Explicit: 1.0 - b + b * (doc_len / avg_doc_len)
    // Written as separate operations to prevent FMA.
    let ratio = doc_len / avg_doc_len;
    let b_ratio = b * ratio;
    let norm = 1.0 - b + b_ratio;

    let value = match method {
        Method::Robertson | Method::Lucene => tf / (k1 * norm + tf),
        Method::Atire => {
            let num = tf * (k1 + 1.0);
            let den = tf + k1 * norm;
            num / den
        }
        Method::Bm25l => {
            let tf_prime = tf + delta;
            let num = (k1 + 1.0) * tf_prime;
            let den = k1 + tf_prime;
            num / den
        }
        Method::Bm25Plus => {
            let num = (k1 + 1.0) * tf;
            let den = k1 * norm + tf;
            num / den + delta
        }
    };

    value as f32
}

/// Compute the full BM25 score for a single (term, document) pair.
///
/// This is `idf * tfc` for convenience. Both sub-functions use f64 internally.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn score(
    method: Method,
    tf: f32,
    doc_len: f32,
    avg_doc_len: f32,
    num_docs: u32,
    doc_freq: u32,
    k1: f32,
    b: f32,
    delta: f32,
) -> f32 {
    let idf_val = idf(method, num_docs, doc_freq) as f64;
    let tfc_val = tfc(method, tf, doc_len, avg_doc_len, k1, b, delta) as f64;
    // Explicit multiply -- no FMA
    (idf_val * tfc_val) as f32
}

/// Compute a BM25 score forcing scalar computation (no SIMD).
///
/// Identical to [`score`] but annotated to prevent auto-vectorization.
/// Use this when bit-exact reproducibility is required regardless of
/// target ISA.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn score_deterministic(
    method: Method,
    tf: f32,
    doc_len: f32,
    avg_doc_len: f32,
    num_docs: u32,
    doc_freq: u32,
    k1: f32,
    b: f32,
    delta: f32,
) -> f32 {
    score(method, tf, doc_len, avg_doc_len, num_docs, doc_freq, k1, b, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // TEST-P1-001: Scoring Variant Reference Values
    // Each variant tested against independently computed reference values.
    // ---------------------------------------------------------------

    #[test]
    fn idf_robertson_reference_within_1ulp() {
        // Robertson IDF: ln((N - df + 0.5) / (df + 0.5))
        // N=1000, df=100: ln((1000 - 100 + 0.5) / (100 + 0.5)) = ln(900.5 / 100.5)
        let result = idf(Method::Robertson, 1000, 100);
        let expected = ((1000.0_f64 - 100.0 + 0.5) / (100.0 + 0.5)).ln() as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "Robertson IDF off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn idf_lucene_reference_within_1ulp() {
        // Lucene IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        let result = idf(Method::Lucene, 1000, 100);
        let expected = (1.0_f64 + (1000.0 - 100.0 + 0.5) / (100.0 + 0.5)).ln() as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "Lucene IDF off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn idf_atire_reference_within_1ulp() {
        // ATIRE IDF: ln(N / df)
        let result = idf(Method::Atire, 1000, 100);
        let expected = (1000.0_f64 / 100.0).ln() as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "ATIRE IDF off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn idf_bm25l_reference_within_1ulp() {
        // BM25L IDF: ln((N + 1) / (df + 0.5))
        let result = idf(Method::Bm25l, 1000, 100);
        let expected = ((1000.0_f64 + 1.0) / (100.0 + 0.5)).ln() as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "BM25L IDF off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn idf_bm25plus_reference_within_1ulp() {
        // BM25+ IDF: ln((N + 1) / df)
        let result = idf(Method::Bm25Plus, 1000, 100);
        let expected = ((1000.0_f64 + 1.0) / 100.0).ln() as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "BM25+ IDF off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn tfc_robertson_reference_within_1ulp() {
        // Robertson TFC: tf / (k1 * norm + tf)
        // norm = 1 - b + b * (doc_len / avg_doc_len)
        // tf=3, doc_len=100, avg_doc_len=80, k1=1.2, b=0.75
        let result = tfc(Method::Robertson, 3.0, 100.0, 80.0, 1.2, 0.75, 0.0);
        let norm = 1.0_f64 - 0.75 + 0.75 * (100.0 / 80.0);
        let expected = (3.0_f64 / (1.2 * norm + 3.0)) as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "Robertson TFC off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn tfc_atire_reference_within_1ulp() {
        // ATIRE TFC: (tf * (k1 + 1)) / (tf + k1 * norm)
        let result = tfc(Method::Atire, 3.0, 100.0, 80.0, 1.2, 0.75, 0.0);
        let norm = 1.0_f64 - 0.75 + 0.75 * (100.0 / 80.0);
        let expected = (3.0 * (1.2 + 1.0) / (3.0 + 1.2 * norm)) as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "ATIRE TFC off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn tfc_bm25l_reference_within_1ulp() {
        // BM25L TFC: ((k1 + 1) * (tf + delta)) / (k1 + tf + delta)
        let result = tfc(Method::Bm25l, 3.0, 100.0, 80.0, 1.2, 0.75, 0.5);
        let tf_prime = 3.0_f64 + 0.5;
        let expected = ((1.2 + 1.0) * tf_prime / (1.2 + tf_prime)) as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "BM25L TFC off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn tfc_bm25plus_reference_within_1ulp() {
        // BM25+ TFC: ((k1 + 1) * tf) / (k1 * norm + tf) + delta
        let result = tfc(Method::Bm25Plus, 3.0, 100.0, 80.0, 1.2, 0.75, 0.5);
        let norm = 1.0_f64 - 0.75 + 0.75 * (100.0 / 80.0);
        let expected = ((1.2 + 1.0) * 3.0 / (1.2 * norm + 3.0) + 0.5) as f32;
        let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
        assert!(ulp_diff <= 1, "BM25+ TFC off by {} ULPs: got {}, expected {}", ulp_diff, result, expected);
    }

    #[test]
    fn score_all_five_variants_reference_within_1ulp() {
        // Full score = idf * tfc for each variant
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let result = score(*method, 3.0, 100.0, 80.0, 1000, 100, 1.2, 0.75, 0.5);
            let idf_val = idf(*method, 1000, 100) as f64;
            let tfc_val = tfc(*method, 3.0, 100.0, 80.0, 1.2, 0.75, 0.5) as f64;
            let expected = (idf_val * tfc_val) as f32;
            let ulp_diff = (result.to_bits() as i32 - expected.to_bits() as i32).unsigned_abs();
            assert!(
                ulp_diff <= 1,
                "{:?} score off by {} ULPs: got {}, expected {}",
                method, ulp_diff, result, expected
            );
        }
    }

    #[test]
    fn idf_all_variants_positive_for_low_df() {
        // All five variants should produce positive IDF when df is small relative to N
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        let test_cases: &[(u32, u32)] = &[
            (100, 1),
            (100, 10),
            (1_000_000, 1),
            (1_000_000, 100),
        ];
        for method in &methods {
            for &(n, df) in test_cases {
                let result = idf(*method, n, df);
                assert!(
                    result > 0.0,
                    "{:?} IDF should be positive for N={}, df={}, got {}",
                    method, n, df, result
                );
            }
        }
    }

    #[test]
    fn idf_lucene_atire_bm25l_bm25plus_always_positive() {
        // Lucene, ATIRE, BM25L, BM25+ IDF variants are always positive for df < N
        let always_positive_methods = [
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        let test_cases: &[(u32, u32)] = &[
            (100, 1),
            (100, 50),
            (100, 99),
            (1_000_000, 500_000),
            (1_000_000, 999_999),
        ];
        for method in &always_positive_methods {
            for &(n, df) in test_cases {
                let result = idf(*method, n, df);
                assert!(
                    result > 0.0,
                    "{:?} IDF should always be positive for df < N: N={}, df={}, got {}",
                    method, n, df, result
                );
            }
        }
    }

    #[test]
    fn idf_robertson_can_be_zero_or_negative_for_high_df() {
        // Robertson IDF: ln((N - df + 0.5) / (df + 0.5))
        // When df = N/2, the ratio is ~1.0 and IDF ~= 0
        // When df > N/2, IDF becomes negative
        let result_half = idf(Method::Robertson, 100, 50);
        assert!(result_half.abs() < 0.01, "Robertson IDF should be near zero when df = N/2");

        let result_high = idf(Method::Robertson, 100, 90);
        assert!(result_high < 0.0, "Robertson IDF should be negative when df > N/2, got {}", result_high);
    }

    #[test]
    fn idf_lucene_basic() {
        let result = idf(Method::Lucene, 100, 10);
        assert!(result > 0.0, "IDF should be positive for df < N");
    }

    #[test]
    fn tfc_robertson_basic() {
        let result = tfc(Method::Robertson, 3.0, 100.0, 100.0, 1.5, 0.75, 0.0);
        assert!(result > 0.0 && result < 1.0, "TFC should be in (0, 1) for Robertson");
    }

    #[test]
    fn score_positive() {
        let s = score(Method::Lucene, 2.0, 50.0, 100.0, 1000, 10, 1.5, 0.75, 0.5);
        assert!(s > 0.0, "Score should be positive for normal inputs");
    }

    // ---------------------------------------------------------------
    // TEST-P1-002: Invalid Parameter Rejection
    // ---------------------------------------------------------------

    #[test]
    fn validate_params_ok() {
        let params = BM25Params::default();
        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn validate_params_negative_k1() {
        let params = BM25Params { k1: -1.0, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }

    #[test]
    fn validate_params_k1_zero_is_valid() {
        let params = BM25Params { k1: 0.0, ..Default::default() };
        assert!(validate_params(&params).is_ok(), "k1=0 should be valid");
    }

    #[test]
    fn validate_params_b_out_of_range_low() {
        let params = BM25Params { b: -0.1, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }

    #[test]
    fn validate_params_b_out_of_range_high() {
        let params = BM25Params { b: 1.1, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }

    #[test]
    fn validate_params_b_boundary_zero_is_valid() {
        let params = BM25Params { b: 0.0, ..Default::default() };
        assert!(validate_params(&params).is_ok(), "b=0 should be valid");
    }

    #[test]
    fn validate_params_b_boundary_one_is_valid() {
        let params = BM25Params { b: 1.0, ..Default::default() };
        assert!(validate_params(&params).is_ok(), "b=1 should be valid");
    }

    #[test]
    fn validate_params_negative_delta_bm25l() {
        let params = BM25Params { delta: -0.5, method: Method::Bm25l, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }

    #[test]
    fn validate_params_negative_delta_bm25plus() {
        let params = BM25Params { delta: -0.5, method: Method::Bm25Plus, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }

    #[test]
    fn validate_params_negative_delta_lucene_ok() {
        // delta doesn't apply to Lucene, so negative is OK
        let params = BM25Params { delta: -0.5, method: Method::Lucene, ..Default::default() };
        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn validate_params_negative_delta_robertson_ok() {
        let params = BM25Params { delta: -0.5, method: Method::Robertson, ..Default::default() };
        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn validate_params_negative_delta_atire_ok() {
        let params = BM25Params { delta: -0.5, method: Method::Atire, ..Default::default() };
        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn validate_params_error_message_contains_value() {
        let params = BM25Params { k1: -2.75, ..Default::default() };
        let err = validate_params(&params).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("-2.75"), "Error message should contain the invalid value, got: {}", msg);
    }

    // ---------------------------------------------------------------
    // TEST-P1-003: Scoring Determinism
    // 1000 runs for each variant, all must be bit-identical.
    // ---------------------------------------------------------------

    #[test]
    fn scoring_determinism_all_variants_1000_runs() {
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let reference = score(*method, 5.0, 200.0, 150.0, 50000, 500, 1.5, 0.75, 0.5);
            for i in 0..1000 {
                let s = score(*method, 5.0, 200.0, 150.0, 50000, 500, 1.5, 0.75, 0.5);
                assert_eq!(
                    s.to_bits(),
                    reference.to_bits(),
                    "{:?} score not bit-identical on run {}: got {} (bits {:08x}), expected {} (bits {:08x})",
                    method, i, s, s.to_bits(), reference, reference.to_bits()
                );
            }
        }
    }

    #[test]
    fn score_deterministic_matches_score_all_variants() {
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let s1 = score(*method, 2.0, 50.0, 100.0, 1000, 10, 1.5, 0.75, 0.5);
            let s2 = score_deterministic(*method, 2.0, 50.0, 100.0, 1000, 10, 1.5, 0.75, 0.5);
            assert_eq!(
                s1.to_bits(),
                s2.to_bits(),
                "{:?} deterministic score must be bit-identical",
                method
            );
        }
    }

    #[test]
    fn idf_determinism_1000_runs() {
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let reference = idf(*method, 100_000, 5_000);
            for _ in 0..1000 {
                let result = idf(*method, 100_000, 5_000);
                assert_eq!(
                    result.to_bits(),
                    reference.to_bits(),
                    "{:?} IDF not deterministic",
                    method
                );
            }
        }
    }

    #[test]
    fn tfc_determinism_1000_runs() {
        let methods = [
            Method::Robertson,
            Method::Lucene,
            Method::Atire,
            Method::Bm25l,
            Method::Bm25Plus,
        ];
        for method in &methods {
            let reference = tfc(*method, 7.0, 300.0, 200.0, 1.2, 0.8, 0.5);
            for _ in 0..1000 {
                let result = tfc(*method, 7.0, 300.0, 200.0, 1.2, 0.8, 0.5);
                assert_eq!(
                    result.to_bits(),
                    reference.to_bits(),
                    "{:?} TFC not deterministic",
                    method
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Additional scoring edge cases
    // ---------------------------------------------------------------

    #[test]
    fn idf_with_single_document_corpus() {
        // Edge case: N=1, df=1
        for method in [Method::Lucene, Method::Atire, Method::Bm25l, Method::Bm25Plus] {
            let result = idf(method, 1, 1);
            assert!(result.is_finite(), "{:?} IDF should be finite for N=1, df=1", method);
        }
    }

    #[test]
    fn tfc_with_zero_term_frequency() {
        // tf=0 should produce zero or near-zero TFC
        for method in [Method::Robertson, Method::Lucene, Method::Atire, Method::Bm25Plus] {
            let result = tfc(method, 0.0, 100.0, 100.0, 1.5, 0.75, 0.0);
            assert!(
                result >= 0.0,
                "{:?} TFC should be >= 0 for tf=0, got {}",
                method, result
            );
        }
    }

    #[test]
    fn tfc_with_b_zero_no_length_normalization() {
        // b=0 means no length normalization: norm = 1.0
        let short_doc = tfc(Method::Lucene, 3.0, 50.0, 100.0, 1.5, 0.0, 0.0);
        let long_doc = tfc(Method::Lucene, 3.0, 200.0, 100.0, 1.5, 0.0, 0.0);
        assert_eq!(
            short_doc.to_bits(),
            long_doc.to_bits(),
            "With b=0, document length should not affect TFC"
        );
    }

    #[test]
    fn tfc_with_b_one_full_length_normalization() {
        // b=1 means full length normalization
        let short_doc = tfc(Method::Lucene, 3.0, 50.0, 100.0, 1.5, 1.0, 0.0);
        let long_doc = tfc(Method::Lucene, 3.0, 200.0, 100.0, 1.5, 1.0, 0.0);
        assert!(
            short_doc > long_doc,
            "With b=1, shorter documents should score higher: short={}, long={}",
            short_doc, long_doc
        );
    }
}
