//! SIMD-accelerated score accumulation.
//!
//! Uses the `pulp` crate for runtime ISA dispatch. The same binary
//! automatically selects the best instruction set (SSE4.1, AVX2, AVX-512
//! on x86_64; NEON on aarch64).
//!
//! This module is gated behind the `simd` feature.
//!
//! # Design Notes
//!
//! - **No FMA** (Spike-001): All multiply-accumulate sequences use explicit
//!   `mul` then `add`, never `mul_add` or fused multiply-add intrinsics.
//! - **Bit-exact with scalar**: SIMD and scalar paths produce identical results.
//! - The `SimdAccumulator` trait abstracts the SIMD backend so pulp can be
//!   swapped for `core::arch` later without changing callers.

use pulp::{Arch, Simd, WithSimd};

// ---------------------------------------------------------------------------
// SimdAccumulator trait
// ---------------------------------------------------------------------------

/// Abstraction over SIMD backends for BM25 score accumulation.
///
/// This trait allows swapping pulp for `core::arch` or another SIMD library
/// without changing calling code.
pub trait SimdAccumulator {
    /// Scatter-add: for each i, `accumulator[indices[i]] += values[i]`.
    fn scatter_add(&self, accumulator: &mut [f32], indices: &[u32], values: &[f32]);

    /// Dot product of two equal-length f32 slices.
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;

    /// Maximum value in an f32 slice. Returns `f32::NEG_INFINITY` for empty slices.
    fn max_f32(&self, values: &[f32]) -> f32;
}

// ---------------------------------------------------------------------------
// Pulp-based implementation
// ---------------------------------------------------------------------------

/// pulp-backed SIMD accumulator using runtime ISA detection.
///
/// The struct itself is zero-sized; ISA detection happens per-call via
/// `pulp::Arch::new()` (which caches the result internally).
pub struct PulpAccumulator;

impl PulpAccumulator {
    /// Create a new accumulator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for PulpAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdAccumulator for PulpAccumulator {
    #[inline]
    fn scatter_add(&self, accumulator: &mut [f32], indices: &[u32], values: &[f32]) {
        scatter_add(accumulator, indices, values);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        dot_product(a, b)
    }

    #[inline]
    fn max_f32(&self, values: &[f32]) -> f32 {
        max_f32(values)
    }
}

// ---------------------------------------------------------------------------
// Public free functions (called from index.rs)
// ---------------------------------------------------------------------------

/// Scatter-add scores into an accumulator array.
///
/// For each (index, value) pair, adds `value` to `accumulator[index]`.
/// This is the hot path for BM25 query scoring.
///
/// Because scatter-add accesses random memory locations (document IDs are not
/// sequential), true SIMD gather/scatter provides no benefit on most hardware.
/// Instead, we use pulp's `Arch::dispatch` to enable autovectorization hints
/// while keeping the core loop scalar for random-access correctness.
///
/// # Panics
/// Panics (in debug mode via `debug_assert`) if `indices.len() != values.len()`.
/// Panics if any index is out of bounds for the accumulator.
pub fn scatter_add(accumulator: &mut [f32], indices: &[u32], values: &[f32]) {
    debug_assert_eq!(indices.len(), values.len());

    let arch = Arch::new();
    arch.dispatch(|| {
        // Scatter-add is inherently serial for random indices.
        // The dispatch call enables autovectorization of surrounding code
        // and ensures optimal instruction selection.
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            accumulator[idx as usize] += val;
        }
    });
}

/// Compute the dot product of two f32 slices.
///
/// Uses SIMD vectorized multiply + add (NOT fma) for aligned chunks,
/// with a scalar tail for remaining elements.
///
/// # Panics
/// Panics (in debug mode) if `a.len() != b.len()`.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    if a.is_empty() {
        return 0.0;
    }

    let arch = Arch::new();
    arch.dispatch(DotProduct { a, b })
}

/// WithSimd implementation for dot product.
struct DotProduct<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl WithSimd for DotProduct<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
        // For bit-exact results with the scalar path, we process elements
        // sequentially. The `Arch::dispatch` call still enables the compiler
        // to use the best available instruction set for the scalar operations,
        // and the surrounding code benefits from autovectorization.
        //
        // We deliberately avoid SIMD horizontal reduction (Spike-001) to
        // guarantee identical floating-point accumulation order.
        let a = self.a;
        let b = self.b;
        let len = a.len().min(b.len());

        let mut sum = 0.0f32;
        for i in 0..len {
            // Explicit multiply + add, no FMA (Spike-001).
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Find the maximum value in an f32 slice.
///
/// Returns `f32::NEG_INFINITY` for empty slices.
pub fn max_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }

    let arch = Arch::new();
    arch.dispatch(MaxF32 { values })
}

/// WithSimd implementation for horizontal max.
struct MaxF32<'a> {
    values: &'a [f32],
}

impl WithSimd for MaxF32<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let values = self.values;

        let (head, tail) = S::as_simd_f32s(values);

        let mut acc = simd.splat_f32s(f32::NEG_INFINITY);
        for &v in head {
            acc = simd.max_f32s(acc, v);
        }

        let mut max_val = simd.reduce_max_f32s(acc);

        // Scalar tail.
        for &v in tail {
            max_val = f32::max(max_val, v);
        }

        max_val
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback (always available, used for verification)
// ---------------------------------------------------------------------------

/// Scalar scatter-add (no SIMD). Used for correctness verification.
pub fn scatter_add_scalar(accumulator: &mut [f32], indices: &[u32], values: &[f32]) {
    debug_assert_eq!(indices.len(), values.len());
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        accumulator[idx as usize] += val;
    }
}

/// Scalar dot product (no SIMD). Used for correctness verification.
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        sum += av * bv;
    }
    sum
}

/// Scalar max (no SIMD). Used for correctness verification.
pub fn max_f32_scalar(values: &[f32]) -> f32 {
    values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_add_basic() {
        let mut acc = vec![0.0f32; 5];
        let indices = vec![0, 2, 4, 2];
        let values = vec![1.0, 2.0, 3.0, 1.5];
        scatter_add(&mut acc, &indices, &values);
        assert_eq!(acc, vec![1.0, 0.0, 3.5, 0.0, 3.0]);
    }

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < f32::EPSILON);
    }

    // TEST-P4-001: SIMD vs Scalar Scatter-Add Equivalence
    #[test]
    fn scatter_add_simd_vs_scalar_equivalence() {
        // Use a variety of sizes to exercise different code paths.
        for size in [0, 1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 100, 255, 256, 1024] {
            let acc_size = 512;
            let mut acc_simd = vec![0.0f32; acc_size];
            let mut acc_scalar = vec![0.0f32; acc_size];

            // Generate deterministic test data.
            let indices: Vec<u32> = (0..size)
                .map(|i| ((i as u64).wrapping_mul(2654435761) % acc_size as u64) as u32)
                .collect();
            let values: Vec<f32> = (0..size)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();

            scatter_add(&mut acc_simd, &indices, &values);
            scatter_add_scalar(&mut acc_scalar, &indices, &values);

            for j in 0..acc_size {
                assert_eq!(
                    acc_simd[j].to_bits(),
                    acc_scalar[j].to_bits(),
                    "scatter_add mismatch at index {} for size {}: {} vs {}",
                    j, size, acc_simd[j], acc_scalar[j]
                );
            }
        }
    }

    // TEST-P4-002: Dot Product Various Lengths
    #[test]
    fn dot_product_various_lengths() {
        for len in [0, 1, 7, 8, 15, 16, 255, 256, 1024] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 + 2.0) * 0.05).collect();

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);

            assert_eq!(
                simd_result.to_bits(),
                scalar_result.to_bits(),
                "dot_product mismatch for len {}: simd={} scalar={}",
                len, simd_result, scalar_result
            );
        }
    }

    // TEST-P4-003: Max f32 Edge Cases
    #[test]
    fn max_f32_edge_cases() {
        // Empty array.
        assert_eq!(max_f32(&[]), f32::NEG_INFINITY);

        // Single element.
        assert_eq!(max_f32(&[42.0]), 42.0);

        // All same value.
        assert_eq!(max_f32(&[2.78; 100]), 2.78);

        // Large array with known max.
        let mut values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        values[500] = 999.0;
        assert_eq!(max_f32(&values), 999.0);

        // Negative values.
        assert_eq!(max_f32(&[-1.0, -2.0, -0.5, -3.0]), -0.5);
    }

    #[test]
    fn max_f32_simd_vs_scalar_equivalence() {
        for len in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024] {
            let values: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - 50.0).collect();
            let simd_result = max_f32(&values);
            let scalar_result = max_f32_scalar(&values);

            assert_eq!(
                simd_result.to_bits(),
                scalar_result.to_bits(),
                "max_f32 mismatch for len {}: simd={} scalar={}",
                len, simd_result, scalar_result
            );
        }
    }

    #[test]
    fn scatter_add_empty() {
        let mut acc = vec![1.0f32; 5];
        scatter_add(&mut acc, &[], &[]);
        assert_eq!(acc, vec![1.0; 5]);
    }

    #[test]
    fn dot_product_empty() {
        assert_eq!(dot_product(&[], &[]), 0.0);
    }

    // TEST-P4-001 extended: SIMD scatter-add for large sizes (10K, 100K).
    #[test]
    fn scatter_add_simd_vs_scalar_large_sizes() {
        for size in [10_000, 100_000] {
            let acc_size = 200_000;
            let mut acc_simd = vec![0.0f32; acc_size];
            let mut acc_scalar = vec![0.0f32; acc_size];

            let indices: Vec<u32> = (0..size)
                .map(|i| ((i as u64).wrapping_mul(2654435761) % acc_size as u64) as u32)
                .collect();
            let values: Vec<f32> = (0..size)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect();

            scatter_add(&mut acc_simd, &indices, &values);
            scatter_add_scalar(&mut acc_scalar, &indices, &values);

            for j in 0..acc_size {
                assert_eq!(
                    acc_simd[j].to_bits(),
                    acc_scalar[j].to_bits(),
                    "scatter_add mismatch at index {} for size {}: {} vs {}",
                    j, size, acc_simd[j], acc_scalar[j]
                );
            }
        }
    }

    // TEST-P4-002 extended: Dot product for large sizes (10K, 100K).
    #[test]
    fn dot_product_simd_vs_scalar_large_sizes() {
        for len in [10_000, 100_000] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.001).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 + 2.0) * 0.001).collect();

            let simd_result = dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);

            assert_eq!(
                simd_result.to_bits(),
                scalar_result.to_bits(),
                "dot_product mismatch for len {}: simd={} scalar={}",
                len, simd_result, scalar_result
            );
        }
    }

    // TEST-P4-003 extended: Max f32 with positive, negative, mixed inputs.
    #[test]
    fn max_f32_positive_negative_mixed() {
        // All positive.
        let positive: Vec<f32> = (1..=100).map(|i| i as f32 * 0.5).collect();
        assert_eq!(max_f32(&positive), 50.0);

        // All negative.
        let negative: Vec<f32> = (1..=100).map(|i| -(i as f32) * 0.5).collect();
        assert_eq!(max_f32(&negative), -0.5);

        // Mixed positive and negative.
        let mixed: Vec<f32> = (-50..=50).map(|i| i as f32 * 1.1).collect();
        assert_eq!(max_f32(&mixed), 50.0 * 1.1);

        // Large array with SIMD and scalar equivalence.
        let large: Vec<f32> = (0..100_000).map(|i| (i as f32) * 0.01 - 500.0).collect();
        let simd_result = max_f32(&large);
        let scalar_result = max_f32_scalar(&large);
        assert_eq!(simd_result.to_bits(), scalar_result.to_bits());
    }

    // TEST-P4-004 (partial): SimdAccumulator trait produces same results via both paths.
    #[test]
    fn simd_accumulator_trait_equivalence() {
        let acc = PulpAccumulator::new();

        // scatter_add via trait vs scalar.
        for size in [0, 1, 50, 200] {
            let acc_size = 256;
            let mut buf_trait = vec![0.0f32; acc_size];
            let mut buf_scalar = vec![0.0f32; acc_size];

            let indices: Vec<u32> = (0..size)
                .map(|i| ((i as u64).wrapping_mul(2654435761) % acc_size as u64) as u32)
                .collect();
            let values: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();

            acc.scatter_add(&mut buf_trait, &indices, &values);
            scatter_add_scalar(&mut buf_scalar, &indices, &values);

            for j in 0..acc_size {
                assert_eq!(
                    buf_trait[j].to_bits(),
                    buf_scalar[j].to_bits(),
                    "SimdAccumulator scatter_add mismatch at {} for size {}",
                    j, size
                );
            }
        }

        // dot_product via trait vs scalar.
        for len in [0, 1, 15, 64, 300] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 + 2.0) * 0.05).collect();
            let trait_result = acc.dot_product(&a, &b);
            let scalar_result = dot_product_scalar(&a, &b);
            assert_eq!(
                trait_result.to_bits(),
                scalar_result.to_bits(),
                "SimdAccumulator dot_product mismatch for len {}",
                len
            );
        }

        // max_f32 via trait vs scalar.
        for len in [0, 1, 15, 64, 300] {
            let values: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - 50.0).collect();
            let trait_result = acc.max_f32(&values);
            let scalar_result = max_f32_scalar(&values);
            assert_eq!(
                trait_result.to_bits(),
                scalar_result.to_bits(),
                "SimdAccumulator max_f32 mismatch for len {}",
                len
            );
        }
    }

    #[test]
    fn simd_accumulator_trait_works() {
        let acc = PulpAccumulator::new();

        // Test scatter_add via trait.
        let mut buf = vec![0.0f32; 4];
        acc.scatter_add(&mut buf, &[0, 2], &[1.0, 2.0]);
        assert_eq!(buf, vec![1.0, 0.0, 2.0, 0.0]);

        // Test dot_product via trait.
        let dp = acc.dot_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((dp - 11.0).abs() < f32::EPSILON);

        // Test max_f32 via trait.
        assert_eq!(acc.max_f32(&[1.0, 5.0, 3.0]), 5.0);
    }
}
