//! Numeric primitives replicating the Python engine's quantization discipline.
//!
//! Authority: `scripts/kartrider_vehicle_engine.py` module-level helpers
//! (`_f32`, `_fadd`, `_fsub`, `_fmul`, `_fdiv`, `_dot3_f32`, `_length3_f32`,
//! `_sub3_f32`, `_cross3_f32`, `_matrix_columns`).
//!
//! The Python engine stores state as Python floats (f64) and computes in f64
//! except where it explicitly quantizes through binary32. Rust mirrors that
//! exactly: all values are `f64`; `f32q` rounds through binary32
//! (`v as f32 as f64`, exact widening back). The `f*` helpers quantize both
//! operands and the result, matching `_f32(_f32(a) op _f32(b))`.
//!
//! Rules from docs/kart-physics-rust-handoff-2026-07-28.md §5:
//! - never reorder/reassociate: Python's evaluation order is authoritative;
//! - never use `mul_add` (no FMA);
//! - negative zero matters: `_f32(-value)` and `0.0 - value` differ.

/// `_f32`: round through IEEE-754 binary32, widen back to f64 (lossless).
#[inline]
pub fn f32q(value: f64) -> f64 {
    value as f32 as f64
}

/// `_fadd(a, b)` = `_f32(_f32(a) + _f32(b))`.
#[inline]
pub fn fadd(a: f64, b: f64) -> f64 {
    f32q(f32q(a) + f32q(b))
}

/// `_fsub(a, b)` = `_f32(_f32(a) - _f32(b))`.
#[inline]
pub fn fsub(a: f64, b: f64) -> f64 {
    f32q(f32q(a) - f32q(b))
}

/// `_fmul(a, b)` = `_f32(_f32(a) * _f32(b))`.
#[inline]
pub fn fmul(a: f64, b: f64) -> f64 {
    f32q(f32q(a) * f32q(b))
}

/// `_fdiv(a, b)` = `_f32(_f32(a) / _f32(b))`.
#[inline]
pub fn fdiv(a: f64, b: f64) -> f64 {
    f32q(f32q(a) / f32q(b))
}

/// `_f32_vector`: elementwise quantization.
#[inline]
pub fn f32_vector(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = f32q(*v);
    }
}

/// Naive left fold `a[0]*b[0] + a[1]*b[1] + a[2]*b[2]`.
///
/// This matches ONLY Python sites written as an explicit
/// `a[0]*b[0] + a[1]*b[1] + a[2]*b[2]` expression. It must NOT be used for
/// Python `_dot3` / `sum(a[i] * b[i] ...)` sites: CPython 3.12+ `sum()` is
/// Neumaier-compensated and starts at int `0`, so it maps an all-`-0.0`
/// product list to `+0.0` where this fold keeps `-0.0`, and it recovers the
/// cancelled term in `sum([1e16, 1.0, -1e16]) == 1.0` where this fold
/// yields `0.0`. Use [`dot3_sum`] for every `sum()`-semantics dot product.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `_dot3` (kartrider_vehicle_engine.py:663-664):
/// `sum(float(a[i]) * float(b[i]) for i in range(3))` — a CPython `sum()`
/// over three floats, i.e. Neumaier summation via [`sum_f64`], NOT a naive
/// left fold. The distinction is observable: with all three products equal
/// to `-0.0` (e.g. a forward axis carrying `-0.0` components under
/// `forward_axis_sign = -1.0` dotted against a zero velocity) CPython
/// `sum()` yields `+0.0` while [`dot3`] yields `-0.0` (this broke
/// `drive_target.exponent_input` sign parity in float64 mode). Every Python
/// `_dot3` or `sum(x * y ...)` dot product must route through this helper.
#[inline]
pub fn dot3_sum(a: [f64; 3], b: [f64; 3]) -> f64 {
    sum_f64(&[a[0] * b[0], a[1] * b[1], a[2] * b[2]])
}

/// `_dot3_f32`: running `_fadd` accumulator (starts at `+0.0`) over `_fmul`
/// products, sequential. The explicit zero start matters for `-0.0` first
/// products: `_fadd(0.0, -0.0) == +0.0`.
#[inline]
pub fn dot3_f32(a: [f64; 3], b: [f64; 3]) -> f64 {
    let mut value = f32q(0.0);
    for index in 0..3 {
        value = fadd(value, fmul(a[index], b[index]));
    }
    value
}

/// `_length3`: f64 `math.sqrt(sum(v*v))` (Neumaier, see [`sum_f64`]) or f32
/// chain, per `numeric_mode`. The f64 branch is NOT a naive fold: for
/// `[1.0, 1e-8, 4e-8]` CPython's compensated square-sum is one ulp above the
/// fold and `sqrt` propagates the difference.
#[inline]
pub fn length3(values: [f64; 3], float32_mode: bool) -> f64 {
    if float32_mode {
        length3_f32(values)
    } else {
        sum_f64(&[
            values[0] * values[0],
            values[1] * values[1],
            values[2] * values[2],
        ])
        .sqrt()
    }
}

/// `_length3_f32`: running `_fadd` accumulator (starts at `+0.0`) over
/// `_fmul(v, v)`, then `_f32(sqrt)`.
#[inline]
pub fn length3_f32(values: [f64; 3]) -> f64 {
    let mut length_squared = f32q(0.0);
    for value in values {
        length_squared = fadd(length_squared, fmul(value, value));
    }
    f32q(length_squared.sqrt())
}

/// `_sub3_f32`: elementwise quantized subtraction, fixed operand order.
#[inline]
pub fn sub3_f32(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        fsub(a[0], b[0]),
        fsub(a[1], b[1]),
        fsub(a[2], b[2]),
    ]
}

/// `_cross3_f32`: elementwise quantized with fixed operand order.
#[inline]
pub fn cross3_f32(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        fsub(fmul(a[1], b[2]), fmul(a[2], b[1])),
        fsub(fmul(a[2], b[0]), fmul(a[0], b[2])),
        fsub(fmul(a[0], b[1]), fmul(a[1], b[0])),
    ]
}

/// `_matrix_columns`: row-major storage; columns are the vehicle axes.
#[inline]
pub fn matrix_columns(m: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [m[0], m[3], m[6]],
        [m[1], m[4], m[7]],
        [m[2], m[5], m[8]],
    ]
}

/// f64 bit pattern, for bit-exact parity traces (Python: `struct.pack('<d')`).
#[inline]
pub fn f64_bits(value: f64) -> u64 {
    value.to_bits()
}

/// CPython two-operand `min(a, b)`: `b if b < a else a`. Unlike
/// `f64::min`, this propagates NaN exactly like CPython (`min(nan, 1.0)`
/// is `nan`, `min(1.0, nan)` is `1.0`) and keeps the first operand on
/// ties, so `min(-0.0, 0.0)` stays `-0.0`.
#[inline]
pub fn min2_f64(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// CPython two-operand `max(a, b)`: `b if b > a else a`. Same NaN/tie
/// reasoning as [`min2_f64`]; `max(-0.0, 0.0)` stays `-0.0` while
/// `max(0.0, -0.0)` is `+0.0`.
#[inline]
pub fn max2_f64(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// CPython 3.12+ `sum()` over floats: Neumaier (compensated Kahan-Babuska)
/// summation, NOT a naive left fold. Verified bit-exact against CPython
/// 3.14.6 over 200k random vectors (2026-07-28). Starting at int 0 and
/// summing only floats, CPython accumulates `hi`/`lo` as below and returns
/// `hi + lo`. Every `sum(...)` in the Python authority that sees more than
/// two floats must go through this helper; two-element sums happen to
/// coincide with plain `a + b` (the compensation rounds back), but routing
/// them here too keeps the transliteration honest.
pub fn sum_f64(values: &[f64]) -> f64 {
    let mut hi = 0.0_f64;
    let mut lo = 0.0_f64;
    for &x in values {
        let t = hi + x;
        if hi.abs() >= x.abs() {
            lo += (hi - t) + x;
        } else {
            lo += (x - t) + hi;
        }
        hi = t;
    }
    hi + lo
}

/// Dekker `dl_mul` (CPython mathmodule.c): lossless product split into
/// `(lo, hi)` using ONE FMA. `mul_add` is used deliberately here — this
/// replicates CPython's algorithm, which calls the C `fma`; it is not an
/// FMA contraction of engine arithmetic.
#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let hi = x * y;
    (x.mul_add(y, -hi), hi)
}

/// `dl_fast_sum` (CPython mathmodule.c): Fast2Sum lossless addition,
/// `(lo, hi)`. Requires `|a| >= |b|`; both call sites in `vector_norm`
/// satisfy it.
#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    ((a - hi) + b, hi)
}

/// `ldexp(1.0, k)` for k in -1074..=1023 (exact power of two).
#[inline]
fn pow2(k: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&k));
    if k >= -1021 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// CPython 3.12+ `math.hypot(a, b)` (Modules/mathmodule.c `vector_norm`
/// with n=2): lossless power-of-two scaling, double-double accumulation
/// (`dl_mul`/`dl_fast_sum`), then a differential correction. Verified
/// bit-exact against CPython 3.14.6 over 1M random pairs plus subnormal
/// and huge-magnitude edges (2026-07-28).
///
/// This is NOT the C library hypot: Rust `f64::hypot` matches the platform
/// libm `hypot`, which differs from CPython's algorithm on ~16% of random
/// inputs. Every `math.hypot` in the Python authority must go through this
/// helper.
pub fn cpython_hypot2(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let mut vec = [a.abs(), b.abs()];
    // math_hypot_impl: max starts at 0.0 and takes strict `x > max`.
    let mut max = 0.0_f64;
    for value in vec {
        if value > max {
            max = value;
        }
    }
    cpython_vector_norm2(&mut vec, max)
}

/// `vector_norm(n=2, vec, max, found_nan=0)` with finite, non-NaN inputs.
fn cpython_vector_norm2(vec: &mut [f64; 2], max: f64) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if max == 0.0 {
        return max;
    }
    // frexp(max): exponent of a positive finite double. Subnormals take the
    // DBL_MIN rescaling path (max_e < -1023).
    let exp_field = ((max.to_bits() >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        for value in vec.iter_mut() {
            *value /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * cpython_vector_norm2(vec, max / f64::MIN_POSITIVE);
    }
    cpython_vector_norm_scaled(vec, max)
}

/// The scaled double-double accumulation + differential-correction tail of
/// CPython `vector_norm`, shared by `math.hypot` (n=2) and `math.dist`.
/// `vec` entries must be finite, `max` must be the maximum entry, positive,
/// and normal (not subnormal).
fn cpython_vector_norm_scaled(vec: &mut [f64], max: f64) -> f64 {
    let exp_field = ((max.to_bits() >> 52) & 0x7ff) as i32;
    let max_e = exp_field - 1022;
    let scale = pow2(-max_e);
    let mut csum = 1.0_f64;
    let mut frac1 = 0.0_f64;
    let mut frac2 = 0.0_f64;
    for value in vec.iter() {
        let x = *value * scale; // lossless scaling
        let (pr_lo, pr_hi) = dl_mul(x, x); // lossless squaring
        let (sm_lo, sm_hi) = dl_fast_sum(csum, pr_hi); // lossless addition
        csum = sm_hi;
        frac1 += pr_lo; // lossy addition
        frac2 += sm_lo; // lossy addition
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (pr_lo, pr_hi) = dl_mul(-h, h);
    let (sm_lo, sm_hi) = dl_fast_sum(csum, pr_hi);
    csum = sm_hi;
    frac1 += pr_lo;
    frac2 += sm_lo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h); // differential correction
    h / scale
}

/// CPython 3.12+ `math.dist(p, q)` for two 3D points (Modules/mathmodule.c
/// `math_dist_impl` -> `vector_norm` with n=3): per-component absolute
/// differences, then the same lossless power-of-two scaling, double-double
/// accumulation and differential correction as [`cpython_hypot2`]. Rust
/// `f64::hypot` chains or `sqrt(sum())` do NOT match this algorithm; every
/// `math.dist` in the Python authority must go through this helper.
///
/// NaN inputs return NaN up front (CPython reports the first NaN through the
/// `found_nan` path); the rail leaf modules validate finiteness before
/// calling `math.dist`, so the nuance of CPython's mixed NaN/inf handling is
/// unreachable from those call sites.
pub fn cpython_dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let mut vec = [
        (a[0] - b[0]).abs(),
        (a[1] - b[1]).abs(),
        (a[2] - b[2]).abs(),
    ];
    // math_dist_impl: max starts at 0.0 and takes strict `x > max`.
    let mut max = 0.0_f64;
    let mut found_nan = false;
    for value in vec {
        found_nan = found_nan || value.is_nan();
        if value > max {
            max = value;
        }
    }
    if found_nan {
        return f64::NAN;
    }
    if max.is_infinite() {
        return max;
    }
    if max == 0.0 {
        return max;
    }
    // frexp(max): subnormals take the DBL_MIN rescaling path.
    let exp_field = ((max.to_bits() >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        for value in vec.iter_mut() {
            *value /= f64::MIN_POSITIVE;
        }
        let scaled = cpython_dist3_scaled(vec, max / f64::MIN_POSITIVE);
        return f64::MIN_POSITIVE * scaled;
    }
    cpython_vector_norm_scaled(&mut vec, max)
}

/// Subnormal-rescaling recursion helper for [`cpython_dist3`] (the scaled
/// `max` is normal here).
fn cpython_dist3_scaled(vec: [f64; 3], max: f64) -> f64 {
    let mut vec = vec;
    cpython_vector_norm_scaled(&mut vec, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32q_rounds_through_binary32() {
        assert_eq!(f32q(0.1), 0.1_f32 as f64);
        assert_eq!(f32q(1.0), 1.0);
    }

    #[test]
    fn fadd_quantizes_operands_and_result() {
        // 0.1f64 + 0.2f64 in f64 != fadd(0.1, 0.2) (f32 quantized).
        assert_ne!(0.1 + 0.2, fadd(0.1, 0.2));
        assert_eq!(fadd(0.1, 0.2), (0.1_f32 + 0.2_f32) as f64);
    }

    #[test]
    fn dot3_is_sequential_not_fma() {
        let a = [1.000_000_1_f64, 3.0, -2.0];
        let b = [1.000_000_1_f64, 1.0e-7, 4.0];
        assert_ne!(dot3_f32(a, b).to_bits(), dot3(a, b).to_bits());
    }

    #[test]
    fn negative_zero_is_preserved() {
        assert!(f32q(-0.0).is_sign_negative());
        assert_ne!(f32q(-0.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn length3_f32_quantizes_sqrt() {
        let v = [0.1_f32 as f64, 0.2_f32 as f64, 0.3_f32 as f64];
        // Computed with the Python authority (_fmul/_fadd chain + _f32(sqrt)).
        assert_eq!(length3_f32(v), 0.37416577339172363_f64);
    }

    #[test]
    fn sum_f64_is_neumaier_not_naive() {
        // CPython 3.14.6: sum([1.0, 1e-8, 4e-8, 9e-8]) differs from a naive
        // left fold in the last ulp (bit patterns below are struct.pack('<d')).
        let values = [1.0, 1e-8, 4e-8, 9e-8];
        assert_eq!(sum_f64(&values).to_bits(), 0x3ff0_0000_2594_ba0c);
        let mut naive = 0.0_f64;
        for &x in &values {
            naive += x;
        }
        assert_eq!(naive.to_bits(), 0x3ff0_0000_2594_ba0b);
        // Two-element sums coincide with plain addition (compensation rounds
        // back to the correctly rounded sum).
        assert_eq!(sum_f64(&[0.1, 0.2]), 0.1 + 0.2);
        // Integer-0 start maps a leading -0.0 to +0.0 (CPython sum([-0.0])).
        assert!(sum_f64(&[-0.0]).is_sign_positive());
        assert!(sum_f64(&[-0.0, -0.0]).is_sign_positive());
    }

    #[test]
    fn dot3_sum_matches_cpython_sum_semantics() {
        // Signed zero: all three products are -0.0. CPython
        // sum([-0.0, -0.0, -0.0]) is +0.0 (accumulator starts at int 0);
        // the naive fold keeps -0.0.
        let a = [-0.0, -1.0, -1.0];
        let b = [1.0, 0.0, 0.0];
        assert!(dot3(a, b).is_sign_negative());
        assert_eq!(dot3_sum(a, b).to_bits(), 0.0_f64.to_bits());
        // Cancellation: CPython sum([1e16, 1.0, -1e16]) == 1.0 (Neumaier
        // recovers the small term); the naive fold loses it.
        let big = [1e16, 1.0, -1e16];
        let one = [1.0, 1.0, 1.0];
        assert_eq!(dot3(big, one), 0.0);
        assert_eq!(dot3_sum(big, one), 1.0);
        // Ordinary values agree with the fold.
        let c = [1.5, -2.0, 0.25];
        let d = [2.0, 0.5, 4.0];
        assert_eq!(dot3_sum(c, d), dot3(c, d));
    }

    #[test]
    fn length3_float64_uses_neumaier_square_sum() {
        // CPython 3.14.6: math.sqrt(sum(v*v for v in [1.0, 1e-8, 4e-8]))
        // (bit patterns are struct.pack('<d')).
        let values = [1.0, 1e-8, 4e-8];
        assert_eq!(length3(values, false).to_bits(), 0x3ff0_0000_0000_0004);
        // A naive fold of the squares lands one ulp lower.
        let naive =
            (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]).sqrt();
        assert_eq!(naive.to_bits(), 0x3ff0_0000_0000_0003);
    }

    #[test]
    fn min2_max2_match_cpython_ties_and_nan() {
        // CPython: min(0.0, -0.0) == 0.0; min(-0.0, 0.0) == -0.0.
        assert!(min2_f64(0.0, -0.0).is_sign_positive());
        assert!(min2_f64(-0.0, 0.0).is_sign_negative());
        // CPython: max(0.0, -0.0) == 0.0; max(-0.0, 0.0) == -0.0.
        assert!(max2_f64(0.0, -0.0).is_sign_positive());
        assert!(max2_f64(-0.0, 0.0).is_sign_negative());
        // CPython: min(nan, 1.0) is nan; min(1.0, nan) == 1.0.
        assert!(min2_f64(f64::NAN, 1.0).is_nan());
        assert_eq!(min2_f64(1.0, f64::NAN), 1.0);
        assert!(max2_f64(f64::NAN, 1.0).is_nan());
        assert_eq!(max2_f64(1.0, f64::NAN), 1.0);
    }

    #[test]
    fn cpython_hypot2_matches_python_bit_patterns() {
        // Constants computed with CPython 3.14.6 math.hypot (bit patterns
        // are struct.pack('<d')).
        assert_eq!(cpython_hypot2(1.5, 1.2).to_bits(), 0x3ffe_bc28_b859_9866);
        assert_eq!(cpython_hypot2(3.0, 4.0).to_bits(), 0x4014_0000_0000_0000);
        assert_eq!(cpython_hypot2(0.1, 0.2).to_bits(), 0x3fcc_9f25_c5bf_edda);
        assert_eq!(
            cpython_hypot2(1e-300, 1e-300).to_bits(),
            0x01ae_4e8d_1276_2225
        );
        assert_eq!(
            cpython_hypot2(1e300, -1e300).to_bits(),
            0x7e40_e4d5_0f99_b211
        );
        // Subnormal inputs take the DBL_MIN rescaling path.
        assert_eq!(
            cpython_hypot2(5e-324, 0.0).to_bits(),
            0x0000_0000_0000_0001
        );
        assert_eq!(
            cpython_hypot2(5e-324, 5e-324).to_bits(),
            0x0000_0000_0000_0001
        );
        assert_eq!(
            cpython_hypot2(f64::MIN_POSITIVE, 1.0).to_bits(),
            0x3ff0_0000_0000_0000
        );
        // Huge magnitudes saturate to +inf exactly like CPython.
        assert_eq!(
            cpython_hypot2(f64::MAX, f64::MAX).to_bits(),
            0x7ff0_0000_0000_0000
        );
        assert_eq!(
            cpython_hypot2(f64::INFINITY, 1.0).to_bits(),
            0x7ff0_0000_0000_0000
        );
        assert_eq!(cpython_hypot2(0.0, -0.0).to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(
            cpython_hypot2(
                f64::from_bits(0x3fd8_2eb4_60ad_c180),
                f64::from_bits(0x3fdf_10f5_ede9_8bfc),
            )
            .to_bits(),
            0x3fe3_af34_da25_9e62
        );
    }
}
