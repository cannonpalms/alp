//! Scores one exponent pair of the `f64` search four values at a time with AVX2.

use super::vector::{Kernel, Scale, Seen, estimate_pair};
use super::{ALPFloat, Exponents};
use std::arch::x86_64::*;

/// `2^63`, the first magnitude `i64` cannot hold.
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// Scores the pair like [`super::estimate_scaled`] does for `f64`, with the whole per-pair loop in
/// one function that has AVX2 enabled, so that the helpers below inline into it.
///
/// The slices stay memory-safe if their lengths differ, since scoring stops at the shorter, but
/// then the size means nothing.
///
/// Only `private::Sealed::pair_scorer` calls this, and only after it has found that the CPU
/// supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) fn score_pair(
    values: &[f64],
    scaled: &[f64],
    exponents: Exponents,
    limit: usize,
    tie_loses: bool,
) -> Option<usize> {
    // SAFETY: AVX2 is enabled for this function.
    unsafe { estimate_pair::<Lanes>(values, scaled, exponents, limit, tie_loses) }
}

/// Running bounds in four lanes.
///
/// AVX2 has no packed conversion between `f64` and `i64`, so the encoded values stay `f64` and
/// the conversion `rounded as i64 as f64` is reproduced exactly:
///
/// - Clamping to `+-2^63` matches the saturation of `as i64`. `i64::MAX` comes back as `2^63`,
///   which orders the same, so bounds held in `f64` give the bit width the `i64` bounds give.
/// - Truncating toward zero is needed even though the value was rounded to an integer, because
///   rounding by the sweet number leaves a fraction on negative values between `-2^52` and
///   `-2^51`, where the floats are spaced by `0.5`, and `as i64` truncates it.
/// - No `-0.0` reaches the bounds, though the integer zero converts back to positive zero:
///   `(x + SWEET) - SWEET` is never `-0.0`, nor a value in `(-1, 0)` that truncation would turn
///   into `-0.0`.
/// - `NaN` becomes `0.0`, as `NaN as i64` is zero. Without this the clamp, whose result for a
///   `NaN` operand is its second operand, would give `-2^63`.
///
/// The kept bounds start at `+-infinity`, which no encoded value reaches. An exception passes
/// `NaN` instead of its value, and `VMINPD` and `VMAXPD` return their second operand when either
/// is `NaN`, so with the running bound as the second operand the exception drops out.
struct Lanes {
    kept_min: __m256d,
    kept_max: __m256d,
    all_min: __m256d,
    all_max: __m256d,
    patches: __m256i,
}

impl Kernel for Lanes {
    #[inline(always)]
    unsafe fn new() -> Self {
        // SAFETY: the caller has AVX2 enabled.
        unsafe {
            Self {
                kept_min: _mm256_set1_pd(f64::INFINITY),
                kept_max: _mm256_set1_pd(f64::NEG_INFINITY),
                all_min: _mm256_set1_pd(f64::INFINITY),
                all_max: _mm256_set1_pd(f64::NEG_INFINITY),
                patches: _mm256_setzero_si256(),
            }
        }
    }

    #[inline(always)]
    unsafe fn score(&mut self, values: &[f64; 8], scaled: &[f64; 8], scale: Scale) {
        // SAFETY: the caller has AVX2 enabled, and each array holds the two registers loaded.
        unsafe {
            let sweet = _mm256_set1_pd(f64::SWEET);
            let inverse_f = _mm256_set1_pd(scale.inverse_f);
            let power_f = _mm256_set1_pd(scale.power_f);
            let inverse_e = _mm256_set1_pd(scale.inverse_e);
            let low = _mm256_set1_pd(-TWO_POW_63);
            let high = _mm256_set1_pd(TWO_POW_63);
            let ones = _mm256_set1_epi64x(-1);
            for half in 0..2 {
                let value = _mm256_loadu_pd(values.as_ptr().add(half * 4));
                let scaled = _mm256_loadu_pd(scaled.as_ptr().add(half * 4));
                let rounded = _mm256_sub_pd(
                    _mm256_add_pd(_mm256_mul_pd(scaled, inverse_f), sweet),
                    sweet,
                );
                let ordered = _mm256_cmp_pd::<_CMP_ORD_Q>(rounded, rounded);
                let clamped = _mm256_min_pd(_mm256_max_pd(rounded, low), high);
                let truncated =
                    _mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(clamped);
                let encoded = _mm256_and_pd(truncated, ordered);
                let decoded = _mm256_mul_pd(_mm256_mul_pd(encoded, power_f), inverse_e);
                let kept =
                    _mm256_cmpeq_epi64(_mm256_castpd_si256(decoded), _mm256_castpd_si256(value));
                let missed = _mm256_xor_si256(kept, ones);
                let kept_encoded =
                    _mm256_castsi256_pd(_mm256_or_si256(_mm256_castpd_si256(encoded), missed));
                self.kept_min = _mm256_min_pd(kept_encoded, self.kept_min);
                self.kept_max = _mm256_max_pd(kept_encoded, self.kept_max);
                self.all_min = _mm256_min_pd(encoded, self.all_min);
                self.all_max = _mm256_max_pd(encoded, self.all_max);
                // A missed lane is all ones, which is minus one.
                self.patches = _mm256_sub_epi64(self.patches, missed);
            }
        }
    }

    #[inline(always)]
    unsafe fn seen(&self) -> Seen {
        // SAFETY: the caller has AVX2 enabled.
        unsafe {
            let fold = |min: __m256d, max: __m256d| {
                let min = _mm_min_pd(_mm256_castpd256_pd128(min), _mm256_extractf128_pd::<1>(min));
                let max = _mm_max_pd(_mm256_castpd256_pd128(max), _mm256_extractf128_pd::<1>(max));
                let min = _mm_cvtsd_f64(_mm_min_sd(min, _mm_unpackhi_pd(min, min)));
                let max = _mm_cvtsd_f64(_mm_max_sd(max, _mm_unpackhi_pd(max, max)));
                // A bound that saw nothing is still at its infinity, and then min exceeds max.
                (min <= max).then_some((min, max))
            };
            let pairs = _mm_add_epi64(
                _mm256_castsi256_si128(self.patches),
                _mm256_extracti128_si256::<1>(self.patches),
            );
            let patches = _mm_cvtsi128_si64(_mm_add_epi64(pairs, _mm_unpackhi_epi64(pairs, pairs)));
            Seen {
                kept: fold(self.kept_min, self.kept_max),
                all: fold(self.all_min, self.all_max),
                patches: patches as usize,
            }
        }
    }
}
