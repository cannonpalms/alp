//! Scores one exponent pair of the `f64` search two values at a time with NEON.

use super::vector::{Kernel, Scale, Seen, estimate_pair};
use super::{ALPFloat, Exponents};
use std::arch::aarch64::*;

/// Scores the pair like [`super::estimate_scaled`] does for `f64`.
///
/// NEON is part of every `aarch64` target this module is compiled for, so no detection is needed.
///
/// The slices stay memory-safe if their lengths differ, since scoring stops at the shorter, but
/// then the size means nothing.
pub(super) fn score_pair(
    values: &[f64],
    scaled: &[f64],
    exponents: Exponents,
    limit: usize,
    tie_loses: bool,
) -> Option<usize> {
    // SAFETY: this module is compiled only when NEON is enabled for the target.
    unsafe { estimate_pair::<Lanes>(values, scaled, exponents, limit, tie_loses) }
}

/// Running bounds in two lanes.
///
/// NEON converts between `f64` and `i64` directly, and `FCVTZS` truncates toward zero, saturates
/// and maps `NaN` to zero as `as i64` does, so `rounded as i64 as f64` is one instruction each
/// way. `i64::MAX` comes back as `2^63`, which orders the same, as [`super::vector`] explains.
///
/// Every bound starts as `NaN`, and `FMINNM` and `FMAXNM` return the other operand when one is
/// `NaN`. An exception passes `NaN` in place of its value, so it drops out of the kept bounds, and
/// a bound that saw nothing is still `NaN` once the lanes are folded.
struct Lanes {
    kept_min: float64x2_t,
    kept_max: float64x2_t,
    all_min: float64x2_t,
    all_max: float64x2_t,
    patches: uint64x2_t,
}

impl Kernel for Lanes {
    #[inline(always)]
    unsafe fn new() -> Self {
        // SAFETY: NEON is enabled for the target.
        unsafe {
            let nan = vdupq_n_f64(f64::NAN);
            Self {
                kept_min: nan,
                kept_max: nan,
                all_min: nan,
                all_max: nan,
                patches: vdupq_n_u64(0),
            }
        }
    }

    #[inline(always)]
    unsafe fn score(&mut self, values: &[f64; 8], scaled: &[f64; 8], scale: Scale) {
        // SAFETY: NEON is enabled for the target, and each array holds the four registers loaded.
        unsafe {
            let sweet = vdupq_n_f64(f64::SWEET);
            let inverse_f = vdupq_n_f64(scale.inverse_f);
            let power_f = vdupq_n_f64(scale.power_f);
            let inverse_e = vdupq_n_f64(scale.inverse_e);
            let ones = vdupq_n_u64(u64::MAX);
            for pair in 0..4 {
                let value = vld1q_f64(values.as_ptr().add(pair * 2));
                let scaled = vld1q_f64(scaled.as_ptr().add(pair * 2));
                let rounded = vsubq_f64(vaddq_f64(vmulq_f64(scaled, inverse_f), sweet), sweet);
                let encoded = vcvtq_f64_s64(vcvtq_s64_f64(rounded));
                let decoded = vmulq_f64(vmulq_f64(encoded, power_f), inverse_e);
                let kept = vceqq_u64(vreinterpretq_u64_f64(decoded), vreinterpretq_u64_f64(value));
                // All ones is a `NaN`, so a missed lane leaves the kept bounds alone.
                let kept_encoded =
                    vreinterpretq_f64_u64(vornq_u64(vreinterpretq_u64_f64(encoded), kept));
                self.kept_min = vminnmq_f64(self.kept_min, kept_encoded);
                self.kept_max = vmaxnmq_f64(self.kept_max, kept_encoded);
                self.all_min = vminnmq_f64(self.all_min, encoded);
                self.all_max = vmaxnmq_f64(self.all_max, encoded);
                // A missed lane is all ones, which is minus one.
                self.patches = vsubq_u64(self.patches, veorq_u64(kept, ones));
            }
        }
    }

    #[inline(always)]
    unsafe fn seen(&self) -> Seen {
        // SAFETY: NEON is enabled for the target.
        unsafe {
            let fold = |min: float64x2_t, max: float64x2_t| {
                let (min, max) = (vminnmvq_f64(min), vmaxnmvq_f64(max));
                (!min.is_nan()).then_some((min, max))
            };
            Seen {
                kept: fold(self.kept_min, self.kept_max),
                all: fold(self.all_min, self.all_max),
                patches: vaddvq_u64(self.patches) as usize,
            }
        }
    }
}
