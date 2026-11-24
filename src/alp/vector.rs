//! The parts of the vectorized `f64` pair scorer that do not depend on the instruction set.
//!
//! A kernel scores eight values at a time and keeps running bounds in registers. This module
//! drives it over a pair's sample, checks the bound after each chunk of eight, scores the values
//! past the last whole chunk one at a time, and turns the bounds into a size. Everything is
//! exact: the result of a pair is the one [`super::estimate_scaled`] computes.
//!
//! Encoded values are kept as `f64` holding the integer `rounded as i64` would give. Every such
//! integer below `2^63` is exact in `f64`, and `i64::MAX` becomes `2^63`, which orders the same as
//! `i64::MAX` and converts back to it, so a range held in `f64` has the bit width the `i64` range
//! has.

use super::{ALPFloat, Exponents, encoded_bytes_for_range, patch_bytes};

/// The multipliers of one pair.
#[derive(Clone, Copy)]
pub(super) struct Scale {
    pub(super) inverse_f: f64,
    pub(super) power_f: f64,
    pub(super) inverse_e: f64,
}

/// A range of encoded values, `None` until a value is seen.
type Range = Option<(f64, f64)>;

/// What the scoring of one pair has seen so far.
pub(super) struct Seen {
    /// The range of the values that round-trip.
    pub(super) kept: Range,
    /// The range of every value, round-tripping or not.
    pub(super) all: Range,
    pub(super) patches: usize,
}

fn widen(range: &mut Range, value: f64) {
    *range = Some(range.map_or((value, value), |(min, max)| {
        (min.min(value), max.max(value))
    }));
}

fn bytes(len: usize, range: Range) -> usize {
    encoded_bytes_for_range(len, range.map(|(min, max)| (min as i64, max as i64)))
}

impl Seen {
    /// The size the pair cannot come in under, whatever the values still to be scored are. The
    /// final range includes the kept values seen so far, and the patch count only grows.
    fn lower_bound(&self, len: usize) -> usize {
        let range = self.kept.map_or(0, |kept| bytes(len, Some(kept)));
        range + self.patches * patch_bytes::<f64>()
    }
}

/// Running bounds that score eight values per call.
///
/// # Safety
///
/// Every method is unsafe because a kernel runs instructions the CPU may lack. The caller must
/// have checked that it supports them, and must call from a function that enables them so the
/// methods inline into it.
pub(super) trait Kernel: Sized {
    /// Bounds that have seen nothing.
    unsafe fn new() -> Self;

    /// Scores the next eight values.
    unsafe fn score(&mut self, values: &[f64; 8], scaled: &[f64; 8], scale: Scale);

    /// The bounds so far, with the lanes folded together.
    unsafe fn seen(&self) -> Seen;
}

/// Scores the values after the last whole chunk of eight, one at a time, the way the scalar path
/// does, including the stop once the exceptions alone cost more than `limit`.
#[inline(always)]
fn score_tail(
    seen: &mut Seen,
    values: &[f64],
    scaled: &[f64],
    scale: Scale,
    limit: usize,
) -> Option<()> {
    for (&value, &scaled) in values.iter().zip(scaled) {
        let rounded = (scaled * scale.inverse_f).fast_round();
        let encoded = rounded.as_int() as f64;
        let decoded = encoded * scale.power_f * scale.inverse_e;
        widen(&mut seen.all, encoded);
        if decoded.is_eq(value) {
            widen(&mut seen.kept, encoded);
        } else {
            seen.patches += 1;
            if seen.patches * patch_bytes::<f64>() > limit {
                return None;
            }
        }
    }
    Some(())
}

/// Returns the pair's exact size, or `None` once a lower bound exceeds `limit`, or reaches it
/// when `tie_loses`.
///
/// The bound is checked after each chunk of eight values, not after the first four as the scalar
/// path does, and the values past the last chunk stop the pair only through the exceptions. So a
/// pair can be stopped here and not by the scalar path, for example eight `NaN`s and then `1.0`
/// at limit 80 with `tie_loses`, or the other way round. What both decide the same way is whether
/// the pair can beat `limit`: a pair stopped here cannot, and a pair that comes back with its
/// exact size exceeds `limit` (or equals it when `tie_loses`) if it cannot, so the caller
/// discards it just the same.
///
/// The slices stay memory-safe if their lengths differ, since scoring stops at the shorter, but
/// then the size means nothing.
///
/// # Safety
///
/// As for [`Kernel`].
#[inline(always)]
pub(super) unsafe fn estimate_pair<K: Kernel>(
    values: &[f64],
    scaled: &[f64],
    exponents: Exponents,
    limit: usize,
    tie_loses: bool,
) -> Option<usize> {
    let len = values.len();
    let scale = Scale {
        inverse_f: f64::IF10[exponents.f as usize],
        power_f: f64::F10[exponents.f as usize],
        inverse_e: f64::IF10[exponents.e as usize],
    };
    // SAFETY: the caller upholds the kernel's contract.
    let mut kernel = unsafe { K::new() };

    let (chunks, remainder) = values.as_chunks::<8>();
    let (scaled_chunks, scaled_remainder) = scaled.as_chunks::<8>();
    for (chunk, scaled_chunk) in chunks.iter().zip(scaled_chunks) {
        // SAFETY: as above.
        unsafe { kernel.score(chunk, scaled_chunk, scale) };
        if limit != usize::MAX {
            // SAFETY: as above.
            let bound = unsafe { kernel.seen() }.lower_bound(len);
            if bound > limit || (tie_loses && bound == limit) {
                return None;
            }
        }
    }
    // SAFETY: as above.
    let mut seen = unsafe { kernel.seen() };
    score_tail(&mut seen, remainder, scaled_remainder, scale, limit)?;

    // `encode` fills patched slots within the kept range, unless every value is patched, in
    // which case it is the range of all of them.
    let range = if seen.patches == len {
        seen.all
    } else {
        seen.kept
    };
    Some(bytes(len, range) + seen.patches * patch_bytes::<f64>())
}
