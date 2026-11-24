//! ALP encoding benchmarks separating exponent search, allocation, and exception handling.
//!
//! The `alp_exponent_search_*` cases other than `rotating_*` search one input over and over. The
//! branch predictor learns that input's data-dependent branches, so those cases under-report the
//! cost for a caller that searches a new block each time. The `rotating_*` cases give every call a
//! different input from a pool of 128, so the predictor cannot memorize one input. The
//! `rotating_*_hinted` cases pass the exponents the search finds for that input as the hint.

#![allow(missing_docs)]

use alp::{ALPFloat, Exponents};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::mem::MaybeUninit;

const LEN: usize = 1024;
const POOL: usize = 128;
const SHORT_LEN: usize = 16;

/// A small linear congruential generator, to keep the benchmarks dependency-free and repeatable.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(0x517C_C1B7_2722_0A95 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn round_to(value: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    (value * scale).round() / scale
}

/// Pools of `POOL` inputs of `LEN` values, each pool of one kind, and one of `SHORT_LEN` values.
/// Within a pool the inputs differ in magnitude or in the number of decimals, so the best
/// exponents move from input to input.
fn pools<F: ALPFloat>() -> Vec<(&'static str, Vec<Vec<F>>)> {
    let convert = |values: Vec<f64>| -> Vec<F> {
        values
            .into_iter()
            .map(|v| F::from(v).expect("pool values fit both float types"))
            .collect()
    };
    let build = |make: &dyn Fn(usize, &mut Lcg) -> Vec<f64>| -> Vec<Vec<F>> {
        (0..POOL)
            .map(|k| convert(make(k, &mut Lcg::new(k as u64))))
            .collect()
    };
    let decimal = build(&|k, rng| {
        let decimals = 1 + (k % 4) as i32;
        let base = 10f64.powf(rng.next_unit() * 4.0);
        (0..LEN)
            .map(|i| {
                let wave = (i as f64 * 0.01).sin() * base * 0.05;
                round_to(base + wave + rng.next_unit() - 0.5, decimals)
            })
            .collect()
    });
    let integer = build(&|k, rng| {
        // ALP rounds an f32 exactly only below 2^22, and larger integers no longer encode with the
        // identity pair. The largest value here is below 1.01e6.
        let digits = if size_of::<F>() == 4 {
            (k % 9).min(3)
        } else {
            k % 9
        };
        let max = 10u64.pow(1 + digits as u32);
        let start = (rng.next_unit() * (max * 100) as f64) as u64;
        (0..LEN)
            .map(|_| (start + (rng.next_unit() * max as f64) as u64) as f64)
            .collect()
    });
    let full_precision = build(&|_, rng| {
        let offset = rng.next_unit() * 1000.0;
        (0..LEN).map(|_| offset + rng.next_unit()).collect()
    });
    let two_decimals = build(&|_, rng| {
        let base = 10f64.powf(rng.next_unit() * 4.0);
        (0..LEN)
            .map(|i| {
                let wave = (i as f64 * 0.01).sin() * base * 0.05;
                round_to(base + wave + rng.next_unit() - 0.5, 2)
            })
            .collect()
    });
    // Short blocks are where a vectorized scorer has the least to gain.
    let short_decimal = decimal
        .iter()
        .map(|input| input[..SHORT_LEN].to_vec())
        .collect();
    vec![
        ("rotating_decimal", decimal),
        ("rotating_decimal_16", short_decimal),
        ("rotating_integer", integer),
        ("rotating_full_precision", full_precision),
        ("rotating_two_decimals", two_decimals),
    ]
}

fn bench_float<F: ALPFloat>(c: &mut Criterion, name: &str) {
    let exponents = Exponents { e: 3, f: 0 };
    let values: Vec<F> = (0..LEN)
        .map(|i| {
            F::from((i * 7919) % 100000).expect("small integers fit both float types") * F::IF10[3]
        })
        .collect();
    let mut perturbed = values.clone();
    for value in perturbed.iter_mut().skip(64).step_by(32) {
        *value =
            *value + F::from(0.0000123456789).expect("finite perturbation fits both float types");
    }
    let mut dense = values.clone();
    for value in dense.iter_mut().step_by(2) {
        *value =
            *value + F::from(0.0000123456789).expect("finite perturbation fits both float types");
    }
    let all_patched = vec![F::nan(); LEN];
    let mut group = c.benchmark_group(format!("alp_encode_{name}"));
    group.throughput(Throughput::Elements(LEN as u64));
    for (case, input) in [
        ("clean", &values),
        ("patched", &perturbed),
        ("dense", &dense),
        ("all_patched", &all_patched),
    ] {
        let (_, encoded, positions, exceptions, _) = alp::encode::<F>(input, Some(exponents));
        assert_eq!(positions.is_empty(), case == "clean");
        let mut decoded = alp::decode::<F>(&encoded, exponents);
        for (&position, &value) in positions.iter().zip(&exceptions) {
            decoded[position as usize] = value;
        }
        assert!(
            decoded
                .iter()
                .zip(input)
                .all(|(&actual, &expected)| actual.is_eq(expected))
        );

        // The setup closure gives each input by reference. Criterion black-boxes the inputs and
        // drops the encoded vectors after the measurement stops.
        group.bench_function(format!("{case}/alloc"), |b| {
            b.iter_batched_ref(
                || (input.as_slice(), exponents),
                |(input, exponents)| alp::encode::<F>(input, Some(*exponents)),
                BatchSize::SmallInput,
            );
        });

        // The setup closure makes empty output buffers, so their allocation is not part of the
        // measurement.
        group.bench_function(format!("{case}/into"), |b| {
            b.iter_batched_ref(
                || {
                    (
                        input.as_slice(),
                        exponents,
                        vec![MaybeUninit::<F::ALPInt>::uninit(); LEN],
                        Vec::with_capacity(LEN),
                        Vec::with_capacity(LEN),
                        Vec::with_capacity(1),
                    )
                },
                |(input, exponents, encoded, positions, exceptions, offsets)| {
                    alp::encode_into::<F>(
                        input,
                        Some(*exponents),
                        encoded,
                        positions,
                        exceptions,
                        offsets,
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();

    let wide: Vec<F> = (0..LEN)
        .map(|i| {
            F::from((i % 61) as i32 - 30)
                .expect("small exponents fit both float types")
                .exp()
        })
        .collect();
    let mut group = c.benchmark_group(format!("alp_exponent_search_{name}"));
    group.throughput(Throughput::Elements(LEN as u64));
    for (case, input) in [
        ("clean", &values),
        ("patched", &perturbed),
        ("wide", &wide),
        ("all_patched", &all_patched),
    ] {
        group.bench_function(case, |b| {
            b.iter_batched_ref(
                || input.as_slice(),
                |input| alp::find_best_exponents(input),
                BatchSize::SmallInput,
            );
        });
    }
    // The setup closure picks the next input, so the choice is not measured.
    for (case, pool) in pools::<F>() {
        group.throughput(Throughput::Elements(pool[0].len() as u64));
        group.bench_function(case, |b| {
            let mut next = 0;
            b.iter_batched_ref(
                || {
                    next = (next + 1) % POOL;
                    pool[next].as_slice()
                },
                |input| alp::find_best_exponents(input),
                BatchSize::SmallInput,
            );
        });
        // The hint is what `find_best_exponents` returns for the same input, found outside the
        // measured region, as when a column's consecutive blocks share their exponents.
        let hints: Vec<Exponents> = pool
            .iter()
            .map(|input| alp::find_best_exponents(input))
            .collect();
        group.bench_function(format!("{case}_hinted"), |b| {
            let mut next = 0;
            b.iter_batched_ref(
                || {
                    next = (next + 1) % POOL;
                    (pool[next].as_slice(), hints[next])
                },
                |(input, hint)| alp::find_best_exponents_with_hint(input, *hint),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_float::<f64>(c, "f64");
    bench_float::<f32>(c, "f32");
}

criterion_group!(benches, bench);
criterion_main!(benches);
