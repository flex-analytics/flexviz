/// Microbenchmarks for the `arg_min_max` kernel internals.
///
/// Run with:
///   cargo bench --bench bench_argminmax
///
/// Benchmarks:
/// 1. SIMD (argminmax crate) vs scalar fallback — validates the SIMD speedup
///    for the common case (f64, 5M rows, 500 buckets).
/// 2. Bucket count scaling — measures how cost grows with n_buckets (1 → 1000)
///    on a fixed 5M-row dataset. Confirms single-pass scaling (cost ∝ n_buckets
///    only in partitioning overhead, not data scans).
/// 3. Data size scaling — measures 1M / 5M / 10M rows at fixed n_buckets=500.
///    Confirms linear O(n) growth as expected for a memory-bandwidth-bound kernel.
use argminmax::ArgMinMax;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Helpers (mirrors expressions.rs internals)
// ---------------------------------------------------------------------------

fn uniform_offsets(len: usize, n_out: usize) -> Vec<(usize, usize)> {
    let mut offsets = Vec::with_capacity(n_out);
    let base = len / n_out;
    let remainder = len % n_out;
    let mut start = 0usize;
    for i in 0..n_out {
        let window_len = base + usize::from(i < remainder);
        offsets.push((start, window_len));
        start += window_len;
    }
    offsets
}

/// SIMD path using the `argminmax` crate — matches `simd_argminmax` in expressions.rs.
fn simd_path(values: &[f64], offsets: &[(usize, usize)]) -> Vec<u32> {
    let pairs: Vec<(Option<u64>, Option<u64>)> = offsets
        .par_iter()
        .map(|(start, len)| {
            if *len == 0 {
                return (None, None);
            }
            let slice = &values[*start..*start + *len];
            let (min_i, max_i) = slice.argminmax();
            (Some((*start + min_i) as u64), Some((*start + max_i) as u64))
        })
        .collect();
    let (mins, maxs): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let mut indices: Vec<u32> = mins
        .into_iter()
        .chain(maxs)
        .flatten()
        .map(|i| i as u32)
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Serial SIMD path — mirrors `argminmax_contiguous` in expressions.rs. That is
/// what each `par_by_window` worker runs, and what the whole kernel runs below
/// the MIN_PAR threshold, so its cost is the per-worker floor. `simd_path` above
/// is the parallel shape; keeping both makes the gap between them visible.
fn simd_serial_path(values: &[f64], offsets: &[(usize, usize)]) -> Vec<u32> {
    let mut min_indices = Vec::with_capacity(offsets.len());
    let mut max_indices = Vec::with_capacity(offsets.len());
    for &(start, len) in offsets {
        if len == 0 {
            min_indices.push(None);
            max_indices.push(None);
            continue;
        }
        let slice = &values[start..start + len];
        let (min_i, max_i) = slice.argminmax();
        min_indices.push(Some((start + min_i) as u64));
        max_indices.push(Some((start + max_i) as u64));
    }
    let mut indices: Vec<u32> = min_indices
        .into_iter()
        .chain(max_indices)
        .flatten()
        .map(|i| i as u32)
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Scalar fallback using manual PartialOrd comparisons (previous implementation).
fn scalar_path(values: &[f64], offsets: &[(usize, usize)]) -> Vec<u32> {
    let pairs: Vec<(Option<u64>, Option<u64>)> = offsets
        .par_iter()
        .map(|(start, len)| {
            if *len == 0 {
                return (None, None);
            }
            let mut min_idx = 0usize;
            let mut max_idx = 0usize;
            let mut min_val = values[*start];
            let mut max_val = values[*start];
            for i in 1..*len {
                let v = values[*start + i];
                if v < min_val {
                    min_idx = i;
                    min_val = v;
                }
                if v > max_val {
                    max_idx = i;
                    max_val = v;
                }
            }
            (
                Some((*start + min_idx) as u64),
                Some((*start + max_idx) as u64),
            )
        })
        .collect();
    let (mins, maxs): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let mut indices: Vec<u32> = mins
        .into_iter()
        .chain(maxs)
        .flatten()
        .map(|i| i as u32)
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

// ---------------------------------------------------------------------------
// 1. SIMD vs scalar (fixed 5M rows / 500 buckets)
// ---------------------------------------------------------------------------

fn bench_simd_vs_scalar(c: &mut Criterion) {
    const N_ROWS: usize = 5_000_000;
    const N_BUCKETS: usize = 500; // n_points=1000 → n_buckets = n_points/2 = 500

    let data: Vec<f64> = (0..N_ROWS).map(|i| (i % 1000) as f64 + 0.5).collect();
    let offsets = uniform_offsets(N_ROWS, N_BUCKETS);

    let mut group = c.benchmark_group("simd_vs_scalar");

    group.bench_with_input(
        BenchmarkId::new("simd (argminmax crate)", N_ROWS),
        &(&data, &offsets),
        |b, (data, offsets)| b.iter(|| simd_path(black_box(data), black_box(offsets))),
    );

    group.bench_with_input(
        BenchmarkId::new("simd serial (shipped)", N_ROWS),
        &(&data, &offsets),
        |b, (data, offsets)| b.iter(|| simd_serial_path(black_box(data), black_box(offsets))),
    );

    group.bench_with_input(
        BenchmarkId::new("scalar (previous impl)", N_ROWS),
        &(&data, &offsets),
        |b, (data, offsets)| b.iter(|| scalar_path(black_box(data), black_box(offsets))),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Bucket count scaling (fixed 5M rows, varying n_buckets)
//    n_buckets = n_points // 2, so n_points = 2, 20, 200, 1000, 2000
// ---------------------------------------------------------------------------

fn bench_bucket_scaling(c: &mut Criterion) {
    const N_ROWS: usize = 5_000_000;
    let data: Vec<f64> = (0..N_ROWS).map(|i| (i % 1000) as f64 + 0.5).collect();

    let mut group = c.benchmark_group("bucket_scaling");

    for n_buckets in [1usize, 10, 100, 500, 1000] {
        let offsets = uniform_offsets(N_ROWS, n_buckets);
        group.bench_with_input(
            BenchmarkId::new("simd", n_buckets),
            &(&data, &offsets),
            |b, (data, offsets)| b.iter(|| simd_path(black_box(data), black_box(offsets))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Data size scaling (fixed n_buckets=500, varying N)
//    Validates that cost is O(n) — memory-bandwidth-bound.
// ---------------------------------------------------------------------------

fn bench_size_scaling(c: &mut Criterion) {
    const N_BUCKETS: usize = 500;

    let mut group = c.benchmark_group("size_scaling");

    for n_rows in [1_000_000usize, 5_000_000, 10_000_000] {
        let data: Vec<f64> = (0..n_rows).map(|i| (i % 1000) as f64 + 0.5).collect();
        let offsets = uniform_offsets(n_rows, N_BUCKETS);
        group.bench_with_input(
            BenchmarkId::new("simd", n_rows),
            &(&data, &offsets),
            |b, (data, offsets)| b.iter(|| simd_path(black_box(data), black_box(offsets))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_vs_scalar,
    bench_bucket_scaling,
    bench_size_scaling
);
criterion_main!(benches);
