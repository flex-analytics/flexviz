#![allow(clippy::unused_unit)]
use std::mem::{align_of, size_of};
use std::sync::OnceLock;

use argminmax::ArgMinMax;
use half::f16;
use polars::prelude::*;
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// The rayon pool the parallel kernels run on.
///
/// A cdylib plugin statically links its own `polars-core`, so its `THREAD_POOL`
/// is a *different* static from the one inside Polars' extension module and the
/// two cannot be shared (pola-rs/polars#19650, still open). A second pool is
/// therefore unavoidable — the only thing under our control is its size.
///
/// rayon's global pool sizes itself from `available_parallelism()`, which knows
/// nothing about `POLARS_MAX_THREADS`. A deployment that limits Polars to N
/// threads would otherwise still get N + `available_parallelism()` threads here,
/// which on a CPU-quota'd container is exactly the oversubscription that causes
/// the whole cgroup to be throttled. So take the same knob Polars takes.
fn kernel_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("POLARS_MAX_THREADS")
            .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("flexviz-kernel-{i}"))
            .build()
            .expect("failed to build the flexviz kernel thread pool")
    })
}

// ---------------------------------------------------------------------------
// every_nth
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EveryNthKwargs {
    n_points: usize,
}

fn every_nth_output(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(inputs[0].clone())
}

#[polars_expr(output_type_func = every_nth_output)]
fn every_nth(inputs: &[Series], kwargs: EveryNthKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.n_points > 0,
        InvalidOperation: "n_points must be greater than 0"
    );

    let s = &inputs[0];
    if s.is_empty() {
        return Ok(s.clone());
    }

    let len = s.len();
    let stride = (len / kwargs.n_points).max(1);

    // Compute ceil(len / stride) but cap at n_points to guarantee at most n_points output.
    let n_out = len.div_ceil(stride).min(kwargs.n_points);

    let indices: Vec<u32> = (0..n_out).map(|i| (i * stride) as u32).collect();
    let idx_ca = UInt32Chunked::from_iter_values("".into(), indices.into_iter());
    s.take(&idx_ca)
}

// ---------------------------------------------------------------------------
// arg_min_max
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ArgMinMaxKwargs {
    n_points: usize,
}

/// One `x_domain` bound.
///
/// Integer x is searched on exact integers: a nanosecond epoch (~1.7e18) has no
/// exact f64, so an f64 edge collapses neighbouring rows into one bucket. The
/// kwargs travel as a pickle, which keeps a Python `int` and a Python `float`
/// distinct, so an untagged enum reads back the bound the caller sent.
#[derive(Deserialize, Clone, Copy)]
#[serde(untagged)]
enum XBound {
    Int(i64),
    Float(f64),
}

impl XBound {
    fn as_f64(self) -> f64 {
        match self {
            XBound::Int(v) => v as f64,
            XBound::Float(v) => v,
        }
    }
}

/// Whether `[lo, hi]` spans anything, compared without an f64 round trip so
/// that two integer bounds one apart never look equal.
fn positive_span(lo: XBound, hi: XBound) -> bool {
    match (lo, hi) {
        (XBound::Int(lo), XBound::Int(hi)) => hi > lo,
        _ => hi.as_f64() > lo.as_f64(),
    }
}

#[derive(Deserialize)]
struct MinmaxLineKwargs {
    n_points: usize,
    x_name: Option<String>,
    y_name: Option<String>,
    /// `Some((lo, hi))` switches the buckets from equal row count to equal x
    /// width over `[lo, hi]`, which needs x sorted ascending. `None` keeps the
    /// row-count buckets.
    #[serde(default)]
    x_domain: Option<(XBound, XBound)>,
}

// ---------------------------------------------------------------------------
// minmax_pairs_line
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MinmaxPairsKwargs {
    n_buckets: usize,
    /// Same meaning as in `MinmaxLineKwargs`: `Some((lo, hi))` switches to
    /// equal-x-width buckets, `None` keeps the row-count buckets.
    #[serde(default)]
    x_domain: Option<(XBound, XBound)>,
}

// ---------------------------------------------------------------------------
// fixed_hist
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct FixedHistKwargs {
    n_bins: usize,
}

const FIXED_HIST_ROUND_EPS: f64 = 1e-9;

/// Span pad shared by every 2D binner (`fixed_hist2d`, its rayon twin, and
/// `fixed_hist2d_reduce`) so a value exactly at `hi` lands in the top bin.
/// One definition on purpose: the scalar and parallel counts must be identical.
const FIXED_HIST2D_SPAN_EPS: f64 = 1e-10;

trait FixedHistValue: Copy {
    const CHECK_NAN: bool;

    fn to_hist_f64(self) -> f64;
}

trait ArgMinMaxValue: Copy + PartialOrd {
    fn is_nan(self) -> bool {
        false
    }
}

macro_rules! impl_fixed_hist_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FixedHistValue for $ty {
                const CHECK_NAN: bool = false;

                #[inline(always)]
                fn to_hist_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_fixed_hist_value!(i64, i32, i16, i8, u64, u32, u16, u8);

macro_rules! impl_argminmax_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ArgMinMaxValue for $ty {}
        )*
    };
}

impl_argminmax_value!(i64, i32, i16, i8, u64, u32, u16, u8);

impl ArgMinMaxValue for f64 {
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl ArgMinMaxValue for f32 {
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl ArgMinMaxValue for f16 {
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl FixedHistValue for f64 {
    const CHECK_NAN: bool = true;

    #[inline(always)]
    fn to_hist_f64(self) -> f64 {
        self
    }
}

impl FixedHistValue for f32 {
    const CHECK_NAN: bool = true;

    #[inline(always)]
    fn to_hist_f64(self) -> f64 {
        self as f64
    }
}

impl FixedHistValue for pf16 {
    const CHECK_NAN: bool = true;

    #[inline(always)]
    fn to_hist_f64(self) -> f64 {
        self.into()
    }
}

fn arg_min_max_output(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(inputs[0].name().clone(), DataType::UInt32))
}

#[polars_expr(output_type_func = arg_min_max_output)]
fn arg_min_max(inputs: &[Series], kwargs: ArgMinMaxKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.n_points > 0,
        InvalidOperation: "n_points must be greater than 0"
    );

    let s = &inputs[0];
    let indices = arg_min_max_indices(s, kwargs.n_points)?;
    Ok(UInt32Chunked::from_iter_values(s.name().clone(), indices.into_iter()).into_series())
}

fn minmax_line_output(inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![inputs[0].clone(), inputs[1].clone()]),
    ))
}

/// `arg_min_max` plus both gathers in one call. Exists because Polars does not
/// CSE opaque plugin expressions: `x.gather(idx)` / `y.gather(idx)` on the same
/// `arg_min_max` expression runs the whole scan twice.
#[polars_expr(output_type_func = minmax_line_output)]
fn minmax_line(inputs: &[Series], kwargs: MinmaxLineKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.n_points > 0,
        InvalidOperation: "n_points must be greater than 0"
    );

    let x = &inputs[0];
    let y = &inputs[1];
    polars_ensure!(
        x.len() == y.len(),
        ShapeMismatch: "minmax_line: x and y must have the same length"
    );

    let indices = match kwargs.x_domain {
        Some((lo, hi)) => arg_min_max_indices_x_width(x, y, kwargs.n_points, lo, hi)?,
        None => arg_min_max_indices(y, kwargs.n_points)?,
    };
    let idx_ca = UInt32Chunked::from_iter_values(PlSmallStr::EMPTY, indices.into_iter());
    let mut x_taken = x.take(&idx_ca)?;
    let mut y_taken = y.take(&idx_ca)?;
    let x_name = output_name(kwargs.x_name, x.name(), "x");
    let y_name = output_name(kwargs.y_name, y.name(), "y");
    x_taken.rename(x_name);
    y_taken.rename(y_name);
    let len = x_taken.len();

    let out =
        StructChunked::from_series("minmax_line".into(), len, [&x_taken, &y_taken].into_iter())?;
    Ok(out.into_series())
}

fn minmax_pairs_line_output(inputs: &[Field]) -> PolarsResult<Field> {
    let x = inputs[0].dtype().clone();
    let y = inputs[1].dtype().clone();
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new("x_min".into(), x.clone()),
            Field::new("y_min".into(), y.clone()),
            Field::new("x_max".into(), x),
            Field::new("y_max".into(), y),
        ]),
    ))
}

/// One `(x, y)` pair per non-empty bucket: the argmin row and the argmax row of
/// y, both with their x. `minmax_line` flattens and dedups the same buckets;
/// this keeps the pairing, which the caller's compensation walk needs.
#[polars_expr(output_type_func = minmax_pairs_line_output)]
fn minmax_pairs_line(inputs: &[Series], kwargs: MinmaxPairsKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.n_buckets > 0,
        InvalidOperation: "n_buckets must be greater than 0"
    );

    let x = &inputs[0];
    let y = &inputs[1];
    polars_ensure!(
        x.len() == y.len(),
        ShapeMismatch: "minmax_pairs_line: x and y must have the same length"
    );
    polars_ensure!(
        y.len() <= u32::MAX as usize,
        InvalidOperation: "series length exceeds UInt32 index capacity"
    );

    let pairs = if y.is_empty() {
        Vec::new()
    } else {
        match kwargs.x_domain {
            Some((lo, hi)) if !positive_span(lo, hi) => Vec::new(),
            // The bucket count is not clamped to the row count here: it divides
            // `[lo, hi]`, so a smaller count would widen the buckets and build a
            // different grid than the caller's (see `arg_min_max_indices_x_width`).
            Some((lo, hi)) => {
                let offsets = x_width_offsets(x, lo, hi, kwargs.n_buckets)?;
                pairs_for_offsets(y, &offsets)
            },
            None => {
                let offsets = uniform_offsets(y.len(), kwargs.n_buckets.min(y.len()).max(1));
                pairs_for_offsets(y, &offsets)
            },
        }
    };

    let min_idx = UInt32Chunked::from_iter_values(PlSmallStr::EMPTY, pairs.iter().map(|p| p.0));
    let max_idx = UInt32Chunked::from_iter_values(PlSmallStr::EMPTY, pairs.iter().map(|p| p.1));
    let mut fields = [
        x.take(&min_idx)?,
        y.take(&min_idx)?,
        x.take(&max_idx)?,
        y.take(&max_idx)?,
    ];
    for (field, name) in fields.iter_mut().zip(["x_min", "y_min", "x_max", "y_max"]) {
        field.rename(name.into());
    }

    let out = StructChunked::from_series("minmax_pairs_line".into(), pairs.len(), fields.iter())?;
    Ok(out.into_series())
}

// ---------------------------------------------------------------------------
// Internal helpers (shared by the min-max kernels)
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

fn arg_min_max_indices(s: &Series, n_points: usize) -> PolarsResult<Vec<u32>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }

    let pairs = arg_min_max_pairs(s, (n_points / 2).max(1))?;
    let mut indices: Vec<u32> = pairs
        .into_iter()
        .flat_map(|(arg_min, arg_max)| [arg_min, arg_max])
        .collect();

    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn arg_min_max_pairs(s: &Series, n_buckets: usize) -> PolarsResult<Vec<(u32, u32)>> {
    polars_ensure!(
        s.len() <= u32::MAX as usize,
        InvalidOperation: "series length exceeds UInt32 index capacity"
    );

    if s.is_empty() || n_buckets == 0 {
        return Ok(Vec::new());
    }

    let offsets = uniform_offsets(s.len(), n_buckets.min(s.len()).max(1));
    Ok(pairs_for_offsets(s, &offsets))
}

fn pairs_for_offsets(s: &Series, offsets: &[(usize, usize)]) -> Vec<(u32, u32)> {
    // The fallback is the null path, and nulls are common in real signals: one null
    // in 20M rows drops the whole column here, costing ~12x. It splits by window
    // like the SIMD paths — each entry carries an absolute `start`, so no cursor to
    // seed — so it gets the same treatment for free.
    let (arg_min_vec, arg_max_vec) = simd_argminmax(s, offsets)
        .unwrap_or_else(|| par_by_window(offsets, |o| fallback_window_argminmax(s, o)));

    arg_min_vec
        .into_iter()
        .zip(arg_max_vec)
        .filter_map(|(arg_min, arg_max)| Some((arg_min? as u32, arg_max? as u32)))
        .collect()
}

/// `arg_min_max_indices` over equal-x-width buckets instead of equal-row-count
/// ones. x must be sorted ascending and null-free. Rows outside `[lo, hi]` are
/// dropped.
fn arg_min_max_indices_x_width(
    x: &Series,
    y: &Series,
    n_points: usize,
    lo: XBound,
    hi: XBound,
) -> PolarsResult<Vec<u32>> {
    polars_ensure!(
        y.len() <= u32::MAX as usize,
        InvalidOperation: "series length exceeds UInt32 index capacity"
    );
    if y.is_empty() || !positive_span(lo, hi) {
        return Ok(Vec::new());
    }

    // The bucket count may not be clamped to the row count the way
    // `arg_min_max_pairs` clamps it: it divides `[lo, hi]`, so a smaller count
    // would widen the buckets and rebuild a different grid than the caller's.
    // Only the output allocation is bounded by the rows (`bucket_offsets`).
    let offsets = x_width_offsets(x, lo, hi, (n_points / 2).max(1))?;
    let mut indices: Vec<u32> = pairs_for_offsets(y, &offsets)
        .into_iter()
        .flat_map(|(arg_min, arg_max)| [arg_min, arg_max])
        .collect();

    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

/// Non-empty `(start, len)` windows for `n_buckets` equal-width x buckets.
///
/// Reads x through its physical representation, so a temporal column is an
/// Int64 view with no copy or cast. An integer column searches integer edges in
/// i128; only a float column goes through f64. Nulls in x are read as their
/// physical value, not skipped, and a NaN breaks the edge search. Callers
/// already require an x free of both.
fn x_width_offsets(
    x: &Series,
    lo: XBound,
    hi: XBound,
    n_buckets: usize,
) -> PolarsResult<Vec<(usize, usize)>> {
    let phys = x.to_physical_repr();

    macro_rules! int_dispatch {
        ($($downcast:ident),* $(,)?) => {
            $(
                if let Ok(ca) = phys.$downcast() {
                    let (lo_i, hi_i) = int_bounds(lo, hi, x.dtype())?;
                    // Ceiling division, so the kernel rebuilds the same width
                    // as `_bucket_grid`. Truncating would narrow it and open
                    // more buckets than the budget. Span and count are positive
                    // (`positive_span`), so add-then-floor is the ceiling.
                    let n = n_buckets as i128;
                    let width = (hi_i - lo_i + n - 1) / n;
                    return Ok(chunked_bucket_offsets(
                        ca,
                        n_buckets,
                        |v| v as i128,
                        |i| {
                            if i == n_buckets {
                                hi_i
                            } else {
                                lo_i + i as i128 * width
                            }
                        },
                    ));
                }
            )*
        };
    }

    macro_rules! float_dispatch {
        ($($downcast:ident),* $(,)?) => {
            $(
                if let Ok(ca) = phys.$downcast() {
                    let (lo_f, hi_f) = (lo.as_f64(), hi.as_f64());
                    let width = (hi_f - lo_f) / n_buckets as f64;
                    return Ok(chunked_bucket_offsets(
                        ca,
                        n_buckets,
                        |v| v.to_hist_f64(),
                        |i| {
                            if i == n_buckets {
                                hi_f
                            } else {
                                lo_f + i as f64 * width
                            }
                        },
                    ));
                }
            )*
        };
    }

    int_dispatch!(i64, i32, i16, i8, u64, u32, u16, u8);
    float_dispatch!(f64, f32, f16);

    polars_bail!(
        InvalidOperation:
        "x_domain is only supported for numeric or temporal x, got {}",
        x.dtype()
    )
}

/// The `[lo, hi]` bounds of an integer x, which must have arrived as integers,
/// so an integer edge search never round-trips through f64.
fn int_bounds(lo: XBound, hi: XBound, dtype: &DataType) -> PolarsResult<(i128, i128)> {
    match (lo, hi) {
        (XBound::Int(lo), XBound::Int(hi)) => Ok((lo as i128, hi as i128)),
        _ => polars_bail!(
            InvalidOperation:
            "x_domain must be integer bounds for an x of type {}, got a float bound",
            dtype
        ),
    }
}

/// `bucket_offsets` over a chunked x, searched without a per-element `get`.
///
/// `ChunkedArray::get` resolves the chunk by walking the chunk lengths, so a
/// binary search over it costs O(chunks) per access: measurably ~10x slower per
/// edge on a 1000-chunk Parquet read. Searching the per-chunk last values first
/// and then one contiguous slice keeps every edge at O(log chunks + log len).
fn chunked_bucket_offsets<T, K>(
    ca: &ChunkedArray<T>,
    n_buckets: usize,
    to_key: impl Fn(T::Native) -> K,
    edge: impl Fn(usize) -> K,
) -> Vec<(usize, usize)>
where
    T: PolarsNumericType,
    K: Copy + PartialOrd,
{
    let chunks: Vec<&[T::Native]> = ca
        .downcast_iter()
        .map(|arr| arr.values().as_slice())
        .collect();
    let mut starts = Vec::with_capacity(chunks.len() + 1);
    let mut chunk_last: Vec<Option<K>> = Vec::with_capacity(chunks.len());
    let mut offset = 0usize;
    let mut last: Option<K> = None;
    for chunk in &chunks {
        starts.push(offset);
        offset += chunk.len();
        // An empty chunk keeps its predecessor's last value, so `chunk_last`
        // stays sorted and the empty chunk is never the first one to pass.
        if let Some(v) = chunk.last() {
            last = Some(to_key(*v));
        }
        chunk_last.push(last);
    }
    let len = offset;

    // First index whose x passes `edge`: `x >= edge`, or `x > edge` when strict.
    let search = |edge: K, strict: bool| {
        let fails = |v: K| !if strict { v > edge } else { v >= edge };
        let chunk_idx = chunk_last.partition_point(|&v| match v {
            Some(v) => fails(v),
            None => true,
        });
        match chunks.get(chunk_idx) {
            Some(chunk) => starts[chunk_idx] + chunk.partition_point(|v| fails(to_key(*v))),
            None => len,
        }
    };

    bucket_offsets(n_buckets, len, edge, search)
}

/// `(start, len)` of every non-empty bucket, given its edges and a search.
///
/// Generic over the edge type so an integer x can search exact i128 edges while
/// a float x searches f64 ones. `n_rows` bounds the allocation: every bucket
/// kept holds at least one row, so more buckets than rows cannot all be kept.
fn bucket_offsets<E, FE, FS>(
    n_buckets: usize,
    n_rows: usize,
    edge: FE,
    search: FS,
) -> Vec<(usize, usize)>
where
    E: Copy,
    FE: Fn(usize) -> E,
    FS: Fn(E, bool) -> usize,
{
    let mut offsets = Vec::with_capacity(n_buckets.min(n_rows));
    let mut start = search(edge(0), false);
    for i in 1..=n_buckets {
        // The last bucket is closed so that x == hi is kept. `.max(start)` guards
        // an x that is not really sorted, which `assume_sorted_x=True` allows: the
        // search then returns an index below `start` and `end - start` underflows.
        let end = search(edge(i), i == n_buckets).max(start);
        if end > start {
            offsets.push((start, end - start));
        }
        start = end;
    }
    offsets
}

fn output_name(preferred: Option<String>, current: &PlSmallStr, fallback: &str) -> PlSmallStr {
    if let Some(name) = preferred {
        if !name.is_empty() {
            return name.into();
        }
    }
    if !current.is_empty() {
        return current.clone();
    }
    fallback.into()
}

/// Per-window `(arg_min, arg_max)` row indices; `None` where a window is
/// empty or all-null.
type ArgMinMaxIdx = (Vec<Option<u64>>, Vec<Option<u64>>);

// Dispatch: try SIMD path first; fall back to Polars arg_min/arg_max per window.
fn simd_argminmax(s: &Series, offsets: &[(usize, usize)]) -> Option<ArgMinMaxIdx> {
    macro_rules! try_dispatch {
        ($downcast:ident) => {
            if let Ok(ca) = s.$downcast() {
                if ca.null_count() == 0 {
                    if let Ok(values) = ca.cont_slice() {
                        return Some(par_by_window(offsets, |o| argminmax_contiguous(values, o)));
                    }
                    let chunks: Vec<_> = ca
                        .downcast_iter()
                        .map(|arr| arr.values().as_slice())
                        .collect();
                    return Some(par_by_window(offsets, |o| argminmax_chunked(&chunks, o)));
                }
            }
        };
    }

    try_dispatch!(i64);
    try_dispatch!(i32);
    try_dispatch!(i16);
    try_dispatch!(i8);
    try_dispatch!(u64);
    try_dispatch!(u32);
    try_dispatch!(u16);
    try_dispatch!(u8);
    try_dispatch!(f64);
    try_dispatch!(f32);

    if let Ok(ca) = s.f16() {
        if ca.null_count() == 0 {
            if let Ok(values) = ca.cont_slice() {
                let values = pf16_as_half_slice(values);
                return Some(par_by_window(offsets, |o| argminmax_contiguous(values, o)));
            }
            let chunks: Vec<_> = ca
                .downcast_iter()
                .map(|arr| pf16_as_half_slice(arr.values().as_slice()))
                .collect();
            return Some(par_by_window(offsets, |o| argminmax_chunked(&chunks, o)));
        }
    }

    None
}

/// Run `f` over `offsets` on [`kernel_pool`], split into one contiguous run of
/// windows per worker.
///
/// The split is by *window*, never inside one, so every `argminmax()` call sees
/// exactly the slice it would have seen serially — the result is bit-identical to
/// the serial path by construction, with no index merging or tie-break logic.
///
/// `par_chunks` rather than a per-window `par_iter`: a few thousand windows as
/// individual rayon tasks is pure overhead, and neighbouring workers writing
/// adjacent output slots is false sharing. A contiguous run per worker avoids both.
///
/// This is a trade against the memory-bandwidth ceiling, not a free speedup.
/// Concurrent callers (N traces batched into one `select`) each take the whole
/// pool, so their scans queue instead of overlapping; whether that beats N
/// overlapped serial scans depends on how much faster one pool-wide scan is,
/// which is a property of the host's per-core vs package bandwidth. Measured
/// 2026-08-25 (100M f64 rows, n_points=1000, fused kernel): on a
/// bandwidth-saturated Zen 3 the split is worth ~1.3-1.6x at one trace and
/// costs at most ~9%, peaking at 3-5 concurrent traces and fading to ~2% by 20
/// as contended installs degrade toward the caller's own thread; on Apple
/// M-series it wins at every measured trace count. Do not add an
/// in-flight-count gate: measured twice (2026-08-24), it recovered nothing,
/// because the first arrivals still take the whole pool.
fn par_by_window<F>(offsets: &[(usize, usize)], f: F) -> ArgMinMaxIdx
where
    F: Fn(&[(usize, usize)]) -> ArgMinMaxIdx + Send + Sync,
{
    use rayon::prelude::*;

    // Below this the task and merge overhead exceed the scan. Same knob as the
    // histogram kernels; measured on the same shape.
    const MIN_PAR: usize = 1 << 17;

    let total: usize = offsets.iter().map(|&(_, len)| len).sum();
    if total < MIN_PAR || offsets.len() < 2 {
        return f(offsets);
    }

    let pool = kernel_pool();
    // ponytail: parallelism is capped by window count, so a tiny `n_points` over a
    // huge frame stays serial. n_points defaults to 1000 (500 windows), so this
    // only bites on deliberately degenerate input; split within a window if that
    // ever becomes a real shape.
    // "part" not "chunk": in this file `chunks` already means the Arrow buffers
    // backing a ChunkedArray, which are a different thing entirely.
    // Split by *rows*, not by window count: x-width buckets are wildly uneven, and
    // a few windows can hold most of the series. For equal-row-count windows this
    // lands on the same cut points as an even window split.
    let n_parts = pool.current_num_threads().max(1).min(offsets.len());
    let rows_per_part = total.div_ceil(n_parts);
    let mut bounds = Vec::with_capacity(n_parts + 1);
    bounds.push(0usize);
    let mut acc = 0usize;
    for (i, &(_, len)) in offsets.iter().enumerate() {
        acc += len;
        if acc >= rows_per_part && bounds.len() < n_parts {
            bounds.push(i + 1);
            acc = 0;
        }
    }
    bounds.push(offsets.len());

    let parts: Vec<ArgMinMaxIdx> = pool.install(|| {
        bounds
            .par_windows(2)
            .map(|b| f(&offsets[b[0]..b[1]]))
            .collect()
    });

    let mut min_indices = Vec::with_capacity(offsets.len());
    let mut max_indices = Vec::with_capacity(offsets.len());
    for (mins, maxs) in parts {
        min_indices.extend(mins);
        max_indices.extend(maxs);
    }
    (min_indices, max_indices)
}

fn argminmax_contiguous<T>(values: &[T], offsets: &[(usize, usize)]) -> ArgMinMaxIdx
where
    for<'a> &'a [T]: ArgMinMax,
{
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
    (min_indices, max_indices)
}

fn argminmax_chunked<T>(chunks: &[&[T]], offsets: &[(usize, usize)]) -> ArgMinMaxIdx
where
    T: ArgMinMaxValue,
    for<'a> &'a [T]: ArgMinMax,
{
    let mut min_indices = Vec::with_capacity(offsets.len());
    let mut max_indices = Vec::with_capacity(offsets.len());
    // Single monotone cursor — advances only when a chunk is fully consumed by a window.
    // Correct because both offset builders emit adjacent, non-overlapping windows in
    // order: an empty x-width bucket is dropped, which leaves the row ranges adjacent.
    //
    // The cursor is seeded from the *first* window rather than assumed to start at
    // row 0, which is what makes this correct for any contiguous run of windows and
    // therefore safe to hand a sub-slice of `offsets` (see `par_by_window`), or windows
    // that begin after the rows below `x_domain.0`.
    // ponytail: linear scan, chunk counts are small; binary search if that changes.
    let first_start = offsets.first().map_or(0, |&(start, _)| start);
    let mut chunk_idx = 0usize;
    let mut chunk_pos = 0usize; // global offset of chunks[chunk_idx][0]
    while chunk_idx < chunks.len() && chunk_pos + chunks[chunk_idx].len() <= first_start {
        chunk_pos += chunks[chunk_idx].len();
        chunk_idx += 1;
    }

    for &(window_start, window_len) in offsets {
        if window_len == 0 {
            min_indices.push(None);
            max_indices.push(None);
            continue;
        }
        let window_end = window_start + window_len;
        let mut min_best: Option<(usize, T)> = None;
        let mut max_best: Option<(usize, T)> = None;

        while chunk_idx < chunks.len() && chunk_pos < window_end {
            let chunk = chunks[chunk_idx];
            if chunk.is_empty() {
                chunk_idx += 1;
                continue;
            }

            let chunk_end = chunk_pos + chunk.len();
            // lo > 0 only for the first overlapping chunk; 0 for all subsequent ones.
            let lo = window_start.saturating_sub(chunk_pos);
            let hi = window_end.min(chunk_end) - chunk_pos;
            let slice = &chunk[lo..hi];
            let (min_i, max_i) = slice.argminmax();
            let base = chunk_pos + lo;
            let cmin = slice[min_i];
            let cmax = slice[max_i];
            if min_best.is_none_or(|(_, v)| min_is_better(cmin, v)) {
                min_best = Some((base + min_i, cmin));
            }
            if max_best.is_none_or(|(_, v)| max_is_better(cmax, v)) {
                max_best = Some((base + max_i, cmax));
            }
            if chunk_end <= window_end {
                // Chunk fully consumed by this window; advance cursor.
                chunk_pos = chunk_end;
                chunk_idx += 1;
            } else {
                // Window ends mid-chunk; keep cursor here for the next window.
                break;
            }
        }

        min_indices.push(min_best.map(|(i, _)| i as u64));
        max_indices.push(max_best.map(|(i, _)| i as u64));
    }

    (min_indices, max_indices)
}

#[inline(always)]
fn min_is_better<T: ArgMinMaxValue>(candidate: T, current: T) -> bool {
    !candidate.is_nan() && (current.is_nan() || candidate < current)
}

#[inline(always)]
fn max_is_better<T: ArgMinMaxValue>(candidate: T, current: T) -> bool {
    !candidate.is_nan() && (current.is_nan() || candidate > current)
}

// SAFETY CHECK: Ensure pf16 and f16 have the same layout so we can safely cast slices.
const _: () = {
    assert!(size_of::<pf16>() == size_of::<f16>());
    assert!(align_of::<pf16>() == align_of::<f16>());
};

fn pf16_as_half_slice(values: &[pf16]) -> &[f16] {
    // SAFETY: Polars 0.53 defines `pf16` as `#[repr(transparent)] pub struct
    // pf16(pub half::f16)`, so a contiguous `[pf16]` has the same layout and
    // alignment as `[half::f16]`. The returned slice is tied to `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<f16>(), values.len()) }
}

fn fallback_window_argminmax(s: &Series, offsets: &[(usize, usize)]) -> ArgMinMaxIdx {
    let mut arg_min_idx: Vec<Option<u64>> = Vec::with_capacity(offsets.len());
    let mut arg_max_idx: Vec<Option<u64>> = Vec::with_capacity(offsets.len());

    for (start, len) in offsets {
        let window = s.slice(*start as i64, *len);
        arg_min_idx.push(window.arg_min().map(|idx| (*start + idx) as u64));
        arg_max_idx.push(window.arg_max().map(|idx| (*start + idx) as u64));
    }

    (arg_min_idx, arg_max_idx)
}

fn fixed_hist_output(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new("breakpoint".into(), DataType::Float64),
            Field::new("count".into(), DataType::UInt32),
        ]),
    ))
}

/// Scalar binning over a whole Series.
///
/// No longer an entry point — `fixed_hist` is the rayon kernel. This stays as
/// the fallback the parallel path takes for null or non-contiguous chunks and
/// for dtypes it does not dispatch (see `fixed_hist_counts_par`).
fn fixed_hist_counts(values: &Series, lo: f64, hi: f64, n_bins: usize) -> PolarsResult<Vec<u32>> {
    macro_rules! try_dispatch {
        ($downcast:ident) => {
            if let Ok(ca) = values.$downcast() {
                return Ok(fixed_hist_counts_ca(ca, lo, hi, n_bins));
            }
        };
    }

    try_dispatch!(i64);
    try_dispatch!(i32);
    try_dispatch!(i16);
    try_dispatch!(i8);
    try_dispatch!(u64);
    try_dispatch!(u32);
    try_dispatch!(u16);
    try_dispatch!(u8);
    try_dispatch!(f64);
    try_dispatch!(f32);
    try_dispatch!(f16);

    polars_bail!(
        InvalidOperation:
        "fixed_hist is only supported for primitive numeric data, got {}",
        values.dtype()
    )
}

fn fixed_hist_counts_ca<T>(ca: &ChunkedArray<T>, lo: f64, hi: f64, n_bins: usize) -> Vec<u32>
where
    T: PolarsNumericType,
    T::Native: FixedHistValue,
{
    let mut counts = vec![0u32; n_bins];

    if ca.len() == ca.null_count() {
        return counts;
    }

    if hi <= lo {
        count_degenerate(ca, &mut counts);
        return counts;
    }

    // Match the old fixed_hist boundary behavior without casting the full column.
    // The small epsilon keeps integer-like values on visual bin boundaries from
    // falling into the previous bin when the caller has nudged `hi` upward.
    let scale = n_bins as f64 / (hi - lo);
    let max_idx = n_bins - 1;

    for chunk in ca.downcast_iter() {
        for item in chunk.non_null_values_iter() {
            count_value(&mut counts, item, lo, scale, max_idx);
        }
    }

    counts
}

fn count_degenerate<T>(ca: &ChunkedArray<T>, counts: &mut [u32])
where
    T: PolarsNumericType,
    T::Native: FixedHistValue,
{
    if !T::Native::CHECK_NAN {
        // Integers can't be NaN; null count is already tracked.
        counts[0] += (ca.len() - ca.null_count()) as u32;
        return;
    }
    for chunk in ca.downcast_iter() {
        for item in chunk.non_null_values_iter() {
            if !item.to_hist_f64().is_nan() {
                counts[0] += 1;
            }
        }
    }
}

#[inline(always)]
fn count_value<T: FixedHistValue>(
    counts: &mut [u32],
    item: T,
    lo: f64,
    scale: f64,
    max_idx: usize,
) {
    let v = item.to_hist_f64();
    if T::CHECK_NAN && v.is_nan() {
        return;
    }
    // Rust's saturating float→usize cast handles v < lo (negative → 0);
    // .min(max_idx) handles v >= hi (≥ n_bins → max_idx).
    counts[(((v - lo) * scale + FIXED_HIST_ROUND_EPS) as usize).min(max_idx)] += 1;
}

// ---------------------------------------------------------------------------
// fixed_hist2d
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct FixedHist2DKwargs {
    nb_x: usize,
    nb_y: usize,
}

#[derive(Clone, Copy)]
enum FixedHist2DReducer {
    Sum,
    Mean,
    Min,
    Max,
}

impl FixedHist2DReducer {
    fn parse(value: &str) -> PolarsResult<Self> {
        match value {
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            other => polars_bail!(
                InvalidOperation:
                "fixed_hist2d_reduce: histfunc must be one of sum, mean, min, max; got {}",
                other
            ),
        }
    }
}

#[derive(Deserialize)]
struct FixedHist2DReduceKwargs {
    nb_x: usize,
    nb_y: usize,
    histfunc: String,
}

fn fixed_hist2d_output(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new("z_flat".into(), DataType::List(Box::new(DataType::UInt32))),
            Field::new("x_lo".into(), DataType::Float64),
            Field::new("x_hi".into(), DataType::Float64),
            Field::new("y_lo".into(), DataType::Float64),
            Field::new("y_hi".into(), DataType::Float64),
        ]),
    ))
}

fn fixed_hist2d_reduce_output(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new("z_flat".into(), DataType::List(Box::new(DataType::Float64))),
            Field::new("x_lo".into(), DataType::Float64),
            Field::new("x_hi".into(), DataType::Float64),
            Field::new("y_lo".into(), DataType::Float64),
            Field::new("y_hi".into(), DataType::Float64),
        ]),
    ))
}

// ---------------------------------------------------------------------------
// fixed_hist2d — optimized: typed dispatch + cont_slice fast path
//
//   1. Type dispatch via FixedHistValue: avoids a `cast(&DataType::Float64)`
//      allocation for same-type pairs.  For f64 data (the common case) the
//      downcast is zero-cost.  Mixed-type pairs fall back to cast — rare.
//
//   2. cont_slice fast path: when both series are single-chunk and null-free
//      (the usual state after a `.collect()`), iterate raw `&[T]` slices,
//      eliminating the `Option<T>` wrap/unwrap overhead from `.iter()`.
//
// Algorithm reference: fast-histogram (astrofrog) uses the same O(n) direct
// floor-division strategy in C, achieving 20-25× over numpy.histogram2d.
// ---------------------------------------------------------------------------

// Generic inner loop shared by all typed paths.
/// The 2D binning loop over one contiguous, null-free slice pair.
///
/// Shared by the scalar and the rayon path so the two cannot drift apart in
/// their bin arithmetic — the counts must stay bit-identical. Generic over the
/// two axes *independently*: a grouped histogram bins an integer group code
/// against a float value, and casting either side to match the other would
/// materialize a full-length column before the kernel ever runs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn count_2d_slices<X, Y>(
    xs: &[X],
    ys: &[Y],
    counts: &mut [u32],
    x_lo: f64,
    x_scale: f64,
    max_xi: usize,
    y_lo: f64,
    y_scale: f64,
    max_yi: usize,
    nb_x: usize,
) where
    X: FixedHistValue,
    Y: FixedHistValue,
{
    for (&xv, &yv) in xs.iter().zip(ys.iter()) {
        let xf = xv.to_hist_f64();
        let yf = yv.to_hist_f64();
        if (X::CHECK_NAN && xf.is_nan()) || (Y::CHECK_NAN && yf.is_nan()) {
            continue;
        }
        let xi = ((xf - x_lo) * x_scale + FIXED_HIST_ROUND_EPS) as usize;
        let yi = ((yf - y_lo) * y_scale + FIXED_HIST_ROUND_EPS) as usize;
        counts[yi.min(max_yi) * nb_x + xi.min(max_xi)] += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn count_2d_ca<T>(
    x_ca: &ChunkedArray<T>,
    y_ca: &ChunkedArray<T>,
    counts: &mut [u32],
    x_lo: f64,
    x_scale: f64,
    max_xi: usize,
    y_lo: f64,
    y_scale: f64,
    max_yi: usize,
    nb_x: usize,
) where
    T: PolarsNumericType,
    T::Native: FixedHistValue,
{
    // Fast path: single contiguous chunk, no nulls → raw slice pair, no Option.
    if x_ca.null_count() == 0 && y_ca.null_count() == 0 {
        if let (Ok(xs), Ok(ys)) = (x_ca.cont_slice(), y_ca.cont_slice()) {
            count_2d_slices(
                xs, ys, counts, x_lo, x_scale, max_xi, y_lo, y_scale, max_yi, nb_x,
            );
            return;
        }
    }
    // General path: multi-chunk or nulls present → paired Option iterator.
    for (xv_opt, yv_opt) in x_ca.iter().zip(y_ca.iter()) {
        if let (Some(xv), Some(yv)) = (xv_opt, yv_opt) {
            let xf = xv.to_hist_f64();
            let yf = yv.to_hist_f64();
            if T::Native::CHECK_NAN && (xf.is_nan() || yf.is_nan()) {
                continue;
            }
            let xi = ((xf - x_lo) * x_scale + FIXED_HIST_ROUND_EPS) as usize;
            let yi = ((yf - y_lo) * y_scale + FIXED_HIST_ROUND_EPS) as usize;
            counts[yi.min(max_yi) * nb_x + xi.min(max_xi)] += 1;
        }
    }
}

#[inline(always)]
fn update_reducer(
    values: &mut [f64],
    seen: &mut [bool],
    counts: &mut [usize],
    idx: usize,
    z: f64,
    reducer: FixedHist2DReducer,
) {
    match reducer {
        FixedHist2DReducer::Sum => {
            values[idx] += z;
            seen[idx] = true;
        },
        FixedHist2DReducer::Mean => {
            values[idx] += z;
            counts[idx] += 1;
            seen[idx] = true;
        },
        FixedHist2DReducer::Min => {
            if !seen[idx] || z < values[idx] {
                values[idx] = z;
            }
            seen[idx] = true;
        },
        FixedHist2DReducer::Max => {
            if !seen[idx] || z > values[idx] {
                values[idx] = z;
            }
            seen[idx] = true;
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn reduce_2d_ca<T>(
    x_ca: &ChunkedArray<T>,
    y_ca: &ChunkedArray<T>,
    z_ca: &Float64Chunked,
    values: &mut [f64],
    seen: &mut [bool],
    counts: &mut [usize],
    x_lo: f64,
    x_scale: f64,
    max_xi: usize,
    y_lo: f64,
    y_scale: f64,
    max_yi: usize,
    nb_x: usize,
    reducer: FixedHist2DReducer,
) where
    T: PolarsNumericType,
    T::Native: FixedHistValue,
{
    if x_ca.null_count() == 0 && y_ca.null_count() == 0 && z_ca.null_count() == 0 {
        if let (Ok(xs), Ok(ys), Ok(zs)) = (x_ca.cont_slice(), y_ca.cont_slice(), z_ca.cont_slice())
        {
            for ((&xv, &yv), &zv) in xs.iter().zip(ys.iter()).zip(zs.iter()) {
                let xf = xv.to_hist_f64();
                let yf = yv.to_hist_f64();
                if zv.is_nan() || (T::Native::CHECK_NAN && (xf.is_nan() || yf.is_nan())) {
                    continue;
                }
                let xi = ((xf - x_lo) * x_scale + FIXED_HIST_ROUND_EPS) as usize;
                let yi = ((yf - y_lo) * y_scale + FIXED_HIST_ROUND_EPS) as usize;
                let idx = yi.min(max_yi) * nb_x + xi.min(max_xi);
                update_reducer(values, seen, counts, idx, zv, reducer);
            }
            return;
        }
    }

    for ((xv_opt, yv_opt), zv_opt) in x_ca.iter().zip(y_ca.iter()).zip(z_ca.iter()) {
        if let (Some(xv), Some(yv), Some(zv)) = (xv_opt, yv_opt, zv_opt) {
            let xf = xv.to_hist_f64();
            let yf = yv.to_hist_f64();
            if zv.is_nan() || (T::Native::CHECK_NAN && (xf.is_nan() || yf.is_nan())) {
                continue;
            }
            let xi = ((xf - x_lo) * x_scale + FIXED_HIST_ROUND_EPS) as usize;
            let yi = ((yf - y_lo) * y_scale + FIXED_HIST_ROUND_EPS) as usize;
            let idx = yi.min(max_yi) * nb_x + xi.min(max_xi);
            update_reducer(values, seen, counts, idx, zv, reducer);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fixed_hist2d_counts(
    x: &Series,
    y: &Series,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    nb_x: usize,
    nb_y: usize,
) -> PolarsResult<Vec<u32>> {
    polars_ensure!(
        x.len() == y.len(),
        ShapeMismatch: "fixed_hist2d: x and y must have the same length"
    );
    polars_ensure!(
        x_hi >= x_lo && y_hi >= y_lo,
        InvalidOperation: "fixed_hist2d: x_hi must be >= x_lo and y_hi must be >= y_lo"
    );

    let Some(bin_count) = nb_x.checked_mul(nb_y) else {
        polars_bail!(InvalidOperation: "fixed_hist2d: nb_x * nb_y overflows usize");
    };
    let mut counts = vec![0u32; bin_count];
    if nb_x == 0 || nb_y == 0 || x.is_empty() {
        return Ok(counts);
    }

    let x_scale = nb_x as f64 / (x_hi - x_lo + FIXED_HIST2D_SPAN_EPS);
    let y_scale = nb_y as f64 / (y_hi - y_lo + FIXED_HIST2D_SPAN_EPS);
    let max_xi = nb_x - 1;
    let max_yi = nb_y - 1;

    // Try same-type zero-copy downcast.  Covers f64/f32/f16 and all integer
    // pairs.  Mixed-type pairs (e.g. f32 x, i64 y) fall through to the cast
    // fallback — an unusual case in practice.
    macro_rules! try_typed {
        ($m:ident) => {
            if let (Ok(xca), Ok(yca)) = (x.$m(), y.$m()) {
                count_2d_ca(
                    xca,
                    yca,
                    &mut counts,
                    x_lo,
                    x_scale,
                    max_xi,
                    y_lo,
                    y_scale,
                    max_yi,
                    nb_x,
                );
                return Ok(counts);
            }
        };
    }

    try_typed!(f64);
    try_typed!(f32);
    try_typed!(f16);
    try_typed!(i64);
    try_typed!(i32);
    try_typed!(i16);
    try_typed!(i8);
    try_typed!(u64);
    try_typed!(u32);
    try_typed!(u16);
    try_typed!(u8);

    // Mixed-type fallback: cast both to f64 (rare).
    let xf = x.cast(&DataType::Float64)?;
    let yf = y.cast(&DataType::Float64)?;
    count_2d_ca(
        xf.f64()?,
        yf.f64()?,
        &mut counts,
        x_lo,
        x_scale,
        max_xi,
        y_lo,
        y_scale,
        max_yi,
        nb_x,
    );
    Ok(counts)
}

#[allow(clippy::too_many_arguments)]
fn fixed_hist2d_reduce_values(
    x: &Series,
    y: &Series,
    z: &Series,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    nb_x: usize,
    nb_y: usize,
    reducer: FixedHist2DReducer,
) -> PolarsResult<Vec<Option<f64>>> {
    polars_ensure!(
        x.len() == y.len() && x.len() == z.len(),
        ShapeMismatch: "fixed_hist2d_reduce: x, y, and z must have the same length"
    );
    polars_ensure!(
        x_hi >= x_lo && y_hi >= y_lo,
        InvalidOperation: "fixed_hist2d_reduce: x_hi must be >= x_lo and y_hi must be >= y_lo"
    );

    let Some(bin_count) = nb_x.checked_mul(nb_y) else {
        polars_bail!(
            InvalidOperation:
            "fixed_hist2d_reduce: nb_x * nb_y overflows usize"
        );
    };

    let mut values = vec![0.0f64; bin_count];
    let mut seen = vec![false; bin_count];
    let mut counts = vec![0usize; bin_count];
    if nb_x == 0 || nb_y == 0 || x.is_empty() {
        return Ok(vec![None; bin_count]);
    }

    let z_f64 = z.cast(&DataType::Float64)?;
    let z_ca = z_f64.f64()?;

    let x_scale = nb_x as f64 / (x_hi - x_lo + FIXED_HIST2D_SPAN_EPS);
    let y_scale = nb_y as f64 / (y_hi - y_lo + FIXED_HIST2D_SPAN_EPS);
    let max_xi = nb_x - 1;
    let max_yi = nb_y - 1;

    macro_rules! try_typed {
        ($m:ident) => {
            if let (Ok(xca), Ok(yca)) = (x.$m(), y.$m()) {
                reduce_2d_ca(
                    xca,
                    yca,
                    z_ca,
                    &mut values,
                    &mut seen,
                    &mut counts,
                    x_lo,
                    x_scale,
                    max_xi,
                    y_lo,
                    y_scale,
                    max_yi,
                    nb_x,
                    reducer,
                );
                return Ok(finalize_reduced_values(values, seen, counts, reducer));
            }
        };
    }

    try_typed!(f64);
    try_typed!(f32);
    try_typed!(f16);
    try_typed!(i64);
    try_typed!(i32);
    try_typed!(i16);
    try_typed!(i8);
    try_typed!(u64);
    try_typed!(u32);
    try_typed!(u16);
    try_typed!(u8);

    let xf = x.cast(&DataType::Float64)?;
    let yf = y.cast(&DataType::Float64)?;
    reduce_2d_ca(
        xf.f64()?,
        yf.f64()?,
        z_ca,
        &mut values,
        &mut seen,
        &mut counts,
        x_lo,
        x_scale,
        max_xi,
        y_lo,
        y_scale,
        max_yi,
        nb_x,
        reducer,
    );
    Ok(finalize_reduced_values(values, seen, counts, reducer))
}

fn finalize_reduced_values(
    values: Vec<f64>,
    seen: Vec<bool>,
    counts: Vec<usize>,
    reducer: FixedHist2DReducer,
) -> Vec<Option<f64>> {
    debug_assert_eq!(values.len(), seen.len());
    debug_assert_eq!(values.len(), counts.len());

    values
        .into_iter()
        .zip(seen)
        .zip(counts)
        .map(|((value, was_seen), count)| {
            if !was_seen {
                None
            } else if matches!(reducer, FixedHist2DReducer::Mean) {
                Some(value / count as f64)
            } else {
                Some(value)
            }
        })
        .collect()
}

/// One work unit for the parallel 2D binner: a contiguous, null-free slice of
/// each axis, already aligned to the same rows.
type SlicePair<'a, X, Y> = (&'a [X], &'a [Y]);

/// Bound on all live private count tables together, shared by the 1D and 2D
/// parallel histogram kernels. A per-table cutoff creates a performance cliff
/// even when fewer workers would fit comfortably.
/// ponytail: tables above the budget stay scalar; use a shared or striped
/// accumulator if a renderer ever needs outputs that large.
const MAX_PRIVATE_BYTES: usize = 32 << 20;

/// Rayon 2D binning: private per-unit count tables, merged by add.
///
/// Same shape as [`fixed_hist_counts_par_ca`] — one `nb_x * nb_y` u32 table per
/// work unit, with the worker count reduced as tables grow so total scratch
/// stays bounded. Returns `None` when the pair is not contiguous and null-free,
/// is too small to split, or two private tables do not fit; the caller then
/// keeps the scalar path.
#[allow(clippy::too_many_arguments)]
fn fixed_hist2d_counts_par_ca<TX, TY>(
    x_ca: &ChunkedArray<TX>,
    y_ca: &ChunkedArray<TY>,
    x_lo: f64,
    x_scale: f64,
    max_xi: usize,
    y_lo: f64,
    y_scale: f64,
    max_yi: usize,
    nb_x: usize,
    bin_count: usize,
) -> Option<Vec<u32>>
where
    TX: PolarsNumericType,
    TX::Native: FixedHistValue + Send + Sync,
    TY: PolarsNumericType,
    TY::Native: FixedHistValue + Send + Sync,
{
    use rayon::prelude::*;

    // Below this the merge and task overhead exceed the scan (same threshold as
    // the 1D kernel).
    const MIN_PAR: usize = 1 << 17;
    let bytes_per_table = bin_count.checked_mul(size_of::<u32>())?;
    let max_units = MAX_PRIVATE_BYTES / bytes_per_table;
    if max_units < 2 {
        return None;
    }

    // x and y may be chunked differently; align before pairing the slices.
    let (xa, ya) = align_chunks_binary(x_ca, y_ca);
    let mut runs: Vec<SlicePair<TX::Native, TY::Native>> = Vec::with_capacity(xa.chunks().len());
    for (xarr, yarr) in xa.downcast_iter().zip(ya.downcast_iter()) {
        if xarr.validity().is_some_and(|v| v.unset_bits() != 0)
            || yarr.validity().is_some_and(|v| v.unset_bits() != 0)
        {
            return None;
        }
        runs.push((xarr.values().as_slice(), yarr.values().as_slice()));
    }
    let total: usize = runs.iter().map(|(xs, _)| xs.len()).sum();
    if total < MIN_PAR {
        return None;
    }

    let pool = kernel_pool();
    let n_chunks = pool.current_num_threads().min(max_units);
    let chunk_len = total.div_ceil(n_chunks).max(MIN_PAR);
    // One flat work list across all runs, so a single huge chunk and many small
    // ones both split evenly.
    let units: Vec<SlicePair<TX::Native, TY::Native>> = runs
        .iter()
        .flat_map(|(xs, ys)| xs.chunks(chunk_len).zip(ys.chunks(chunk_len)))
        .collect();

    // Fragmented input yields at least one unit per run, which can exceed
    // `max_units` — so the budget is enforced on groups: at most `n_chunks`
    // tables are ever live, one per group, however the input is chunked.
    pool.install(|| {
        units
            .par_chunks(units.len().div_ceil(n_chunks))
            .map(|group| {
                let mut local = vec![0u32; bin_count];
                for &(xs, ys) in group {
                    count_2d_slices(
                        xs, ys, &mut local, x_lo, x_scale, max_xi, y_lo, y_scale, max_yi, nb_x,
                    );
                }
                local
            })
            .reduce_with(|mut a, b| {
                for (p, q) in a.iter_mut().zip(b) {
                    *p += q;
                }
                a
            })
    })
}

#[allow(clippy::too_many_arguments)]
fn fixed_hist2d_counts_par(
    x: &Series,
    y: &Series,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    nb_x: usize,
    nb_y: usize,
) -> PolarsResult<Vec<u32>> {
    let scalar = || fixed_hist2d_counts(x, y, x_lo, x_hi, y_lo, y_hi, nb_x, nb_y);
    if nb_x == 0 || nb_y == 0 || x.is_empty() || x.len() != y.len() || x_hi < x_lo || y_hi < y_lo {
        return scalar(); // let the scalar path own the validation and the errors
    }
    let Some(bin_count) = nb_x.checked_mul(nb_y) else {
        return scalar();
    };

    let x_scale = nb_x as f64 / (x_hi - x_lo + FIXED_HIST2D_SPAN_EPS);
    let y_scale = nb_y as f64 / (y_hi - y_lo + FIXED_HIST2D_SPAN_EPS);
    let (max_xi, max_yi) = (nb_x - 1, nb_y - 1);

    // Dispatch the two axes independently. A grouped histogram is
    // `(integer group code, float value)`, so restricting the fast path to
    // same-dtype pairs — as the scalar kernel does — would cast a full-length
    // column and cost more than the binning itself.
    macro_rules! try_pair {
        ($xm:ident, $ym:ident) => {
            if let (Ok(xca), Ok(yca)) = (x.$xm(), y.$ym()) {
                if let Some(counts) = fixed_hist2d_counts_par_ca(
                    xca, yca, x_lo, x_scale, max_xi, y_lo, y_scale, max_yi, nb_x, bin_count,
                ) {
                    return Ok(counts);
                }
            }
        };
    }
    macro_rules! try_x {
        ($xm:ident) => {
            try_pair!($xm, f64);
            try_pair!($xm, f32);
            try_pair!($xm, i64);
            try_pair!($xm, i32);
        };
    }
    try_x!(f64);
    try_x!(f32);
    try_x!(i64);
    try_x!(i32);
    try_x!(u32); // Enum / Categorical physical codes
    try_x!(u8);
    // Rarer dtype pairs keep the scalar path.
    scalar()
}

fn fixed_hist2d_impl(inputs: &[Series], kwargs: FixedHist2DKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.nb_x > 0 && kwargs.nb_y > 0,
        InvalidOperation: "fixed_hist2d: nb_x and nb_y must be greater than 0"
    );

    let x = &inputs[0];
    let y = &inputs[1];
    let x_lo = inputs[2]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d: x_lo is null"))?;
    let x_hi = inputs[3]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d: x_hi is null"))?;
    let y_lo = inputs[4]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d: y_lo is null"))?;
    let y_hi = inputs[5]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d: y_hi is null"))?;

    let counts = fixed_hist2d_counts_par(x, y, x_lo, x_hi, y_lo, y_hi, kwargs.nb_x, kwargs.nb_y)?;

    let mut z_flat = UInt32Chunked::from_vec("z_flat".into(), counts)
        .into_series()
        .implode()?
        .into_series();
    z_flat.rename("z_flat".into());

    let x_lo_s = Float64Chunked::from_vec("x_lo".into(), vec![x_lo]).into_series();
    let x_hi_s = Float64Chunked::from_vec("x_hi".into(), vec![x_hi]).into_series();
    let y_lo_s = Float64Chunked::from_vec("y_lo".into(), vec![y_lo]).into_series();
    let y_hi_s = Float64Chunked::from_vec("y_hi".into(), vec![y_hi]).into_series();

    let out = StructChunked::from_series(
        "fixed_hist2d".into(),
        1,
        [&z_flat, &x_lo_s, &x_hi_s, &y_lo_s, &y_hi_s].into_iter(),
    )?;
    Ok(out.into_series())
}

/// Fixed-bin 2D histogram. Rayon-parallel, with a scalar fallback for input the
/// parallel path cannot take (see `fixed_hist2d_counts_par`); counts are
/// identical either way.
#[polars_expr(output_type_func = fixed_hist2d_output)]
fn fixed_hist2d(inputs: &[Series], kwargs: FixedHist2DKwargs) -> PolarsResult<Series> {
    fixed_hist2d_impl(inputs, kwargs)
}

// ---------------------------------------------------------------------------
// fixed_line_envelope2d — one-pass exact argmin/argmax-by-y per
// (x bucket, free bin) cell (cube line-envelope target, issue #36).
//
// Bin arithmetic is the cube *shared arithmetic* on BOTH axes (not the
// fixed_hist epsilon arithmetic): natural floor `floor((v - lo) / span * n)`
// with NO epsilon and NO clip — out-of-domain rows are filtered, and a value
// exactly at the domain max lands in the degenerate top bin, so indices run
// 0..=n_buckets and 0..=p inclusive. Null/NaN x, y, or free ⇒ row filtered.
// Ties (equal y within a cell): the FIRST row in scan order wins for both
// min and max (strict comparisons) — deterministic.
//
// Inputs are cast to Float64 (cheap no-op for f64 columns; temporal columns
// must be cast to physical by the Python caller, as for fixed_hist). x/y are
// returned as exact f64 — quantization is the codec's job, not the kernel's.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct FixedLineEnvelope2DKwargs {
    n_buckets: usize,
    p: usize,
}

fn fixed_line_envelope2d_output(_inputs: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new("bucket".into(), DataType::UInt32),
            Field::new("free_bin".into(), DataType::UInt32),
            Field::new("y_min".into(), DataType::Float64),
            Field::new("x_at_ymin".into(), DataType::Float64),
            Field::new("y_max".into(), DataType::Float64),
            Field::new("x_at_ymax".into(), DataType::Float64),
        ]),
    ))
}

/// Dense per-cell envelope accumulator; compacted to the sparse long-format
/// output after the scan.
struct EnvelopeAcc {
    seen: Vec<bool>,
    y_min: Vec<f64>,
    x_at_ymin: Vec<f64>,
    y_max: Vec<f64>,
    x_at_ymax: Vec<f64>,
}

impl EnvelopeAcc {
    fn new(n_cells: usize) -> Self {
        Self {
            seen: vec![false; n_cells],
            y_min: vec![0.0; n_cells],
            x_at_ymin: vec![0.0; n_cells],
            y_max: vec![0.0; n_cells],
            x_at_ymax: vec![0.0; n_cells],
        }
    }

    #[inline(always)]
    fn update(&mut self, idx: usize, xv: f64, yv: f64) {
        if !self.seen[idx] {
            self.seen[idx] = true;
            self.y_min[idx] = yv;
            self.x_at_ymin[idx] = xv;
            self.y_max[idx] = yv;
            self.x_at_ymax[idx] = xv;
        } else {
            // Strict comparisons: equal y keeps the earlier row (first wins).
            if yv < self.y_min[idx] {
                self.y_min[idx] = yv;
                self.x_at_ymin[idx] = xv;
            }
            if yv > self.y_max[idx] {
                self.y_max[idx] = yv;
                self.x_at_ymax[idx] = xv;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn envelope_scan(
    x_ca: &Float64Chunked,
    y_ca: &Float64Chunked,
    f_ca: &Float64Chunked,
    x_lo: f64,
    x_hi: f64,
    f_lo: f64,
    f_hi: f64,
    n_buckets: usize,
    p: usize,
) -> PolarsResult<EnvelopeAcc> {
    let stride = n_buckets + 1; // buckets 0..=n_buckets (degenerate top bin)
    let Some(n_cells) = stride.checked_mul(p + 1) else {
        polars_bail!(
            InvalidOperation:
            "fixed_line_envelope2d: (n_buckets + 1) * (p + 1) overflows usize"
        );
    };
    let mut acc = EnvelopeAcc::new(n_cells);

    // Shared arithmetic: floor((v - lo) / span * n) with true IEEE division —
    // matching the JS client's `/` bit-exactly. (Polars' own scalar division,
    // as in cube.py's _free_bin_expr, is multiply-by-reciprocal and can differ
    // by 1 ulp at exact bin edges; the Polars test reference forces true
    // division via a materialized span column.) A degenerate (lo == hi)
    // domain admits only v == lo, which bins to 0.
    let x_span = if x_hi > x_lo { x_hi - x_lo } else { 1.0 };
    let f_span = if f_hi > f_lo { f_hi - f_lo } else { 1.0 };
    let nb_f64 = n_buckets as f64;
    let p_f64 = p as f64;

    let mut visit = |xv: f64, yv: f64, fv: f64| {
        // Filter, don't clip: NaN fails these range checks too, so NaN in any
        // column drops the row exactly like a null.
        if !(xv >= x_lo && xv <= x_hi && fv >= f_lo && fv <= f_hi) || yv.is_nan() {
            return;
        }
        // In-domain ⇒ the ratio is in [0, 1] exactly, so bx <= n_buckets and
        // bf <= p (degenerate top bins included) — idx is always in bounds.
        let bx = ((xv - x_lo) / x_span * nb_f64).floor() as usize;
        let bf = ((fv - f_lo) / f_span * p_f64).floor() as usize;
        acc.update(bf * stride + bx, xv, yv);
    };

    let mut scanned = false;
    if x_ca.null_count() == 0 && y_ca.null_count() == 0 && f_ca.null_count() == 0 {
        // Fast path: single contiguous chunk, no nulls → raw slices.
        if let (Ok(xs), Ok(ys), Ok(fs)) = (x_ca.cont_slice(), y_ca.cont_slice(), f_ca.cont_slice())
        {
            for ((&xv, &yv), &fv) in xs.iter().zip(ys.iter()).zip(fs.iter()) {
                visit(xv, yv, fv);
            }
            scanned = true;
        }
    }
    if !scanned {
        for ((xv_opt, yv_opt), fv_opt) in x_ca.iter().zip(y_ca.iter()).zip(f_ca.iter()) {
            if let (Some(xv), Some(yv), Some(fv)) = (xv_opt, yv_opt, fv_opt) {
                visit(xv, yv, fv);
            }
        }
    }

    Ok(acc)
}

#[polars_expr(output_type_func = fixed_line_envelope2d_output)]
fn fixed_line_envelope2d(
    inputs: &[Series],
    kwargs: FixedLineEnvelope2DKwargs,
) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.n_buckets > 0,
        InvalidOperation: "fixed_line_envelope2d: n_buckets must be greater than 0"
    );
    polars_ensure!(
        kwargs.p > 0,
        InvalidOperation: "fixed_line_envelope2d: p must be greater than 0"
    );

    let x = &inputs[0];
    let y = &inputs[1];
    let free = &inputs[2];
    polars_ensure!(
        x.len() == y.len() && x.len() == free.len(),
        ShapeMismatch: "fixed_line_envelope2d: x, y, and free must have the same length"
    );
    for (s, role) in [(x, "x"), (y, "y"), (free, "free")] {
        polars_ensure!(
            s.dtype().is_primitive_numeric(),
            InvalidOperation:
            "fixed_line_envelope2d: {} must be primitive numeric, got {}",
            role, s.dtype()
        );
    }

    let scalar_bound = |idx: usize, name: &str| -> PolarsResult<f64> {
        inputs[idx]
            .cast(&DataType::Float64)?
            .f64()?
            .get(0)
            .ok_or_else(|| polars_err!(InvalidOperation: "fixed_line_envelope2d: {} is null", name))
    };
    let x_lo = scalar_bound(3, "x_lo")?;
    let x_hi = scalar_bound(4, "x_hi")?;
    let f_lo = scalar_bound(5, "free_lo")?;
    let f_hi = scalar_bound(6, "free_hi")?;
    polars_ensure!(
        x_lo <= x_hi && f_lo <= f_hi,
        InvalidOperation:
        "fixed_line_envelope2d: x_lo must be <= x_hi and free_lo must be <= free_hi"
    );

    // Float64 casts are no-ops for f64 columns; integer/f32/f16 columns cast
    // exactly like the Polars reference (`.cast(pl.Float64)`).
    let x_f64 = x.cast(&DataType::Float64)?;
    let y_f64 = y.cast(&DataType::Float64)?;
    let f_f64 = free.cast(&DataType::Float64)?;

    let acc = envelope_scan(
        x_f64.f64()?,
        y_f64.f64()?,
        f_f64.f64()?,
        x_lo,
        x_hi,
        f_lo,
        f_hi,
        kwargs.n_buckets,
        kwargs.p,
    )?;

    // Compact dense cells → sparse long format, ascending (free_bin, bucket).
    let stride = kwargs.n_buckets + 1;
    let n_out = acc.seen.iter().filter(|&&s| s).count();
    let mut bucket = Vec::with_capacity(n_out);
    let mut free_bin = Vec::with_capacity(n_out);
    let mut y_min = Vec::with_capacity(n_out);
    let mut x_at_ymin = Vec::with_capacity(n_out);
    let mut y_max = Vec::with_capacity(n_out);
    let mut x_at_ymax = Vec::with_capacity(n_out);
    for bf in 0..=kwargs.p {
        for bx in 0..stride {
            let idx = bf * stride + bx;
            if acc.seen[idx] {
                bucket.push(bx as u32);
                free_bin.push(bf as u32);
                y_min.push(acc.y_min[idx]);
                x_at_ymin.push(acc.x_at_ymin[idx]);
                y_max.push(acc.y_max[idx]);
                x_at_ymax.push(acc.x_at_ymax[idx]);
            }
        }
    }

    let bucket_s = UInt32Chunked::from_vec("bucket".into(), bucket).into_series();
    let free_bin_s = UInt32Chunked::from_vec("free_bin".into(), free_bin).into_series();
    let y_min_s = Float64Chunked::from_vec("y_min".into(), y_min).into_series();
    let x_at_ymin_s = Float64Chunked::from_vec("x_at_ymin".into(), x_at_ymin).into_series();
    let y_max_s = Float64Chunked::from_vec("y_max".into(), y_max).into_series();
    let x_at_ymax_s = Float64Chunked::from_vec("x_at_ymax".into(), x_at_ymax).into_series();

    let out = StructChunked::from_series(
        "fixed_line_envelope2d".into(),
        n_out,
        [
            &bucket_s,
            &free_bin_s,
            &y_min_s,
            &x_at_ymin_s,
            &y_max_s,
            &x_at_ymax_s,
        ]
        .into_iter(),
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func = fixed_hist2d_reduce_output)]
fn fixed_hist2d_reduce(inputs: &[Series], kwargs: FixedHist2DReduceKwargs) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.nb_x > 0 && kwargs.nb_y > 0,
        InvalidOperation: "fixed_hist2d_reduce: nb_x and nb_y must be greater than 0"
    );

    let reducer = FixedHist2DReducer::parse(&kwargs.histfunc)?;
    let x = &inputs[0];
    let y = &inputs[1];
    let z = &inputs[2];
    let x_lo = inputs[3]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d_reduce: x_lo is null"))?;
    let x_hi = inputs[4]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d_reduce: x_hi is null"))?;
    let y_lo = inputs[5]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d_reduce: y_lo is null"))?;
    let y_hi = inputs[6]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .ok_or_else(|| polars_err!(InvalidOperation: "fixed_hist2d_reduce: y_hi is null"))?;

    let values = fixed_hist2d_reduce_values(
        x,
        y,
        z,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        kwargs.nb_x,
        kwargs.nb_y,
        reducer,
    )?;

    let mut z_flat = Float64Chunked::from_iter_options("z_flat".into(), values.into_iter())
        .into_series()
        .implode()?
        .into_series();
    z_flat.rename("z_flat".into());

    let x_lo_s = Float64Chunked::from_vec("x_lo".into(), vec![x_lo]).into_series();
    let x_hi_s = Float64Chunked::from_vec("x_hi".into(), vec![x_hi]).into_series();
    let y_lo_s = Float64Chunked::from_vec("y_lo".into(), vec![y_lo]).into_series();
    let y_hi_s = Float64Chunked::from_vec("y_hi".into(), vec![y_hi]).into_series();

    let out = StructChunked::from_series(
        "fixed_hist2d_reduce".into(),
        1,
        [&z_flat, &x_lo_s, &x_hi_s, &y_lo_s, &y_hi_s].into_iter(),
    )?;
    Ok(out.into_series())
}

// ---------------------------------------------------------------------------
// fixed_hist — parallel counting core
//
// The production implementation behind `fixed_hist` (design + measurements:
// benchmarks/results/hist-parallel-2026-08-06.md). The scalar path
// (`fixed_hist_counts_ca`) stays as the fallback for nulls, undispatched
// dtypes, and input below `MIN_PAR`. Two independent ideas shape the hot loop:
//
//  1. Split the index computation from the increment. The index math is pure
//     arithmetic over a slice with no stores into `counts`, so LLVM can
//     vectorise it; the increment is an unavoidable scalar scatter. Interleaved
//     (as in `count_value`) the store blocks vectorisation of the whole loop.
//
//  2. Private per-group histograms merged at the end. n_bins is small (a
//     256-bin u32 table is 1 KiB, comfortably L1), so replication is nearly
//     free and removes all cross-thread contention; `MAX_PRIVATE_BYTES` caps
//     how many tables are ever live at once.
// ---------------------------------------------------------------------------

const HIST_BLOCK: usize = 4096;
const NAN_SENTINEL: u32 = u32::MAX;

#[inline(always)]
fn hist_accumulate<T: FixedHistValue>(
    vals: &[T],
    lo: f64,
    scale: f64,
    max_idx: usize,
    counts: &mut [u32],
) {
    let mut idx = [0u32; HIST_BLOCK];
    for block in vals.chunks(HIST_BLOCK) {
        for (dst, v) in idx[..block.len()].iter_mut().zip(block) {
            let x = v.to_hist_f64();
            // Rust's float->usize cast saturates, so x < lo lands in bin 0 and
            // x >= hi is caught by min(max_idx) — same as `count_value`.
            let b = (x - lo) * scale + FIXED_HIST_ROUND_EPS;
            *dst = if T::CHECK_NAN && x.is_nan() {
                NAN_SENTINEL
            } else {
                (b as usize).min(max_idx) as u32
            };
        }
        for &b in &idx[..block.len()] {
            if !T::CHECK_NAN || b != NAN_SENTINEL {
                counts[b as usize] += 1;
            }
        }
    }
}

fn fixed_hist_counts_par_ca<T>(ca: &ChunkedArray<T>, lo: f64, hi: f64, n_bins: usize) -> Vec<u32>
where
    T: PolarsNumericType,
    T::Native: FixedHistValue + Send + Sync,
{
    use rayon::prelude::*;

    if ca.len() == ca.null_count() || hi <= lo {
        return fixed_hist_counts_ca(ca, lo, hi, n_bins);
    }
    // Collect the contiguous null-free runs. Multi-chunk input is the common
    // case for concatenated frames and for anything the streaming engine
    // buffers, so it must NOT fall back — an earlier version of this only
    // handled a single chunk and silently ran scalar (20x slower) on two.
    let mut runs: Vec<&[T::Native]> = Vec::with_capacity(ca.chunks().len());
    for arr in ca.downcast_iter() {
        if arr.validity().is_some_and(|v| v.unset_bits() != 0) {
            return fixed_hist_counts_ca(ca, lo, hi, n_bins);
        }
        runs.push(arr.values().as_slice());
    }

    let scale = n_bins as f64 / (hi - lo);
    let max_idx = n_bins - 1;
    let total: usize = runs.iter().map(|r| r.len()).sum();

    // Below this the merge and task overhead dominate the scan.
    const MIN_PAR: usize = 1 << 17;
    // The same private-table budget as the 2D kernel: above it, parallel
    // scratch would multiply an already-large table by the worker count.
    let bytes_per_table = n_bins.saturating_mul(size_of::<u32>());
    let max_units = MAX_PRIVATE_BYTES / bytes_per_table.max(1);
    if total < MIN_PAR || max_units < 2 {
        let mut counts = vec![0u32; n_bins];
        for r in &runs {
            hist_accumulate(r, lo, scale, max_idx, &mut counts);
        }
        return counts;
    }

    let pool = kernel_pool();
    let n_chunks = pool.current_num_threads().max(1).min(max_units);
    let chunk_len = total.div_ceil(n_chunks).max(MIN_PAR);
    // One flat work list across all runs, so a single huge chunk and many small
    // ones both split evenly.
    let units: Vec<&[T::Native]> = runs.iter().flat_map(|r| r.chunks(chunk_len)).collect();

    // Fragmented input yields at least one unit per run, which can exceed
    // `max_units` — so the budget is enforced on groups: at most `n_chunks`
    // tables are ever live, one per group, however the input is chunked.
    pool.install(|| {
        units
            .par_chunks(units.len().div_ceil(n_chunks))
            .map(|group| {
                let mut local = vec![0u32; n_bins];
                for &c in group {
                    hist_accumulate(c, lo, scale, max_idx, &mut local);
                }
                local
            })
            .reduce_with(|mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            })
            .unwrap_or_else(|| vec![0u32; n_bins])
    })
}

fn fixed_hist_counts_par(
    values: &Series,
    lo: f64,
    hi: f64,
    n_bins: usize,
) -> PolarsResult<Vec<u32>> {
    macro_rules! try_dispatch {
        ($downcast:ident) => {
            if let Ok(ca) = values.$downcast() {
                return Ok(fixed_hist_counts_par_ca(ca, lo, hi, n_bins));
            }
        };
    }
    try_dispatch!(i64);
    try_dispatch!(i32);
    try_dispatch!(f64);
    try_dispatch!(f32);
    // Rarer dtypes keep the scalar path; this is an experiment, not a rewrite.
    fixed_hist_counts(values, lo, hi, n_bins)
}

/// Fixed-bin 1D histogram. Rayon-parallel, with a scalar fallback for input the
/// parallel path cannot take (see `fixed_hist_counts_par`); counts are identical
/// either way.
#[polars_expr(output_type_func = fixed_hist_output)]
fn fixed_hist(inputs: &[Series], kwargs: FixedHistKwargs) -> PolarsResult<Series> {
    polars_ensure!(kwargs.n_bins > 0, InvalidOperation: "n_bins must be greater than 0");
    let n_bins = kwargs.n_bins;
    let lo = inputs[1]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(0.0);
    let hi = inputs[2]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(1.0);
    polars_ensure!(lo <= hi, InvalidOperation: "fixed_hist: lo must be <= hi");

    let counts = fixed_hist_counts_par(&inputs[0], lo, hi, n_bins)?;
    let step = if hi > lo {
        (hi - lo) / n_bins as f64
    } else {
        0.0
    };
    let breakpoints: Vec<f64> = (0..n_bins).map(|i| lo + (i as f64 + 1.0) * step).collect();
    let out = StructChunked::from_series(
        "fixed_hist".into(),
        n_bins,
        [
            &Float64Chunked::from_vec("breakpoint".into(), breakpoints).into_series(),
            &UInt32Chunked::from_vec("count".into(), counts).into_series(),
        ]
        .into_iter(),
    )?;
    Ok(out.into_series())
}
