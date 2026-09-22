"""Seeded data files for the skill eval cases.

Each case builds its own file from its own seed, so two cases never hide the
same anomaly. ``make_sensors`` also returns the truths the checks compare an
agent answer against. It reads them back from the written file, because that
file is what the agent sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

_BURST = 20.0  # offset of the burst, large enough to stand out in any plot


def make_sensors(
    path: str | Path,
    rows: int = 1_000_000,
    seed: int = 0,
    anomaly_column: str = "sensor_3",
    window: tuple[float, float] = (0.60, 0.65),
) -> dict:
    """Write ``ts`` plus 8 float sensors, with a burst in one sensor.

    ``window`` gives the burst as start and end fractions of the row range.
    Returns the burst column, the time range it covers, and its mean.
    """
    rng = np.random.default_rng(seed)
    lo, hi = int(rows * window[0]), int(rows * window[1])
    sensors = {f"sensor_{i}": rng.standard_normal(rows) for i in range(8)}
    sensors[anomaly_column][lo:hi] += _BURST
    ts = pl.datetime(2024, 1, 1) + pl.duration(seconds=pl.int_range(rows))
    frame = pl.select(ts=ts, **{n: pl.Series(v) for n, v in sensors.items()})
    frame.write_parquet(path)
    burst = (
        pl.scan_parquet(path)
        .with_row_index()
        .filter(pl.col("index").is_between(lo, hi - 1))
        .select(
            start=pl.col("ts").min(),
            end=pl.col("ts").max(),
            mean=pl.col(anomaly_column).mean(),
        )
        .collect()
    )
    return {
        "column": anomaly_column,
        "window_start": burst["start"][0],
        "window_end": burst["end"][0],
        "mean_in_window": burst["mean"][0],
    }


def make_events(path: str | Path, rows: int = 500_000, seed: int = 0) -> None:
    """Write a CSV with three categorical and two numeric columns."""
    rng = np.random.default_rng(seed)
    pl.DataFrame(
        {
            "region": rng.choice(["emea", "apac", "amer"], rows),
            "device": rng.choice([f"dev_{i:02d}" for i in range(40)], rows),
            "status": rng.choice(["ok", "warn", "fail"], rows),
            "latency_ms": rng.gamma(2.0, 20.0, rows),
            "payload_bytes": rng.integers(0, 1_000_000, rows),
        }
    ).write_csv(path)


def make_tiny(path: str | Path) -> None:
    """Write 50 rows: the negative control, small enough for an inline plot."""
    rng = np.random.default_rng(0)
    pl.DataFrame(
        {"t": range(50), "value": rng.standard_normal(50)},
    ).write_csv(path)
