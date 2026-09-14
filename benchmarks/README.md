# Benchmarks

Continuous performance tracking for FlexViz, measured by
[CodSpeed](https://app.codspeed.io/flex-analytics/flexviz) on every push and
pull request.

The suite covers the paths a live dashboard spends its time in:

| File                   | What it measures                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `test_traces.py`       | One aggregation per trace type (line downsamplers, histograms, box, bar, pie, treemap, correlation, geo) |
| `test_engine_events.py`| `FlexEngine.process` for init, zoom, and cross-filter brushes on a two-figure dashboard  |
| `test_cube.py`         | Cross-filter cube construction, binary encoding, and cache keying                        |
| `test_query_layer.py`  | `LFQueryBuilder` domain probes and batched aggregation, predicate compilation            |
| `test_server.py`       | `POST /update` and `POST /dashboard/update` end to end through the FastAPI app           |

These are not correctness tests: they are excluded from `make test` (pytest's
`testpaths` points at `tests/`) and each one asserts only that the call
produced a plausible result.

## Running them

```bash
make bench              # quick local timings
make bench-simulation   # CPU-simulated, same measurement as CI (needs the CodSpeed CLI)
```

Both need the Rust kernels built (`uv sync` or `make build-plugin-release`),
since every data path here goes through `flexviz_polars`.

## Adding a benchmark

Take the `benchmark` fixture, call the thing once, and keep the fixture data in
`conftest.py` so the measurement covers FlexViz rather than data generation:

```python
def test_my_path(benchmark, numeric_df):
    engine, infos = build_engine(numeric_df, [LinePlot(x="ts", y="val")])
    deltas = benchmark(engine.process, INIT, infos)
    assert len(deltas) == 1
```

Sizes are kept at 200k rows: large enough that the Polars and Rust work
dominates Python overhead, small enough that a simulated run stays quick.
