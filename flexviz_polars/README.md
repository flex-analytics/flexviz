# flexviz-polars

Rust [Polars](https://pola.rs) expression kernels for
[FlexViz](https://github.com/flex-analytics/flexviz).

This is an internal component. **You probably want `pip install flexviz`**,
which depends on this package and installs it for you.

Importing it registers a `flexviz` namespace on `pl.Expr`:

```python
import polars as pl
import flexviz_polars  # registers pl.Expr.flexviz

pl.select(pl.lit([1.0, 5.0, 2.0]).flexviz.arg_min_max(2))
```

The namespace exposes `every_nth`, `arg_min_max`, `fpcs`, `fixed_hist`,
`fixed_hist2d`, `fixed_hist2d_reduce`, and `fixed_line_envelope2d`: the
downsampling and fixed-bin binning kernels behind FlexViz's line, histogram,
and heatmap paths.

## License

[Apache-2.0](LICENSE) © Flex Analytics BV
