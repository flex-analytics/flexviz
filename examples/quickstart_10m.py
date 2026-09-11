"""Compact FlexViz quickstart on 10M rows. Run: python quickstart_10m_compact.py"""

import numpy as np
import polars as pl

from flexviz import Dashboard

# generate a 10M-row time series with a small burst of outliers
n = 10_000_000
value = np.sin(np.arange(n) / 5e4) + np.random.default_rng(0).standard_normal(n) * 0.05
value[6_000_000:6_050_000] += 3.0  # a 0.5% burst
ts = pl.datetime(2024, 1, 1) + pl.duration(milliseconds=pl.int_range(n) * 10)
df = pl.select(timestamp=ts, value=pl.Series(value))

# create a dashboard with a line plot and a histogram, and show it
dash = Dashboard(df, cache=True)
dash.add_figure(title="value").add_line(x="timestamp", y="value", n_points=2000)
dash.add_figure(title="distribution").add_histogram(x="value", bins=60)
dash.show()
