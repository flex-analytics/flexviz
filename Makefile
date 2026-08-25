
# Every Python source tree in the repo. `flexviz_polars` covers both the plugin's
# namespace module and its tests, which `make test` already runs.
PY_SOURCES := flexviz tests flexviz_polars

.PHONY: format
format:
	uv run black $(PY_SOURCES)
	uv run ruff check $(PY_SOURCES)

.PHONY: test
test:
	uv run pytest tests flexviz_polars/tests

.PHONY: test-browser
test-browser:
	uv run pytest -m browser -p no:randomly --override-ini="addopts=" -v -n 6

.PHONY: docs
docs:
	uv run --group docs mkdocs serve

.PHONY: build-plugin
build-plugin:
	uv run maturin develop --features nightly --manifest-path flexviz_polars/Cargo.toml

.PHONY: build-plugin-release
build-plugin-release:
	uv run maturin develop --release --features nightly --manifest-path flexviz_polars/Cargo.toml
