
.PHONY: format
format:
	uv run black flexviz tests
	uv run ruff check flexviz tests

.PHONY: test
test:
	uv run pytest tests flexviz_polars/tests

.PHONY: test-browser
test-browser:
	uv run pytest -m browser -p no:randomly --override-ini="addopts=" -v -n 6

.PHONY: build-plugin
build-plugin:
	uv run maturin develop --manifest-path flexviz_polars/Cargo.toml

.PHONY: build-plugin-release
build-plugin-release:
	uv run maturin develop --release --manifest-path flexviz_polars/Cargo.toml
