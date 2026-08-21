"""Adapter base class and helpers.

Renderer-specific adapters (PlotlyAdapter, EchartsAdapter) are imported lazily
to avoid hard dependencies on optional libraries.
"""

from .base import AbstractAdapter
from .registry import (
    build_adapter,
    get_renderer_definition,
    normalize_renderer_name,
    supported_renderers,
    validate_dashboard_renderer,
)

__all__ = [
    "AbstractAdapter",
    "build_adapter",
    "get_renderer_definition",
    "normalize_renderer_name",
    "supported_renderers",
    "validate_dashboard_renderer",
]
