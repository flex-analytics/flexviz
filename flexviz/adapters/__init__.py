"""Adapter base class and helpers.

Renderer-specific adapters (PlotlyAdapter) are imported lazily
to avoid hard dependencies on optional libraries.
"""

from .base import AbstractAdapter
from .registry import (
    build_adapter,
    get_renderer_definition,
    supported_renderers,
    validate_dashboard_renderer,
)

__all__ = [
    "AbstractAdapter",
    "build_adapter",
    "get_renderer_definition",
    "supported_renderers",
    "validate_dashboard_renderer",
]
