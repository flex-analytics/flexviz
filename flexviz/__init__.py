"""flexviz — renderer-agnostic, scalable, linked visualizations."""

from importlib.metadata import version

from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import (
    Stream,
    app,
    mount_into,
    register_source,
    register_stream,
    run_server,
)
from flexviz.spec import GridItem, LayoutSpec, ToolbarConfig

__version__ = version("flexviz")

__all__ = [
    "Dashboard",
    "Figure",
    "GridItem",
    "LayoutSpec",
    "Stream",
    "ToolbarConfig",
    "app",
    "mount_into",
    "register_source",
    "register_stream",
    "run_server",
]
