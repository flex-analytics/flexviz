"""flexviz — renderer-agnostic, scalable, linked visualizations."""

from importlib.metadata import version

from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import app, mount_into, register_source, run_server
from flexviz.spec import GridItem, LayoutSpec, ToolbarConfig

__version__ = version("flexviz")


def __getattr__(name: str):
    # Lazy, so only a database source pays for importing SQLGlot.
    if name == "SQLSource":
        from flexviz.sql import SQLSource

        return SQLSource
    raise AttributeError(f"module 'flexviz' has no attribute {name!r}")


# SQLSource is left out: `from flexviz import *` must work without the sql extra.
__all__ = [
    "Dashboard",
    "Figure",
    "GridItem",
    "LayoutSpec",
    "ToolbarConfig",
    "app",
    "mount_into",
    "register_source",
    "run_server",
]
