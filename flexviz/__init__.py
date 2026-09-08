"""flexviz — renderer-agnostic, scalable, linked visualizations."""

from importlib.metadata import version

from flexviz.dashboard import Dashboard
from flexviz.figure import Figure
from flexviz.server import app, mount_into, register_source

__version__ = version("flexviz")

__all__ = ["Dashboard", "Figure", "app", "mount_into", "register_source"]
