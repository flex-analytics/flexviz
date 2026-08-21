"""flexviz — renderer-agnostic, scalable, linked visualizations."""

from flexviz.figure import Figure
from flexviz.dashboard import Dashboard
from flexviz.server import app, register_source, mount_into

__all__ = ["Figure", "Dashboard", "app", "register_source", "mount_into"]
