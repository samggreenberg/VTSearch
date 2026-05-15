"""Dataset-management route blueprints."""

from vtsearch.routes.datasets.crud import datasets_bp
from vtsearch.routes.datasets.registry import datasets_registry_bp
from vtsearch.routes.datasets.ui import datasets_ui_bp

__all__ = ["datasets_bp", "datasets_registry_bp", "datasets_ui_bp"]
