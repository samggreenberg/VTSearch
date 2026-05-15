"""Dataset-management route blueprints."""

from vtsearch.routes.datasets.listings import datasets_listings_bp
from vtsearch.routes.datasets.load import datasets_load_bp
from vtsearch.routes.datasets.registry import datasets_registry_bp
from vtsearch.routes.datasets.staging import datasets_staging_bp
from vtsearch.routes.datasets.status import datasets_status_bp
from vtsearch.routes.datasets.ui import datasets_ui_bp

__all__ = [
    "datasets_listings_bp",
    "datasets_load_bp",
    "datasets_registry_bp",
    "datasets_staging_bp",
    "datasets_status_bp",
    "datasets_ui_bp",
]
