"""Flask blueprints for organizing application routes."""

from vtsearch.routes.achievements import achievements_bp
from vtsearch.routes.auth import auth_bp
from vtsearch.routes.datasets import (
    datasets_listings_bp,
    datasets_load_bp,
    datasets_registry_bp,
    datasets_staging_bp,
    datasets_status_bp,
    datasets_ui_bp,
)
from vtsearch.routes.detectors import (
    detector_find_bp,
    detector_scoring_bp,
    detectors_crud_bp,
    detectors_labels_bp,
    detectors_registry_bp,
)
from vtsearch.routes.eval import eval_bp
from vtsearch.routes.events import events_bp
from vtsearch.routes.file_browser import file_browser_bp
from vtsearch.routes.labels import exporters_bp, label_importers_bp, labels_bp
from vtsearch.routes.main import main_bp
from vtsearch.routes.media import embed_bp, media_server_bp, medias_bp
from vtsearch.routes.metrics import metrics_bp
from vtsearch.routes.processors import processors_bp
from vtsearch.routes.settings import settings_bp, settings_io_bp, sync_sources_bp
from vtsearch.routes.sorting import sorting_bp

__all__ = [
    "achievements_bp",
    "auth_bp",
    "datasets_listings_bp",
    "datasets_load_bp",
    "datasets_registry_bp",
    "datasets_staging_bp",
    "datasets_status_bp",
    "datasets_ui_bp",
    "detector_find_bp",
    "detector_scoring_bp",
    "detectors_crud_bp",
    "detectors_labels_bp",
    "detectors_registry_bp",
    "embed_bp",
    "eval_bp",
    "events_bp",
    "exporters_bp",
    "file_browser_bp",
    "label_importers_bp",
    "labels_bp",
    "main_bp",
    "media_server_bp",
    "medias_bp",
    "metrics_bp",
    "processors_bp",
    "settings_bp",
    "settings_io_bp",
    "sorting_bp",
    "sync_sources_bp",
]
