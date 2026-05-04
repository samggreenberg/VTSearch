"""Flask blueprints for organizing application routes."""

from vtsearch.routes.auth import auth_bp
from vtsearch.routes.eval import eval_bp
from vtsearch.routes.file_browser import file_browser_bp
from vtsearch.routes.labels import labels_bp
from vtsearch.routes.media_server import media_server_bp
from vtsearch.routes.medias import medias_bp
from vtsearch.routes.datasets import datasets_bp
from vtsearch.routes.datasets_registry import datasets_registry_bp
from vtsearch.routes.detectors import detectors_bp
from vtsearch.routes.exporters import exporters_bp
from vtsearch.routes.label_importers import label_importers_bp
from vtsearch.routes.main import main_bp
from vtsearch.routes.models_registry import models_registry_bp
from vtsearch.routes.processor_importers import processor_importers_bp
from vtsearch.routes.settings import settings_bp
from vtsearch.routes.settings_io import settings_io_bp
from vtsearch.routes.sorting import sorting_bp
from vtsearch.routes.sync_sources import sync_sources_bp
from vtsearch.routes.trainable_models import trainable_models_bp

__all__ = [
    "auth_bp",
    "eval_bp",
    "file_browser_bp",
    "labels_bp",
    "media_server_bp",
    "main_bp",
    "medias_bp",
    "sorting_bp",
    "sync_sources_bp",
    "detectors_bp",
    "datasets_bp",
    "datasets_registry_bp",
    "exporters_bp",
    "label_importers_bp",
    "models_registry_bp",
    "processor_importers_bp",
    "settings_bp",
    "settings_io_bp",
    "trainable_models_bp",
]
