"""Flask blueprints for organizing application routes."""

from vtsearch.routes.medias import medias_bp
from vtsearch.routes.datasets import datasets_bp
from vtsearch.routes.detectors import detectors_bp
from vtsearch.routes.exporters import exporters_bp
from vtsearch.routes.label_importers import label_importers_bp
from vtsearch.routes.main import main_bp
from vtsearch.routes.processor_importers import processor_importers_bp
from vtsearch.routes.settings import settings_bp
from vtsearch.routes.sorting import sorting_bp
from vtsearch.routes.trainable_models import trainable_models_bp

__all__ = [
    "main_bp",
    "medias_bp",
    "sorting_bp",
    "detectors_bp",
    "datasets_bp",
    "exporters_bp",
    "label_importers_bp",
    "processor_importers_bp",
    "settings_bp",
    "trainable_models_bp",
]
