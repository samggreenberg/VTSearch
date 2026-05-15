"""Label route blueprints (voting, importers, exporters)."""

from vtsearch.routes.labels.exporters import exporters_bp
from vtsearch.routes.labels.importers import label_importers_bp
from vtsearch.routes.labels.vote import labels_bp

__all__ = ["exporters_bp", "label_importers_bp", "labels_bp"]
