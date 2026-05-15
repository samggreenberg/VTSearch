"""Settings route blueprints (CRUD, import/export, sync sources)."""

from vtsearch.routes.settings.api import settings_bp
from vtsearch.routes.settings.io import settings_io_bp
from vtsearch.routes.settings.sources import sync_sources_bp

__all__ = ["settings_bp", "settings_io_bp", "sync_sources_bp"]
