"""Media-related route blueprints (listing, serving, embedding)."""

from vtsearch.routes.media.datasource import datasource_importers_bp
from vtsearch.routes.media.embed import embed_bp
from vtsearch.routes.media.list import medias_bp
from vtsearch.routes.media.server import media_server_bp

__all__ = ["datasource_importers_bp", "embed_bp", "media_server_bp", "medias_bp"]
