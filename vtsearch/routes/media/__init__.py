"""Media-related route blueprints (listing, serving, embedding)."""

from vtsearch.routes.media.datasource import datasource_importers_bp
from vtsearch.routes.media.embed import embed_bp
from vtsearch.routes.media.list import medias_bp
from vtsearch.routes.media.seed import seed_importers_bp
from vtsearch.routes.media.server import media_server_bp

# ``medias_bp`` is defined in ``list``; these modules register the rest of its
# routes on it, so they are imported (after ``list``) for that side effect.
from vtsearch.routes.media import content, vote, pile  # noqa: F401

__all__ = [
    "datasource_importers_bp",
    "embed_bp",
    "media_server_bp",
    "medias_bp",
    "seed_importers_bp",
]
