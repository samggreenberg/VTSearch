from vtsearch.media.video.clipper import VideoDefaultClipper, VideoSceneClipper, VideoTilingClipper
from vtsearch.media.video.embedder import VideoXClipEmbedder
from vtsearch.media.video.embedder_languagebind import VideoLanguageBindEmbedder
from vtsearch.media.video.media_type import VideoMediaType

MEDIA_TYPE = VideoMediaType()
EMBEDDERS = [VideoXClipEmbedder(), VideoLanguageBindEmbedder()]
CLIPPERS = [VideoDefaultClipper(), VideoTilingClipper(2.0), VideoSceneClipper()]
