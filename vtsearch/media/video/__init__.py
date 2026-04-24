from vtsearch.media.video.clipper import VideoDefaultClipper, VideoSceneClipper, VideoTilingClipper
from vtsearch.media.video.media_type import VideoMediaType

MEDIA_TYPE = VideoMediaType()
CLIPPERS = [VideoDefaultClipper(), VideoTilingClipper(2.0), VideoSceneClipper()]
