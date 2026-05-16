from vtsearch.media.video.clipper import VideoAutoClipper, VideoDefaultClipper, VideoSceneClipper, VideoTilingClipper
from vtsearch.media.video.media_type import VideoMediaType

MEDIA_TYPE = VideoMediaType()
CLIPPERS = [VideoAutoClipper(), VideoDefaultClipper(), VideoTilingClipper(2.0), VideoSceneClipper()]
