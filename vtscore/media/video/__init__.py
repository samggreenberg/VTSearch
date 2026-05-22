from vtscore.media.video.clipper import VideoAutoClipper, VideoDefaultClipper, VideoSceneClipper, VideoTilingClipper
from vtscore.media.video.media_type import VideoMediaType

MEDIA_TYPE = VideoMediaType()
CLIPPERS = [VideoAutoClipper(), VideoDefaultClipper(), VideoTilingClipper(2.0), VideoSceneClipper()]
