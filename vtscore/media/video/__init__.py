from vtscore.media.video.cleaner import VideoBlankTrimCleaner, VideoLetterboxCropCleaner
from vtscore.media.video.clipper import VideoAutoClipper, VideoDefaultClipper, VideoSceneClipper, VideoTilingClipper
from vtscore.media.video.media_type import VideoMediaType

MEDIA_TYPE = VideoMediaType()
CLIPPERS = [VideoAutoClipper(), VideoDefaultClipper(), VideoTilingClipper(2.0), VideoSceneClipper()]
#: Registration order is run order.  Cropping first means the blank-frame scan
#: measures the region that will actually be embedded (bars are near-black, and
#: they would otherwise drag every frame's blank share up), so the pair is
#: order-insensitive in practice: each gate reads the other's metadata.
CLEANERS = [VideoLetterboxCropCleaner(), VideoBlankTrimCleaner()]
