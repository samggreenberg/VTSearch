"""Synthetic media generators for offline demo datasets.

These generators produce small, deterministic datasets of fake media
(images, audio, video) so users can exercise VTSearch without an internet
connection or downloading public datasets. Each ``generate_<type>_dataset``
function writes ``count`` files into ``output_dir``, cycling through several
"ideas" (e.g. smiley faces, shapes; tones, chords, drums, rain, wind, birds)
so the resulting dataset has enough semantic variety to make sorting demos
interesting.

Generation is deterministic given a seed: the same (count, seed) produces
the same files, so the importer can cache results across reloads.
"""

from vtscore.utils.synthetic.audio import generate_audio_dataset
from vtscore.utils.synthetic.images import generate_image_dataset
from vtscore.utils.synthetic.video import generate_video_dataset

__all__ = [
    "generate_audio_dataset",
    "generate_image_dataset",
    "generate_video_dataset",
]
