"""Video cleaners - 1→1 cleanup gates run on each video unit before embedding.

Both gates here are **metadata-only**, and deliberately so.  A video unit is a
``(parent bytes, time window)`` pair: every clip of a tiled video shares the
parent's payload and says which slice of it it is via ``clip_start`` /
``clip_end``.  Re-encoding a cleaned copy per unit would duplicate that payload
once per tile *and* desync the window, which still indexes the parent's
timeline.  So these cleaners narrow the window (:class:`VideoBlankTrimCleaner`)
and record a pixel region (:class:`VideoLetterboxCropCleaner`) instead of
rewriting bytes, and the readers honour both:
:func:`~vtscore.media.video._frame_sampling.sample_video_frames` for the
embedding and the thumbnailers for the grid preview.

Two consequences of staying metadata-only, both good: nothing is snapshotted
under the ``original_*`` keys (there is no rewritten payload to keep an
"Original" of - the served file is untouched, and the player already loops
within ``[clip_start, clip_end]``), so a cleaned video costs no extra storage
and a reference (thin) import keeps its byte savings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.media.cleaner import MediaCleaner
from vtscore.media.image.edge_trim import (
    DEFAULT_EDGE_TOL,
    DEFAULT_MAX_EDGE_TRIM,
    DEFAULT_MIN_EDGE_TRIM,
    solid_edge_box,
)
from vtscore.media.video import decode
from vtscore.media.video._frame_sampling import resolve_video_path
from vtscore.media.video.crop import clamp_clip_box, crop_frame

log = logging.getLogger(__name__)

#: Don't rewrite a unit's window for a trim this small (seconds, summed over
#: both ends).  Shaving a few frames buys the embedder nothing and only makes
#: the item look edited.
_MIN_TRIM_SECONDS = 0.1

#: Never leave a unit shorter than this (seconds).  Guards the degenerate case
#: where a clip is blank end to end.
_MIN_SPAN_SECONDS = 0.1

#: Hard ceiling on frame probes per end for the blank scan.  Each probe is one
#: ffmpeg seek+decode, so the scan step is coarsened on a long unit rather than
#: letting the probe count grow with its duration.
_MAX_PROBES_PER_END = 16

#: Row/column stride used when measuring how blank a frame is.  A blank frame is
#: blank at any sampling density, and this keeps the check ~16x cheaper than a
#: full-resolution pass.
_BLANK_STRIDE = 4

#: Dominant-tone share at which :class:`VideoLetterboxCropCleaner` writes a
#: frame off as wholly blank (a fade, a leader) and skips it rather than letting
#: it veto the crop.
_BLANK_FRAME_SHARE = 0.99


def _unit_window(media: dict[str, Any], info: decode.VideoInfo) -> tuple[float, float]:
    """Return the ``[start, end]`` seconds of the video this unit covers.

    Falls back to the whole container when the unit carries no usable
    boundaries (an untiled video, or a clipper that recorded none).
    """
    total = info.duration
    start = media.get("clip_start")
    end = media.get("clip_end")
    try:
        t0 = max(0.0, min(float(start), total)) if start is not None else 0.0
        t1 = max(0.0, min(float(end), total)) if end is not None else total
    except (TypeError, ValueError):
        return 0.0, total
    if t1 <= t0:
        return 0.0, total
    return t0, t1


def _blank_share(frame_rgb: np.ndarray, tone_tol: float) -> float:
    """Return the share of *frame_rgb* taken by its dominant flat tone.

    A frame counts as blank when one tone - near-black or near-white within
    *tone_tol* - covers almost all of it: a fade-to-black frame, a black
    leader, a white flash, an empty card.  Taking the larger of the two shares
    rather than their sum is what keeps genuinely high-contrast content (black
    text on a white page, a monochrome line drawing) from reading as blank.
    """
    small = frame_rgb[::_BLANK_STRIDE, ::_BLANK_STRIDE]
    if small.size == 0:
        return 0.0
    near_black = float((small.max(axis=2) <= tone_tol).mean())
    near_white = float((small.min(axis=2) >= 255 - tone_tol).mean())
    return max(near_black, near_white)


class VideoBlankTrimCleaner(MediaCleaner):
    """Trim leading and trailing blank frames off a video unit.

    The video analog of
    :class:`~vtscore.media.audio.cleaner.AudioSilenceTrimCleaner`: a clip that
    opens on a second of black before the fade-in, or ends on a blank tail
    card, spends that share of the embedder's fixed frame budget on nothing.
    This gate walks in from each end while the frames are blank and narrows the
    unit's ``clip_start`` / ``clip_end`` to the first and last frame with
    content in it.  Blank frames *inside* the clip are left alone - a cut to
    black mid-scene is part of the content's rhythm.

    Nothing is re-encoded: the window is metadata the embedder's frame sampler
    and the player both already honour, so a trimmed unit still serves the
    parent's bytes and the player simply loops within the tighter span.

    Costs one frame probe per end in the common case (the first frame has
    content, so the scan stops immediately) and at most
    :data:`_MAX_PROBES_PER_END` when it doesn't.

    Parameters
    ----------
    blank_ratio:
        Share of a frame that must be one flat tone (near-black or near-white)
        for the frame to count as blank.  Defaults to 0.99.
    max_trim:
        Never trim more than this fraction of the unit's duration off either
        end, so a mostly-blank clip can't collapse to a sliver.  Defaults
        to 0.25.
    step:
        Seconds between frame probes while scanning in from an end.  Also the
        resolution of the cut - up to one step of blank can survive, because
        the trim never crosses the first frame seen with content in it.
        Coarsened automatically on a long unit so the probe count stays
        bounded.  Defaults to 0.25.
    """

    def __init__(self, blank_ratio: float = 0.99, max_trim: float = 0.25, step: float = 0.25) -> None:
        if not 0 < blank_ratio <= 1:
            raise ValueError("blank_ratio must be in (0, 1]")
        if not 0 <= max_trim < 0.5:
            raise ValueError("max_trim must be in [0, 0.5)")
        if step <= 0:
            raise ValueError("step must be positive")
        self._blank_ratio = blank_ratio
        self._max_trim = max_trim
        self._step = step

    @property
    def name(self) -> str:
        return "video_blank_trim"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def display_name(self) -> str:
        return "Blank Frame Trim"

    @property
    def description(self) -> str:
        return (
            "Trim leading and trailing blank frames - black leader, fade-ins, empty tail cards - so the "
            "embedder's frame budget is spent on content. Blank frames inside the clip are kept."
        )

    @property
    def summary_template(self) -> str:
        return "Trim head/tail frames that are {blank_ratio} one flat tone, up to {max_trim} of each end."

    @property
    def blank_ratio(self) -> float:
        return self._blank_ratio

    @property
    def max_trim(self) -> float:
        return self._max_trim

    @property
    def step(self) -> float:
        return self._step

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "blank_ratio",
                "label": "Blank threshold",
                "description": (
                    "Share of a frame that must be one flat tone (near-black or near-white) for it to "
                    "count as blank. Lower values treat busier frames as blank - more gets trimmed."
                ),
                "type": "number",
                "default": self._blank_ratio,
                "min": 0.5,
                "max": 1,
                "step": 0.01,
            },
            {
                "key": "max_trim",
                "label": "Maximum trim per end",
                "description": (
                    "Never trim more than this fraction of the clip's duration off either end, so a "
                    "mostly-blank clip can't collapse to a sliver."
                ),
                "type": "number",
                "default": self._max_trim,
                "min": 0,
                "max": 0.45,
                "step": 0.05,
            },
            {
                "key": "step",
                "label": "Scan step (seconds)",
                "description": (
                    "Seconds between frames while scanning in from each end, and the resolution of the "
                    "cut. Coarsened automatically on long clips so the number of probes stays bounded."
                ),
                "type": "number",
                "default": self._step,
                "min": 0.02,
                "max": 5,
                "step": 0.01,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "VideoBlankTrimCleaner":
        return VideoBlankTrimCleaner(
            blank_ratio=float(params.get("blank_ratio", self._blank_ratio)),
            max_trim=float(params.get("max_trim", self._max_trim)),
            step=float(params.get("step", self._step)),
        )

    def _scan_in(
        self,
        path: Path,
        *,
        origin: float,
        budget: float,
        direction: int,
        box: Any,
    ) -> float:
        """Return how far in from *origin* the blank run reaches (seconds).

        Walks *direction* (+1 forward from the head, -1 back from the tail) in
        steps while every probed frame is blank, stopping at *budget* seconds.
        A frame that fails to decode ends the run: an unreadable frame is not
        evidence of blankness.

        Returns the last offset actually *seen* blank, never the first offset
        seen with content: the cut therefore lands on a scan step and can leave
        up to one step of blank standing, which is the right way round.
        Trimming to the content probe instead would delete up to a step of real
        picture between the two, and a gate must not eat content to tidy an
        edge.  A caller who wants a tighter cut lowers ``step``.
        """
        if budget <= 0:
            return 0.0
        step = max(self._step, budget / _MAX_PROBES_PER_END)
        blank = 0.0
        offset = 0.0
        while offset <= budget:
            frame = decode.frame_at(path, origin + direction * offset)
            if frame is None:
                break
            if _blank_share(crop_frame(frame, box), DEFAULT_EDGE_TOL) < self._blank_ratio:
                # Content: everything from here in stays, including the gap back
                # to the last blank probe.
                return blank
            blank = offset
            offset += step
        # Blank as far as we looked (or as far as the frames decoded).
        return min(offset, budget)

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* with its blank head/tail trimmed off, or unchanged.

        Unchanged when the video can't be resolved or probed, the unit is
        already shorter than a trim would leave it, no blank run reaches
        :data:`_MIN_TRIM_SECONDS`, or the frames simply have content at both
        ends - the common case, which costs two frame probes.
        """
        with resolve_video_path(media) as path:
            if path is None:
                return media
            info = decode.probe(path)
            if info is None or info.duration <= 0:
                return media

            t0, t1 = _unit_window(media, info)
            span = t1 - t0
            if span <= _MIN_SPAN_SECONDS:
                return media

            box = media.get("clip_box")
            budget = span * self._max_trim
            # The tail probe starts one frame short of the end, where a seek
            # still lands on a real frame.
            head = self._scan_in(path, origin=t0, budget=budget, direction=1, box=box)
            tail_origin = max(t0, t1 - info.frame_seconds)
            tail = self._scan_in(path, origin=tail_origin, budget=budget, direction=-1, box=box)

        if head + tail < _MIN_TRIM_SECONDS:
            return media
        new_t0, new_t1 = t0 + head, t1 - tail
        if new_t1 - new_t0 < _MIN_SPAN_SECONDS:
            return media

        cleaned = dict(media)
        cleaned["clip_start"] = round(new_t0, 6)
        cleaned["clip_end"] = round(new_t1, 6)
        cleaned["duration"] = round(new_t1 - new_t0, 6)
        return cleaned


class VideoLetterboxCropCleaner(MediaCleaner):
    """Crop away a video's letterbox bars and pillarbox margins.

    A 4:3 broadcast re-published in a 16:9 container, a phone clip padded to
    landscape, a slide capture framed in black: the bars are content-free, and
    the embedder spends real capacity encoding "these edges are a black
    rectangle" while the padding shrinks the subject inside its fixed input
    window.  This gate samples a handful of frames across the unit, asks the
    same detector the image gate uses
    (:func:`~vtscore.media.image.edge_trim.solid_edge_box`) where the content
    starts in each, and records the **union** of those content boxes as the
    unit's ``clip_box``.

    Taking the union rather than the intersection is what makes the crop safe
    on moving pictures: a margin is removed only when *every* sampled frame
    agrees it is blank, so a subject that drifts into the top of frame
    mid-clip keeps its room.  A frame with content all the way to its edges
    aborts the crop outright; a wholly blank frame (a fade, a black leader)
    is skipped, since it says nothing about where the bars are.

    Nothing is re-encoded - the box is honoured by the embedder's frame
    sampler and by the thumbnailers, so a cropped unit's preview and its
    embedding frame the same picture.

    Parameters
    ----------
    samples:
        How many frames to analyse across the unit.  Defaults to 5.
    edge_tol:
        How far from pure white or pure black a pixel may sit and still count
        as bar.  Defaults to :data:`~vtscore.media.image.edge_trim.DEFAULT_EDGE_TOL`.
    max_edge_trim:
        Never crop more than this fraction off any single side.  Defaults to
        :data:`~vtscore.media.image.edge_trim.DEFAULT_MAX_EDGE_TRIM`.
    min_edge_trim:
        Leave the unit uncropped when every margin is thinner than this
        fraction.  Defaults to
        :data:`~vtscore.media.image.edge_trim.DEFAULT_MIN_EDGE_TRIM`.
    """

    def __init__(
        self,
        samples: int = 5,
        edge_tol: float = DEFAULT_EDGE_TOL,
        max_edge_trim: float = DEFAULT_MAX_EDGE_TRIM,
        min_edge_trim: float = DEFAULT_MIN_EDGE_TRIM,
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self._samples = int(samples)
        self._edge_tol = edge_tol
        self._max_edge_trim = max_edge_trim
        self._min_edge_trim = min_edge_trim

    @property
    def name(self) -> str:
        return "video_letterbox_crop"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def display_name(self) -> str:
        return "Letterbox Crop"

    @property
    def description(self) -> str:
        return (
            "Crop letterbox bars and pillarbox margins off each clip, using the tightest box every "
            "sampled frame agrees is content, so the embedder sees the picture instead of its padding."
        )

    @property
    def summary_template(self) -> str:
        return "Crop bars agreed on by {samples} sampled frames, never more than {max_edge_trim} of a side."

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def edge_tol(self) -> float:
        return self._edge_tol

    @property
    def max_edge_trim(self) -> float:
        return self._max_edge_trim

    @property
    def min_edge_trim(self) -> float:
        return self._min_edge_trim

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "samples",
                "label": "Frames sampled",
                "description": (
                    "How many frames across the clip to analyse. More frames make the crop safer on "
                    "moving content and cost one decode each."
                ),
                "type": "number",
                "default": self._samples,
                "min": 1,
                "max": 32,
                "step": 1,
            },
            {
                "key": "edge_tol",
                "label": "Solid tolerance (0-255)",
                "description": (
                    "How far from pure white or pure black a pixel may sit and still count as bar. "
                    "Higher values treat more of the border as padding."
                ),
                "type": "number",
                "default": self._edge_tol,
                "min": 0,
                "max": 64,
                "step": 1,
            },
            {
                "key": "max_edge_trim",
                "label": "Maximum crop per side",
                "description": (
                    "Never remove more than this fraction of the width or height from any single side, "
                    "so a small subject in a large blank field can't blow up to fill the frame."
                ),
                "type": "number",
                "default": self._max_edge_trim,
                "min": 0,
                "max": 0.5,
                "step": 0.05,
            },
            {
                "key": "min_edge_trim",
                "label": "Minimum crop to bother",
                "description": "Leave the clip uncropped when every margin is thinner than this fraction.",
                "type": "number",
                "default": self._min_edge_trim,
                "min": 0,
                "max": 0.5,
                "step": 0.01,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "VideoLetterboxCropCleaner":
        return VideoLetterboxCropCleaner(
            samples=int(float(params.get("samples", self._samples))),
            edge_tol=float(params.get("edge_tol", self._edge_tol)),
            max_edge_trim=float(params.get("max_edge_trim", self._max_edge_trim)),
            min_edge_trim=float(params.get("min_edge_trim", self._min_edge_trim)),
        )

    def _sample_times(self, t0: float, t1: float, info: decode.VideoInfo) -> list[float]:
        """Return the timestamps to analyse, spread across ``[t0, t1]``."""
        last = max(t0, t1 - info.frame_seconds)
        if self._samples == 1 or last <= t0:
            return [t0]
        return [float(t) for t in np.linspace(t0, last, self._samples)]

    def _content_box(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return one frame's content box, or ``None`` when it constrains nothing.

        ``None`` covers both "content reaches the edges" (there is no bar to
        crop, so the caller must abort) and "wholly blank frame" (a fade or
        leader, which the caller skips); the caller tells them apart with
        :func:`_blank_share`.
        """
        from PIL import Image  # noqa: PLC0415

        return solid_edge_box(
            Image.fromarray(frame_rgb),
            edge_tol=self._edge_tol,
            max_edge_trim=self._max_edge_trim,
            min_edge_trim=self._min_edge_trim,
        )

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* carrying the bars' crop box, or unchanged.

        Unchanged when the video can't be resolved, probed, or decoded, when
        any sampled frame carries content to its edges, when every sampled
        frame is blank, or when the agreed box crops less than
        *min_edge_trim* off every side - the common case for content that was
        never padded.
        """
        with resolve_video_path(media) as path:
            if path is None:
                return media
            info = decode.probe(path)
            if info is None or info.duration <= 0:
                return media
            t0, t1 = _unit_window(media, info)
            prior = media.get("clip_box")
            frames = decode.frames_at(path, self._sample_times(t0, t1, info))

        if not frames:
            return media

        boxes: list[tuple[int, int, int, int]] = []
        for frame in frames:
            # Analyse the region this unit actually embeds, so an earlier crop
            # composes with this one instead of being measured through.
            region = crop_frame(frame, prior)
            box = self._content_box(region)
            if box is None:
                if _blank_share(region, self._edge_tol) >= _BLANK_FRAME_SHARE:
                    continue  # a fade or leader frame says nothing about the bars
                return media  # content reaches the edges: nothing to crop
            boxes.append(box)

        if not boxes:
            return media  # every sampled frame was blank

        union = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        full_height, full_width = frames[0].shape[:2]
        height, width = crop_frame(frames[0], prior).shape[:2]
        if clamp_clip_box(union, width, height) is None:
            return media  # the frames agree on the whole region; no bars
        margins = (union[0] / width, union[1] / height, 1.0 - union[2] / width, 1.0 - union[3] / height)
        if max(margins) < self._min_edge_trim:
            return media  # negligible on every side once the frames are combined

        # The box is stored in *source frame* coordinates, so an earlier crop's
        # origin has to be added back onto the region-relative union.
        base = clamp_clip_box(prior, full_width, full_height)
        dx, dy = (base[0], base[1]) if base is not None else (0, 0)
        cleaned = dict(media)
        cleaned["clip_box"] = [union[0] + dx, union[1] + dy, union[2] + dx, union[3] + dy]
        return cleaned
