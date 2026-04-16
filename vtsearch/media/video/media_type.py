"""Video media type — MP4/AVI/MOV/WEBM/MKV files."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    demo_slice,
)

_THUMB_SIZE = 128


def generate_video_thumbnail(video_bytes: bytes, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Extract the middle frame of a video and return it as a PNG thumbnail.

    Uses OpenCV to decode the video and PIL to produce the PNG.  Returns
    ``None`` if the video cannot be decoded or has no frames.
    """
    try:
        import cv2  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except Exception:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        try:
            if not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return None
            mid = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                return None
        finally:
            cap.release()
    finally:
        import os  # noqa: PLC0415

        os.unlink(tmp.name)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_video_thumbnail_from_file(file_path: Path, *, size: int = _THUMB_SIZE) -> bytes | None:
    """Extract the middle frame of a video file and return it as a PNG thumbnail."""
    try:
        import cv2  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except Exception:
        return None

    try:
        cap = cv2.VideoCapture(str(file_path))
        try:
            if not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return None
            mid = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                return None
        finally:
            cap.release()
    except Exception:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_VIDEO_MIME_TYPES: dict[str, str] = {
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


class VideoMediaType(MediaType):
    """Handles video medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.video.embedder.VideoXClipEmbedder`.
    """

    def __init__(self) -> None:
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "video"

    @property
    def name(self) -> str:
        return "Video"

    @property
    def icon(self) -> str:
        return "video"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.mp4", "*.avi", "*.mov", "*.webm", "*.mkv"]

    @property
    def folder_import_name(self) -> str:
        return "video"

    @property
    def tab_title(self) -> str:
        return "Videos"

    @property
    def dir_key(self) -> str:
        return "video_dir"

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        dur = media.get("duration")
        if dur and dur > 0:
            result["Duration"] = dur
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    _DEMO_CATEGORIES = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    ]

    _HMDB51_CATEGORIES = [
        "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
        "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing",
        "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball",
        "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup",
        "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow",
        "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
        "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk",
        "wave",
    ]

    _UCF101_FULL_CATEGORIES = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
        "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
        "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
        "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
        "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
        "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
        "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
        "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
        "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
        "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
        "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
        "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
        "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
        "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
        "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
        "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
        "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
        "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups",
        "WritingOnBoard", "YoYo",
    ]

    _KTH_CATEGORIES = [
        "boxing", "handclapping", "handwaving", "jogging", "running", "walking",
    ]

    @property
    def demo_datasets(self) -> list:
        from vtsearch.datasets.downloader import (  # noqa: PLC0415
            HMDB51_DOWNLOAD_SIZE_MB,
            KTH_DOWNLOAD_SIZE_MB,
            UCF101_FULL_DOWNLOAD_SIZE_MB,
            UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            VIDEO_DIR,
        )

        cats = self._DEMO_CATEGORIES
        folder = VIDEO_DIR / "ucf101"

        hmdb_cats = self._HMDB51_CATEGORIES
        hmdb_folder = VIDEO_DIR / "hmdb51"

        ucf_full_cats = self._UCF101_FULL_CATEGORIES
        ucf_full_folder = VIDEO_DIR / "ucf101_full"

        kth_cats = self._KTH_CATEGORIES
        kth_folder = VIDEO_DIR / "kth"

        desc = "Action recognition videos sourced from YouTube, covering sports and everyday activities."
        hmdb_desc = "Human motion clips from movies and web videos, covering 51 diverse action categories."
        ucf_full_desc = "Full UCF-101 with all 101 action classes — sports, instruments, daily activities."
        kth_desc = "Simple human actions in controlled settings — a classic action recognition benchmark."
        return [
            # UCF-101 subset (10 classes, 171 MB download)
            DemoDataset(
                id="ucf101_s", label="UCF-101 (S)",
                description=desc,
                categories=cats, source="ucf101", required_folder=folder,
                slice_frac_start=0.0, slice_frac_end=1 / 7,
                items_per_category=132, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_m", label="UCF-101 (M)",
                description=desc,
                categories=cats, source="ucf101", required_folder=folder,
                slice_frac_start=1 / 7, slice_frac_end=3 / 7,
                items_per_category=132, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_l", label="UCF-101 (L)",
                description=desc,
                categories=cats, source="ucf101", required_folder=folder,
                slice_frac_start=3 / 7, slice_frac_end=None,
                items_per_category=132, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_a", label="UCF-101 (A)",
                description=desc,
                categories=cats, source="ucf101", required_folder=folder,
                slice_frac_start=0.0, slice_frac_end=None,
                items_per_category=132, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
            ),
            # HMDB51 (51 action classes, ~2 GB download)
            DemoDataset(
                id="hmdb51_s", label="HMDB51 (S)",
                description=hmdb_desc,
                categories=hmdb_cats, source="hmdb51", required_folder=hmdb_folder,
                slice_frac_start=0.0, slice_frac_end=1 / 7,
                items_per_category=137, download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="hmdb51_m", label="HMDB51 (M)",
                description=hmdb_desc,
                categories=hmdb_cats, source="hmdb51", required_folder=hmdb_folder,
                slice_frac_start=1 / 7, slice_frac_end=3 / 7,
                items_per_category=137, download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="hmdb51_l", label="HMDB51 (L)",
                description=hmdb_desc,
                categories=hmdb_cats, source="hmdb51", required_folder=hmdb_folder,
                slice_frac_start=3 / 7, slice_frac_end=None,
                items_per_category=137, download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="hmdb51_a", label="HMDB51 (A)",
                description=hmdb_desc,
                categories=hmdb_cats, source="hmdb51", required_folder=hmdb_folder,
                slice_frac_start=0.0, slice_frac_end=None,
                items_per_category=137, download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
            ),
            # UCF-101 full (101 action classes, ~7 GB download)
            DemoDataset(
                id="ucf101_full_s", label="UCF-101 Full (S)",
                description=ucf_full_desc,
                categories=ucf_full_cats, source="ucf101_full", required_folder=ucf_full_folder,
                slice_frac_start=0.0, slice_frac_end=1 / 7,
                items_per_category=132, download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_full_m", label="UCF-101 Full (M)",
                description=ucf_full_desc,
                categories=ucf_full_cats, source="ucf101_full", required_folder=ucf_full_folder,
                slice_frac_start=1 / 7, slice_frac_end=3 / 7,
                items_per_category=132, download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_full_l", label="UCF-101 Full (L)",
                description=ucf_full_desc,
                categories=ucf_full_cats, source="ucf101_full", required_folder=ucf_full_folder,
                slice_frac_start=3 / 7, slice_frac_end=None,
                items_per_category=132, download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucf101_full_a", label="UCF-101 Full (A)",
                description=ucf_full_desc,
                categories=ucf_full_cats, source="ucf101_full", required_folder=ucf_full_folder,
                slice_frac_start=0.0, slice_frac_end=None,
                items_per_category=132, download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
            ),
            # KTH Actions (6 action classes, ~1.1 GB download)
            DemoDataset(
                id="kth_s", label="KTH Actions (S)",
                description=kth_desc,
                categories=kth_cats, source="kth", required_folder=kth_folder,
                slice_frac_start=0.0, slice_frac_end=1 / 7,
                items_per_category=100, download_size_mb=KTH_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="kth_m", label="KTH Actions (M)",
                description=kth_desc,
                categories=kth_cats, source="kth", required_folder=kth_folder,
                slice_frac_start=1 / 7, slice_frac_end=3 / 7,
                items_per_category=100, download_size_mb=KTH_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="kth_l", label="KTH Actions (L)",
                description=kth_desc,
                categories=kth_cats, source="kth", required_folder=kth_folder,
                slice_frac_start=3 / 7, slice_frac_end=None,
                items_per_category=100, download_size_mb=KTH_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="kth_a", label="KTH Actions (A)",
                description=kth_desc,
                categories=kth_cats, source="kth", required_folder=kth_folder,
                slice_frac_start=0.0, slice_frac_end=None,
                items_per_category=100, download_size_mb=KTH_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    _VIDEO_SOURCE_DOWNLOADERS = {
        "ucf101": "download_ucf101_subset",
        "hmdb51": "download_hmdb51",
        "ucf101_full": "download_ucf101_full",
        "kth": "download_kth",
    }

    def load_demo_source(
        self, source, categories, slice_start, slice_end, clips,
        on_progress=None, embedder=None, slice_frac_start=None, slice_frac_end=None, **kwargs,
    ):
        import hashlib  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        downloader_name = self._VIDEO_SOURCE_DOWNLOADERS.get(source)
        if downloader_name is None:
            raise ValueError(f"Unsupported video source: {source!r}")

        import vtsearch.datasets.downloader as dl_module  # noqa: PLC0415
        from vtsearch.datasets.loader import load_video_metadata_from_folders  # noqa: PLC0415

        download_fn = getattr(dl_module, downloader_name)
        video_dir = download_fn(on_progress=on_progress)
        metadata = load_video_metadata_from_folders(video_dir, categories)

        by_cat: dict[str, list] = {}
        for fname, meta in sorted(metadata.items()):
            cat = meta["category"]
            by_cat.setdefault(cat, []).append((meta["path"], meta))

        video_files: list[tuple] = []
        for cat in categories:
            video_files.extend(demo_slice(
                by_cat.get(cat, []), slice_start, slice_end, slice_frac_start, slice_frac_end,
            ))

        if getattr(embedder, "_model", None) is None:
            on_progress("loading", "Loading video embedding model\u2026", 0, 0)
            original_cb = embedder._on_progress
            embedder._on_progress = on_progress
            try:
                embedder.load_models()
            finally:
                embedder._on_progress = original_cb

        clip_id = max(clips.keys(), default=0) + 1
        total = len(video_files)
        on_progress("embedding", f"Starting embedding for {total} video files...", 0, total)
        demo_origin_template: dict = {"importer": "demo", "params": {}}

        for i, (video_path, meta) in enumerate(video_files):
            rel_name = f"{meta['category']}/{video_path.name}"
            on_progress("embedding", f"Embedding {rel_name}", i + 1, total)
            embedding = embedder.embed_media(video_path)
            if embedding is None:
                continue
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            media_fields = self.load_media_data(video_path)
            clips[clip_id] = {
                "id": clip_id,
                "type": self.type_id,
                "embedder": embedder.name,
                "duration": media_fields["duration"],
                "file_size": len(video_bytes),
                "md5": hashlib.md5(video_bytes).hexdigest(),
                "embedding": embedding,
                "media_bytes": video_bytes,
                "thumbnail_bytes": media_fields.get("thumbnail_bytes"),
                "filename": rel_name,
                "category": meta["category"],
                "origin": {**demo_origin_template},
                "origin_name": rel_name,
            }
            clip_id += 1

        return str(video_dir.absolute())

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["thumbnail_bytes"]

    def load_media_data(self, file_path: Path) -> dict:
        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(str(file_path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0.0
            finally:
                cap.release()
        except Exception:
            duration = 0.0
        thumbnail = generate_video_thumbnail_from_file(file_path)
        return {"media_bytes": media_bytes, "duration": duration, "thumbnail_bytes": thumbnail}

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def image_response(self, media: dict) -> MediaResponse | None:
        """Return the video thumbnail as a PNG image, or *None*."""
        thumb = media.get("thumbnail_bytes")
        if thumb:
            return MediaResponse(
                data=thumb,
                mimetype="image/png",
                download_name=f"media_{media['id']}_thumb.png",
            )
        # Fallback: generate on the fly from media bytes
        raw = self._resolve_media_bytes(media)
        if raw:
            thumb = generate_video_thumbnail(raw)
            if thumb:
                media["thumbnail_bytes"] = thumb
                return MediaResponse(
                    data=thumb,
                    mimetype="image/png",
                    download_name=f"media_{media['id']}_thumb.png",
                )
        return None

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".mp4"
        mimetype = _VIDEO_MIME_TYPES.get(ext, "video/mp4")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
