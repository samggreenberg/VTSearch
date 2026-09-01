"""Demo dataset construction and loading for video media.

Builds the :class:`~vtscore.media.base.DemoDataset` list returned by
``VideoMediaType.demo_datasets`` and implements the per-source download
+ embed dispatcher behind ``VideoMediaType.load_demo_source``.

Both live here rather than on the class because they are pure functions
of the demo-category constants - splitting them out keeps
``media_type.py`` focused on the ``MediaType`` contract.  The loader is
handed the calling :class:`~vtscore.media.base.MediaType` because it
needs :meth:`~vtscore.media.base.MediaType.load_media_data` to derive
each clip's duration and thumbnail; passing it in keeps the dependency
explicit and avoids importing ``media_type`` back from here.
"""

from __future__ import annotations

from vtscore.media.base import DemoDataset, demo_slice_by_category
from vtscore.utils.hashing import content_md5


_MEDIA_TYPE_ID = "video"


_DEMO_CATEGORIES = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
]

_HMDB51_CATEGORIES = [
    "brush_hair",
    "cartwheel",
    "catch",
    "chew",
    "clap",
    "climb",
    "climb_stairs",
    "dive",
    "draw_sword",
    "dribble",
    "drink",
    "eat",
    "fall_floor",
    "fencing",
    "flic_flac",
    "golf",
    "handstand",
    "hit",
    "hug",
    "jump",
    "kick",
    "kick_ball",
    "kiss",
    "laugh",
    "pick",
    "pour",
    "pullup",
    "punch",
    "push",
    "pushup",
    "ride_bike",
    "ride_horse",
    "run",
    "shake_hands",
    "shoot_ball",
    "shoot_bow",
    "shoot_gun",
    "sit",
    "situp",
    "smile",
    "smoke",
    "somersault",
    "stand",
    "swing_baseball",
    "sword",
    "sword_exercise",
    "talk",
    "throw",
    "turn",
    "walk",
    "wave",
]

_UCF101_FULL_CATEGORIES = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "Hammering",
    "HammerThrow",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
]

_KTH_CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking",
]


def build_demo_datasets() -> list[DemoDataset]:
    """Build the demo-dataset catalog exposed by :class:`VideoMediaType`."""
    from vtscore.datasets.downloader import (  # noqa: PLC0415
        HMDB51_DOWNLOAD_SIZE_MB,
        KTH_DOWNLOAD_SIZE_MB,
        UCF101_FULL_DOWNLOAD_SIZE_MB,
        UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        VIDEO_DIR,
    )

    cats = _DEMO_CATEGORIES
    folder = VIDEO_DIR / "ucf101"

    hmdb_cats = _HMDB51_CATEGORIES
    hmdb_folder = VIDEO_DIR / "hmdb51"

    ucf_full_cats = _UCF101_FULL_CATEGORIES
    ucf_full_folder = VIDEO_DIR / "ucf101_full"

    kth_cats = _KTH_CATEGORIES
    kth_folder = VIDEO_DIR / "kth"

    desc = "YouTube action videos"
    hmdb_desc = "Human motion clips"
    ucf_full_desc = "Action videos, 101 classes"
    kth_desc = "Simple human actions"
    return [
        # UCF-101 subset (10 classes, 171 MB download)
        DemoDataset(
            id="ucf101_s",
            label="UCF-101 (S)",
            description=desc,
            categories=cats,
            source="ucf101",
            required_folder=folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=132,
            download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_m",
            label="UCF-101 (M)",
            description=desc,
            categories=cats,
            source="ucf101",
            required_folder=folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=132,
            download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_l",
            label="UCF-101 (L)",
            description=desc,
            categories=cats,
            source="ucf101",
            required_folder=folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=132,
            download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_a",
            label="UCF-101 (A)",
            description=desc,
            categories=cats,
            source="ucf101",
            required_folder=folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=132,
            download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        # HMDB51 (51 action classes, ~2 GB download)
        DemoDataset(
            id="hmdb51_s",
            label="HMDB51 (S)",
            description=hmdb_desc,
            categories=hmdb_cats,
            source="hmdb51",
            required_folder=hmdb_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=137,
            download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="hmdb51_m",
            label="HMDB51 (M)",
            description=hmdb_desc,
            categories=hmdb_cats,
            source="hmdb51",
            required_folder=hmdb_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=137,
            download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="hmdb51_l",
            label="HMDB51 (L)",
            description=hmdb_desc,
            categories=hmdb_cats,
            source="hmdb51",
            required_folder=hmdb_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=137,
            download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="hmdb51_a",
            label="HMDB51 (A)",
            description=hmdb_desc,
            categories=hmdb_cats,
            source="hmdb51",
            required_folder=hmdb_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=137,
            download_size_mb=HMDB51_DOWNLOAD_SIZE_MB,
        ),
        # UCF-101 full (101 action classes, ~7 GB download)
        DemoDataset(
            id="ucf101_full_s",
            label="UCF-101 Full (S)",
            description=ucf_full_desc,
            categories=ucf_full_cats,
            source="ucf101_full",
            required_folder=ucf_full_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=132,
            download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_full_m",
            label="UCF-101 Full (M)",
            description=ucf_full_desc,
            categories=ucf_full_cats,
            source="ucf101_full",
            required_folder=ucf_full_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=132,
            download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_full_l",
            label="UCF-101 Full (L)",
            description=ucf_full_desc,
            categories=ucf_full_cats,
            source="ucf101_full",
            required_folder=ucf_full_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=132,
            download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_full_a",
            label="UCF-101 Full (A)",
            description=ucf_full_desc,
            categories=ucf_full_cats,
            source="ucf101_full",
            required_folder=ucf_full_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=132,
            download_size_mb=UCF101_FULL_DOWNLOAD_SIZE_MB,
        ),
        # KTH Actions (6 action classes, ~1.1 GB download)
        DemoDataset(
            id="kth_s",
            label="KTH Actions (S)",
            description=kth_desc,
            categories=kth_cats,
            source="kth",
            required_folder=kth_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=100,
            download_size_mb=KTH_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="kth_m",
            label="KTH Actions (M)",
            description=kth_desc,
            categories=kth_cats,
            source="kth",
            required_folder=kth_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=100,
            download_size_mb=KTH_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="kth_l",
            label="KTH Actions (L)",
            description=kth_desc,
            categories=kth_cats,
            source="kth",
            required_folder=kth_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=KTH_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="kth_a",
            label="KTH Actions (A)",
            description=kth_desc,
            categories=kth_cats,
            source="kth",
            required_folder=kth_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=KTH_DOWNLOAD_SIZE_MB,
        ),
    ]


_VIDEO_SOURCE_DOWNLOADERS = {
    "ucf101": "download_ucf101_subset",
    "hmdb51": "download_hmdb51",
    "ucf101_full": "download_ucf101_full",
    "kth": "download_kth",
}


def load_demo_source(  # noqa: C901 - flat per-item embed/defer branching
    media_type,
    source,
    categories,
    slice_start,
    slice_end,
    clips,
    on_progress=None,
    embedder=None,
    slice_frac_start=None,
    slice_frac_end=None,
    skip_embedding=False,
    **kwargs,
):

    if on_progress is None:
        from vtscore.concurrency.progress import resolve_progress_callback

        on_progress = resolve_progress_callback()

    if embedder is None:
        from vtscore.media import embedders_for_type

        avail = embedders_for_type(_MEDIA_TYPE_ID)
        if not avail:
            raise ValueError(f"No embedders registered for media type {_MEDIA_TYPE_ID!r}")
        embedder = avail[0]

    downloader_name = _VIDEO_SOURCE_DOWNLOADERS.get(source)
    if downloader_name is None:
        raise ValueError(f"Unsupported video source: {source!r}")

    import vtscore.datasets.downloader as dl_module  # noqa: PLC0415
    from vtscore.datasets.metadata import load_video_metadata_from_folders  # noqa: PLC0415

    download_fn = getattr(dl_module, downloader_name)
    video_dir = download_fn(on_progress=on_progress)
    metadata = load_video_metadata_from_folders(video_dir, categories)

    by_cat: dict[str, list] = {}
    for fname, meta in sorted(metadata.items()):
        cat = meta["category"]
        by_cat.setdefault(cat, []).append((meta["path"], meta))

    video_files = demo_slice_by_category(by_cat, categories, slice_start, slice_end, slice_frac_start, slice_frac_end)

    # Load models (skipped when a clipper will re-embed every clip - see
    # skip_embedding in load_demo_dataset).
    if not skip_embedding and getattr(embedder, "_model", None) is None:
        on_progress("loading", "Loading video embedding model\u2026", 0, 0)
        with embedder.progress_scope(on_progress):
            embedder.load_models()

    clip_id = max(clips.keys(), default=0) + 1
    total = len(video_files)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} video files...", 0, total)
    demo_origin_template: dict = {"importer": "demo", "params": {}}

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    for i, (video_path, meta) in enumerate(video_files):
        rel_name = f"{meta['category']}/{video_path.name}"
        if skip_embedding:
            on_progress("loading", f"Loading {rel_name}", i + 1, total)
            embedding = None
        else:
            on_progress("embedding", f"Embedding {rel_name}", i + 1, total)
            embedding = embedder.embed_media(media_from_path(video_path))
            if embedding is None:
                continue
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        media_fields = media_type.load_media_data(video_path)
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": media_fields["duration"],
            "file_size": len(video_bytes),
            "md5": content_md5(video_bytes),
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
            "media_bytes": video_bytes,
            "thumbnail_bytes": media_fields.get("thumbnail_bytes"),
            "filename": rel_name,
            "category": meta["category"],
            "origin": {**demo_origin_template},
            "origin_name": rel_name,
        }
        clip_id += 1

    return str(video_dir.absolute())
