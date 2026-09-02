"""Demo dataset construction and loading for audio media.

Builds the :class:`~vtscore.media.base.DemoDataset` list returned by
``AudioMediaType.demo_datasets`` and implements the per-source download
+ embed dispatcher behind ``AudioMediaType.load_demo_source``.

Both live here rather than on the class because they are pure functions
of the demo-category constants - splitting them out keeps
``media_type.py`` focused on the ``MediaType`` contract.  The loader is
handed the calling :class:`~vtscore.media.base.MediaType` because it
needs :meth:`~vtscore.media.base.MediaType.load_media_data` to derive
each clip's duration and waveform; passing it in keeps the dependency
explicit and avoids importing ``media_type`` back from here.
"""

from __future__ import annotations

import io
from pathlib import Path

from vtscore.config import DATA_DIR
from vtscore.media._toponymy_demo import SOURCE_ID as _TOPONYMY_SOURCE_ID
from vtscore.media._toponymy_demo import TAXONOMY as _TOPONYMY_TAXONOMY
from vtscore.media.base import DemoDataset, demo_slice, demo_slice_by_category
from vtscore.utils.hashing import content_md5


_MEDIA_TYPE_ID = "audio"


# Synthetic world-map demo tone: a short mono sine whose pitch identifies the
# leaf city, so 108 cities span an audible, visibly-distinct range of waveforms.
_SYNTH_TONE_SECONDS = 0.6
_SYNTH_TONE_SR = 16000
_SYNTH_TONE_LO_HZ = 180.0
_SYNTH_TONE_HI_HZ = 1400.0


def _synthetic_tone_wav(city_index: int, n_cities: int) -> bytes:
    """Render a leaf city as a short 16-bit-PCM sine-tone WAV (no files, no ffmpeg).

    The pitch rises log-linearly with *city_index* across the taxonomy so each
    city gets a recognisably different tone (and waveform thumbnail); a faint
    second harmonic keeps the render from looking perfectly flat.
    """
    import struct  # noqa: PLC0415
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    frac = city_index / max(1, n_cities - 1)
    freq = _SYNTH_TONE_LO_HZ * (_SYNTH_TONE_HI_HZ / _SYNTH_TONE_LO_HZ) ** frac
    t = np.arange(int(_SYNTH_TONE_SR * _SYNTH_TONE_SECONDS), dtype=np.float64) / _SYNTH_TONE_SR
    wave_f = 0.6 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
    samples = np.clip(wave_f, -1.0, 1.0)
    pcm = (samples * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SYNTH_TONE_SR)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))
    return buf.getvalue()


# Shared categories for all S/M/L audio demo datasets.
_DEMO_CATEGORIES = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creep",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw",
]

_GTZAN_CATEGORIES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

_SPEECH_COMMANDS_CATEGORIES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "follow",
    "forward",
    "learn",
    "visual",
]

_URBANSOUND8K_CATEGORIES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

# TUT Sound Events 2017 ships uncut ~4-minute street soundscapes.  We don't
# use its event annotations: every recording goes into one "street" bucket
# so the user clips the long files themselves.
_TUT_CATEGORIES = ["street"]

# Clotho is an audio *captioning* dataset (no class labels), imported as
# real-world Freesound sound clips in one undifferentiated bucket.  It's the
# compositional-scene playground for natural-language text->audio search.
_CLOTHO_CATEGORIES = ["sound"]

# The three long-form demos below share a shape: hours-long unlabelled
# recordings in one undifferentiated bucket, where the interesting content
# is *discrete events scattered through the runtime* rather than a clip
# whose label is the whole clip.  That is what makes them detector
# playgrounds - you clip, listen, vote on a handful of hits, and let the
# ranker find the rest.
_APOLLO11_CATEGORIES = ["mission_audio"]
_BIRDVOX_CATEGORIES = ["night_recording"]
_NIXON_CATEGORIES = ["conversation"]


def build_demo_datasets() -> list[DemoDataset]:
    """Build the demo-dataset catalog exposed by :class:`AudioMediaType`."""
    from vtscore.datasets.downloader import (  # noqa: PLC0415
        APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
        BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
        CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        ESC50_DOWNLOAD_SIZE_MB,
        GTZAN_DOWNLOAD_SIZE_MB,
        NIXON_TAPES_DOWNLOAD_SIZE_MB,
        SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
        URBANSOUND8K_DOWNLOAD_SIZE_MB,
    )

    cats = _DEMO_CATEGORIES
    folder = DATA_DIR / "ESC-50-master" / "audio"
    esc_desc = "Animals, nature, cities, & homes"
    sc_desc = "Spoken keyword utterances"
    us_desc = "Urban recordings"
    # 24 development + 8 evaluation recordings, all one "street" bucket.
    tut_folder = DATA_DIR / "tut_sound_events_2017"
    tut_desc = "Long ~4min street soundscapes (clip them yourself)"
    tut_total = 32
    # Clotho eval split: 1045 real-world Freesound clips, one "sound" bucket.
    clotho_folder = DATA_DIR / "clotho"
    clotho_desc = "Real-world Freesound clips for text search"
    clotho_total = 1045
    # Apollo 11: 103 MP3 tracks, ~174 h of mission loops (median ~85 min).
    apollo_folder = DATA_DIR / "apollo11_audio"
    apollo_desc = "Long NASA mission loops — Quindar beeps, alarms, applause"
    apollo_total = 103
    # BirdVox: 6 ten-hour units, segmented into 10-minute chunks on download.
    birdvox_folder = DATA_DIR / "birdvox_full_night"
    birdvox_desc = "10min chunks of all-night birdsong — sub-second flight calls"
    birdvox_total = 360
    # Nixon: 12 tapes' worth of conversations, one MP3 per conversation.
    nixon_folder = DATA_DIR / "nixon_tapes"
    nixon_desc = "Secret-taping-system conversations (rough audio, by design)"
    nixon_total = 1917
    return [
        DemoDataset(
            id="esc50_s",
            label="ESC-50 (S)",
            description=esc_desc,
            categories=cats,
            source="esc50",
            required_folder=folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=40,
            download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="esc50_m",
            label="ESC-50 (M)",
            description=esc_desc,
            categories=cats,
            source="esc50",
            required_folder=folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=40,
            download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="esc50_l",
            label="ESC-50 (L)",
            description=esc_desc,
            categories=cats,
            source="esc50",
            required_folder=folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=40,
            download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="esc50_a",
            label="ESC-50 (A)",
            description=esc_desc,
            categories=cats,
            source="esc50",
            required_folder=folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=40,
            download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="gtzan_a",
            label="GTZAN Music Genre (A)",
            description="30sec music excerpts",
            categories=_GTZAN_CATEGORIES,
            source="gtzan",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=100,
            download_size_mb=GTZAN_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="speech_commands_v2_s",
            label="Speech Commands v2 (S)",
            description=sc_desc,
            categories=_SPEECH_COMMANDS_CATEGORIES,
            source="speech_commands_v2",
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=3000,
            download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="speech_commands_v2_m",
            label="Speech Commands v2 (M)",
            description=sc_desc,
            categories=_SPEECH_COMMANDS_CATEGORIES,
            source="speech_commands_v2",
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=3000,
            download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="speech_commands_v2_l",
            label="Speech Commands v2 (L)",
            description=sc_desc,
            categories=_SPEECH_COMMANDS_CATEGORIES,
            source="speech_commands_v2",
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=3000,
            download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="speech_commands_v2_a",
            label="Speech Commands v2 (A)",
            description=sc_desc,
            categories=_SPEECH_COMMANDS_CATEGORIES,
            source="speech_commands_v2",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=3000,
            download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="urbansound8k_s",
            label="UrbanSound8K (S)",
            description=us_desc,
            categories=_URBANSOUND8K_CATEGORIES,
            source="urbansound8k",
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=873,
            download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="urbansound8k_m",
            label="UrbanSound8K (M)",
            description=us_desc,
            categories=_URBANSOUND8K_CATEGORIES,
            source="urbansound8k",
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=873,
            download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="urbansound8k_l",
            label="UrbanSound8K (L)",
            description=us_desc,
            categories=_URBANSOUND8K_CATEGORIES,
            source="urbansound8k",
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=873,
            download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="urbansound8k_a",
            label="UrbanSound8K (A)",
            description=us_desc,
            categories=_URBANSOUND8K_CATEGORIES,
            source="urbansound8k",
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=873,
            download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="tut_sound_events_2017_s",
            label="TUT Sound Events 2017 (S)",
            description=tut_desc,
            categories=_TUT_CATEGORIES,
            source="tut_sound_events_2017",
            required_folder=tut_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=tut_total,
            download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="tut_sound_events_2017_m",
            label="TUT Sound Events 2017 (M)",
            description=tut_desc,
            categories=_TUT_CATEGORIES,
            source="tut_sound_events_2017",
            required_folder=tut_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=tut_total,
            download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="tut_sound_events_2017_l",
            label="TUT Sound Events 2017 (L)",
            description=tut_desc,
            categories=_TUT_CATEGORIES,
            source="tut_sound_events_2017",
            required_folder=tut_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=tut_total,
            download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="tut_sound_events_2017_a",
            label="TUT Sound Events 2017 (A)",
            description=tut_desc,
            categories=_TUT_CATEGORIES,
            source="tut_sound_events_2017",
            required_folder=tut_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=tut_total,
            download_size_mb=TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="clotho_s",
            label="Clotho (S)",
            description=clotho_desc,
            categories=_CLOTHO_CATEGORIES,
            source="clotho",
            required_folder=clotho_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 7,
            items_per_category=clotho_total,
            download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="clotho_m",
            label="Clotho (M)",
            description=clotho_desc,
            categories=_CLOTHO_CATEGORIES,
            source="clotho",
            required_folder=clotho_folder,
            slice_frac_start=1 / 7,
            slice_frac_end=3 / 7,
            items_per_category=clotho_total,
            download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="clotho_l",
            label="Clotho (L)",
            description=clotho_desc,
            categories=_CLOTHO_CATEGORIES,
            source="clotho",
            required_folder=clotho_folder,
            slice_frac_start=3 / 7,
            slice_frac_end=None,
            items_per_category=clotho_total,
            download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="clotho_a",
            label="Clotho (A)",
            description=clotho_desc,
            categories=_CLOTHO_CATEGORIES,
            source="clotho",
            required_folder=clotho_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=clotho_total,
            download_size_mb=CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="apollo11_audio_s",
            label="Apollo 11 Mission Audio (S)",
            description=apollo_desc,
            categories=_APOLLO11_CATEGORIES,
            source="apollo11_audio",
            required_folder=apollo_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 12,
            items_per_category=apollo_total,
            download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="apollo11_audio_m",
            label="Apollo 11 Mission Audio (M)",
            description=apollo_desc,
            categories=_APOLLO11_CATEGORIES,
            source="apollo11_audio",
            required_folder=apollo_folder,
            slice_frac_start=1 / 12,
            slice_frac_end=3 / 12,
            items_per_category=apollo_total,
            download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="apollo11_audio_l",
            label="Apollo 11 Mission Audio (L)",
            description=apollo_desc,
            categories=_APOLLO11_CATEGORIES,
            source="apollo11_audio",
            required_folder=apollo_folder,
            slice_frac_start=3 / 12,
            slice_frac_end=None,
            items_per_category=apollo_total,
            download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="apollo11_audio_a",
            label="Apollo 11 Mission Audio (A)",
            description=apollo_desc,
            categories=_APOLLO11_CATEGORIES,
            source="apollo11_audio",
            required_folder=apollo_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=apollo_total,
            download_size_mb=APOLLO11_AUDIO_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="birdvox_full_night_s",
            label="BirdVox Full Night (S)",
            description=birdvox_desc,
            categories=_BIRDVOX_CATEGORIES,
            source="birdvox_full_night",
            required_folder=birdvox_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 6,
            items_per_category=birdvox_total,
            download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="birdvox_full_night_m",
            label="BirdVox Full Night (M)",
            description=birdvox_desc,
            categories=_BIRDVOX_CATEGORIES,
            source="birdvox_full_night",
            required_folder=birdvox_folder,
            slice_frac_start=1 / 6,
            slice_frac_end=3 / 6,
            items_per_category=birdvox_total,
            download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="birdvox_full_night_l",
            label="BirdVox Full Night (L)",
            description=birdvox_desc,
            categories=_BIRDVOX_CATEGORIES,
            source="birdvox_full_night",
            required_folder=birdvox_folder,
            slice_frac_start=3 / 6,
            slice_frac_end=None,
            items_per_category=birdvox_total,
            download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="birdvox_full_night_a",
            label="BirdVox Full Night (A)",
            description=birdvox_desc,
            categories=_BIRDVOX_CATEGORIES,
            source="birdvox_full_night",
            required_folder=birdvox_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=birdvox_total,
            download_size_mb=BIRDVOX_FULL_NIGHT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="nixon_tapes_s",
            label="Nixon White House Tapes (S)",
            description=nixon_desc,
            categories=_NIXON_CATEGORIES,
            source="nixon_tapes",
            required_folder=nixon_folder,
            slice_frac_start=0.0,
            slice_frac_end=1 / 12,
            items_per_category=nixon_total,
            download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="nixon_tapes_m",
            label="Nixon White House Tapes (M)",
            description=nixon_desc,
            categories=_NIXON_CATEGORIES,
            source="nixon_tapes",
            required_folder=nixon_folder,
            slice_frac_start=1 / 12,
            slice_frac_end=3 / 12,
            items_per_category=nixon_total,
            download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="nixon_tapes_l",
            label="Nixon White House Tapes (L)",
            description=nixon_desc,
            categories=_NIXON_CATEGORIES,
            source="nixon_tapes",
            required_folder=nixon_folder,
            slice_frac_start=3 / 12,
            slice_frac_end=None,
            items_per_category=nixon_total,
            download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="nixon_tapes_a",
            label="Nixon White House Tapes (A)",
            description=nixon_desc,
            categories=_NIXON_CATEGORIES,
            source="nixon_tapes",
            required_folder=nixon_folder,
            slice_frac_start=0.0,
            slice_frac_end=None,
            items_per_category=nixon_total,
            download_size_mb=NIXON_TAPES_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="synthetic_world_audio",
            label="Synthetic World Map (signposts demo)",
            description=(
                "Pre-baked 4-level toponymy (Continent → Country → State → City) "
                "with cheating ground-truth signposts — no download, loads instantly."
            ),
            categories=list(_TOPONYMY_TAXONOMY.keys()),
            source=_TOPONYMY_SOURCE_ID,
            items_per_category=0,
            download_size_mb=0,
        ),
    ]


def _load_synthetic_toponymy(clips, embedder, on_progress):
    """Populate *clips* with the synthetic world-map demo (no model, no download).

    Each item is a leaf city rendered as a short sine tone (a per-city
    frequency), tagged with its ``Continent/Country/State/City`` path and a
    pre-baked hierarchical embedding.  Browsing it lights up the ground-truth
    signpost layer straight from those paths — the friction-free way to eval
    the VTSBrowse sign display.  See :mod:`vtscore.media._toponymy_demo`.
    """

    from vtscore.media._toponymy_demo import generate_items, total_cities  # noqa: PLC0415

    # CLAP's audio/text space is 512-D; match it so the baked vectors slot
    # into the primary embedder's slot and text queries don't dimension-clash.
    from vtscore.media.audio.media_type import generate_waveform_thumbnail  # noqa: PLC0415

    items = generate_items(dim=512)
    n_cities = total_cities()
    total = len(items)
    on_progress("loading", f"Generating {total} synthetic clips…", 0, total)

    emb_name = embedder.name if embedder is not None else "clap"
    clip_id = 1
    for i, item in enumerate(items):
        wav_bytes = _synthetic_tone_wav(item.city_index, n_cities)
        thumb = generate_waveform_thumbnail(wav_bytes)
        filename = f"{item.category}/clip{i:04d}.wav"
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": emb_name,
            "duration": _SYNTH_TONE_SECONDS,
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embeddings": {emb_name: item.embedding},
            "media_bytes": wav_bytes,
            "thumbnail_bytes": thumb,
            "filename": filename,
            "category": item.category,
            "origin": {"importer": "demo", "params": {}},
            "origin_name": filename,
        }
        clip_id += 1
        if (i + 1) % 100 == 0:
            on_progress("loading", f"Generating synthetic clips… ({i + 1}/{total})", i + 1, total)
    # Bytes ride inline in the pickle — no external media dir.
    return None


def _collect_longform_audio_files(
    source: str,
    categories: list,
    slice_start: int,
    slice_end: int | None,
    slice_frac_start: float | None,
    slice_frac_end: float | None,
    on_progress,
):
    """Resolve a long-form demo source → ``(audio_files, audio_dir)``, else ``None``.

    Apollo 11, BirdVox-full-night and the Nixon tapes are hours-long
    unlabelled recordings running to 5-10 GB apiece, so they invert the
    order the other demos use: the *manifest* is sliced first and only the
    selected items are downloaded, rather than pulling the whole source and
    slicing afterwards.  Each loads as one undifferentiated bucket - the
    events worth finding are scattered inside the recordings, so there is
    nothing to label at the file level.

    Returns ``None`` when *source* is not one of the three, leaving
    :meth:`_collect_audio_files` to handle it.
    """

    def _bucket(paths: list, default_category: str, audio_dir: Path):
        category = categories[0] if categories else default_category
        return [(p, {"category": category, "path": p}) for p in paths], audio_dir

    def _select(items: list):
        return demo_slice(items, slice_start, slice_end, slice_frac_start, slice_frac_end)

    if source == "apollo11_audio":
        from vtscore.datasets.downloader import (  # noqa: PLC0415
            apollo11_audio_manifest,
            download_apollo11_audio,
        )

        tracks = _select(apollo11_audio_manifest())
        audio_dir = download_apollo11_audio(tracks, on_progress=on_progress)
        paths = [audio_dir / name for name, _size in tracks]
        return _bucket([p for p in paths if p.exists()], "mission_audio", audio_dir)

    if source == "birdvox_full_night":
        from vtscore.datasets.downloader import (  # noqa: PLC0415
            birdvox_full_night_manifest,
            download_birdvox_full_night,
        )

        units = _select(birdvox_full_night_manifest())
        base_dir = download_birdvox_full_night(units, on_progress=on_progress)
        # The download segments each ~10-hour unit into 10-minute chunks.
        paths = sorted(p for unit in units for p in (base_dir / unit).glob("*.flac"))
        return _bucket(paths, "night_recording", base_dir)

    if source == "nixon_tapes":
        from vtscore.datasets.downloader import (  # noqa: PLC0415
            download_nixon_tapes,
            nixon_tape_manifest,
        )

        tapes = _select(nixon_tape_manifest())
        base_dir = download_nixon_tapes(tapes, on_progress=on_progress)
        # NARA serves one MP3 per recorded conversation.
        paths = sorted(p for tape in tapes for p in (base_dir / tape).glob("*.mp3"))
        return _bucket(paths, "conversation", base_dir)

    return None


def _collect_audio_files(
    source: str,
    categories: list,
    slice_start: int,
    slice_end: int | None,
    slice_frac_start: float | None,
    slice_frac_end: float | None,
    on_progress,
):
    """Resolve a demo source name → (audio_files, audio_dir)."""

    def _sliced_by_category(by_cat: dict[str, list]) -> list:
        return demo_slice_by_category(by_cat, categories, slice_start, slice_end, slice_frac_start, slice_frac_end)

    if source == "gtzan":
        from vtscore.datasets.downloader import download_gtzan  # noqa: PLC0415
        from vtscore.datasets.metadata import load_audio_metadata_from_folders  # noqa: PLC0415

        audio_dir = download_gtzan(on_progress=on_progress)
        metadata = load_audio_metadata_from_folders(audio_dir, categories)
        by_cat = _group_metadata_by_category(metadata, categories, filter_to_categories=False)
        return _sliced_by_category(by_cat), audio_dir

    if source == "speech_commands_v2":
        from vtscore.datasets.downloader import download_speech_commands_v2  # noqa: PLC0415
        from vtscore.datasets.metadata import load_audio_metadata_from_folders  # noqa: PLC0415

        audio_dir = download_speech_commands_v2(on_progress=on_progress)
        metadata = load_audio_metadata_from_folders(audio_dir, categories)
        by_cat = _group_metadata_by_category(metadata, categories, filter_to_categories=False)
        return _sliced_by_category(by_cat), audio_dir

    if source == "urbansound8k":
        from vtscore.datasets.downloader import download_urbansound8k  # noqa: PLC0415
        from vtscore.datasets.metadata import load_urbansound8k_metadata  # noqa: PLC0415

        us8k_dir = download_urbansound8k(on_progress=on_progress)
        metadata = load_urbansound8k_metadata(us8k_dir)
        by_cat = _group_metadata_by_category(metadata, categories, filter_to_categories=True)
        return _sliced_by_category(by_cat), us8k_dir / "audio"

    if source == "tut_sound_events_2017":
        from vtscore.datasets.downloader import download_tut_sound_events_2017  # noqa: PLC0415

        audio_dir = download_tut_sound_events_2017(on_progress=on_progress)
        # No annotations: every recording is one undifferentiated bucket.
        category = categories[0] if categories else "street"
        by_cat = {category: [(p, {"category": category, "path": p}) for p in sorted(audio_dir.rglob("*.wav"))]}
        return _sliced_by_category(by_cat), audio_dir

    longform = _collect_longform_audio_files(
        source,
        categories,
        slice_start,
        slice_end,
        slice_frac_start,
        slice_frac_end,
        on_progress,
    )
    if longform is not None:
        return longform

    if source == "clotho":
        from vtscore.datasets.downloader import download_clotho  # noqa: PLC0415

        audio_dir = download_clotho(on_progress=on_progress)
        # Captioning dataset with no class labels: one undifferentiated bucket.
        category = categories[0] if categories else "sound"
        by_cat = {category: [(p, {"category": category, "path": p}) for p in sorted(audio_dir.rglob("*.wav"))]}
        return _sliced_by_category(by_cat), audio_dir

    if not source or source == "esc50":
        from vtscore.datasets.downloader import download_esc50  # noqa: PLC0415
        from vtscore.datasets.metadata import load_esc50_metadata  # noqa: PLC0415

        audio_dir = download_esc50(on_progress=on_progress)
        esc_metadata = load_esc50_metadata(audio_dir.parent)
        by_cat = _esc50_by_category(audio_dir, esc_metadata, categories)
        return _sliced_by_category(by_cat), audio_dir

    raise ValueError(f"Unsupported audio source: {source!r}")


def _esc50_by_category(audio_dir: Path, esc_metadata: dict, categories: list) -> dict[str, list]:
    """Group ESC-50 wav files by their category, keeping only *categories*."""
    by_cat: dict[str, list] = {}
    for audio_path in sorted(audio_dir.glob("*.wav")):
        meta = esc_metadata.get(audio_path.name)
        if meta is not None and meta["category"] in categories:
            by_cat.setdefault(meta["category"], []).append((audio_path, meta))
    return by_cat


def _group_metadata_by_category(
    metadata: dict,
    categories: list,
    *,
    filter_to_categories: bool,
) -> dict[str, list]:
    by_cat: dict[str, list] = {}
    for _key, meta in sorted(metadata.items()):
        cat = meta["category"]
        if filter_to_categories and cat not in categories:
            continue
        by_cat.setdefault(cat, []).append((meta["path"], meta))
    return by_cat


def load_demo_source(
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
        from vtscore.concurrency.progress import update_progress

        on_progress = update_progress

    if embedder is None:
        from vtscore.media import embedders_for_type

        avail = embedders_for_type(_MEDIA_TYPE_ID)
        if not avail:
            raise ValueError(f"No embedders registered for media type {_MEDIA_TYPE_ID!r}")
        embedder = avail[0]

    # Synthetic signposts demo: generated in-memory, no download, no model.
    if source == _TOPONYMY_SOURCE_ID:
        return _load_synthetic_toponymy(clips, embedder, on_progress)

    audio_files, audio_dir = _collect_audio_files(
        source,
        categories,
        slice_start,
        slice_end,
        slice_frac_start,
        slice_frac_end,
        on_progress,
    )

    # Load models (skipped when a clipper will re-embed every clip - see
    # skip_embedding in load_demo_dataset).
    if not skip_embedding and getattr(embedder, "_model", None) is None:
        on_progress("loading", "Loading audio embedding model…", 0, 0)
        with embedder.progress_scope(on_progress):
            embedder.load_models()

    clip_id = max(clips.keys(), default=0) + 1
    total = len(audio_files)
    status = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(status, f"{verb} {total} audio files...", 0, total)
    demo_origin: dict = {"importer": "demo", "params": {}}

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    for i, (audio_path, meta) in enumerate(audio_files):
        rel_name = f"{meta['category']}/{audio_path.name}"
        if skip_embedding:
            on_progress("loading", f"Loading {rel_name}", i + 1, total)
            embedding = None
        else:
            on_progress("embedding", f"Embedding {rel_name}", i + 1, total)
            embedding = embedder.embed_media(media_from_path(audio_path))
            if embedding is None:
                continue

        with open(audio_path, "rb") as f:
            wav_bytes = f.read()

        media_fields = media_type.load_media_data(audio_path)
        clips[clip_id] = {
            "id": clip_id,
            "media_type": _MEDIA_TYPE_ID,
            "embedder": embedder.name,
            "duration": media_fields["duration"],
            "file_size": len(wav_bytes),
            "md5": content_md5(wav_bytes),
            "embeddings": {} if skip_embedding else {embedder.name: embedding},
            "media_bytes": wav_bytes,
            "thumbnail_bytes": media_fields.get("thumbnail_bytes"),
            "filename": rel_name,
            "category": meta["category"],
            "origin": demo_origin,
            "origin_name": rel_name,
        }
        clip_id += 1

    return str(audio_dir.absolute())
