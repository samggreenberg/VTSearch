"""Download + embed an audio demo dataset via the vtscore library tier.

Produces, under ``RESULTS/<dataset>/``:

- ``embeddings_<embedder>.npy`` — (n, d) float32 matrix, ingest-normalized,
  exactly what VTSearch's in-memory embedding matrix would hold.
- ``meta.json`` — aligned per-clip records: filename, category (ground
  truth), duration, absolute wav path (on node scratch) for the audio→text
  stages.

Usage::

    python prepare_dataset.py esc50 [--embedder clap] [--per-cat N]
    python prepare_dataset.py speech_commands --per-cat 40
    python prepare_dataset.py clotho
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

import numpy as np  # noqa: E402

SOURCES = {
    "esc50": "esc50",
    "speech_commands": "speech_commands_v2",
    "clotho": "clotho",
    "urbansound8k": "urbansound8k",
    "gtzan": "gtzan",
}


def discover_categories(source: str) -> list[str]:
    """Download the source (idempotent) and list its ground-truth categories."""
    from vtscore.datasets import downloader
    from vtscore.datasets.metadata import load_esc50_metadata

    if source == "esc50":
        audio_dir = downloader.download_esc50()
        meta = load_esc50_metadata(audio_dir.parent)
        return sorted({m["category"] for m in meta.values()})
    if source == "speech_commands_v2":
        audio_dir = downloader.download_speech_commands_v2()
        return sorted(p.name for p in audio_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if source == "clotho":
        return ["sound"]
    if source == "urbansound8k":
        from vtscore.datasets.loader import load_urbansound8k_metadata

        us8k = downloader.download_urbansound8k()
        meta = load_urbansound8k_metadata(us8k)
        return sorted({m["category"] for m in meta.values()})
    if source == "gtzan":
        audio_dir = downloader.download_gtzan()
        return sorted(p.name for p in audio_dir.iterdir() if p.is_dir())
    raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SOURCES))
    ap.add_argument("--embedder", default="clap")
    ap.add_argument("--per-cat", type=int, default=None, help="clips per category")
    args = ap.parse_args()

    from vtscore import media as media_registry

    timings: dict = {}
    source = SOURCES[args.dataset]
    with common.timed("discover_categories", timings):
        categories = discover_categories(source)
    print(f"{args.dataset}: {len(categories)} categories")

    mt = media_registry.get("audio")
    embedder = media_registry.get_embedder(args.embedder)

    clips: dict = {}
    with common.timed("load_demo_source(download+embed)", timings):
        audio_dir = mt.load_demo_source(
            source=source,
            categories=categories,
            slice_start=0,
            slice_end=args.per_cat,
            clips=clips,
            embedder=embedder,
        )
    print(f"loaded {len(clips)} clips from {audio_dir}")

    ordered = [clips[k] for k in sorted(clips)]
    emb = np.stack([c["embeddings"][embedder.name] for c in ordered]).astype(np.float32)

    # Write each clip's bytes to a flat per-dataset wav cache on scratch so the
    # audio→text stages never have to guess the source layout.
    wav_cache = common.WORK / "wavs" / args.dataset
    wav_cache.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(ordered):
        wav_path = wav_cache / f"{i:05d}.wav"
        if not wav_path.exists():
            wav_path.write_bytes(c["media_bytes"])
        meta.append(
            {
                "filename": c["filename"],
                "category": c["category"],
                "duration": round(float(c["duration"]), 2),
                "wav_path": str(wav_path),
            }
        )

    out = common.ds_dir(args.dataset)
    np.save(out / f"embeddings_{embedder.name}.npy", emb)
    common.save_json(out / "meta.json", meta)
    common.save_json(
        out / "prepare_info.json",
        {
            "dataset": args.dataset,
            "source": source,
            "embedder": embedder.name,
            "n_clips": len(meta),
            "dim": int(emb.shape[1]),
            "n_categories": len(categories),
            "timings_s": timings,
        },
    )


if __name__ == "__main__":
    main()
