"""Synthetic audio generation - tones, chords, drums, rain, wind, birds.

Each generator writes a 16-bit mono PCM WAV file at 48 kHz. Variety across
six "ideas" is enough for CLAP and similar audio embedders to produce
distinguishable clusters when sorted.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ProgressCallback = Callable[[str, str, int, int], None]

SAMPLE_RATE = 48000


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """Write *samples* (float32 in [-1, 1]) as a 16-bit mono WAV at 48 kHz."""
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())


def _envelope(n: int, attack: float = 0.02, release: float = 0.05) -> np.ndarray:
    """Return a simple AR amplitude envelope of length *n* samples."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(SAMPLE_RATE * attack))
    r = max(1, int(SAMPLE_RATE * release))
    if a < n:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r < n:
        env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _tone(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    freq = float(rng.uniform(180, 1200))
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    samples = 0.4 * np.sin(2 * math.pi * freq * t) * _envelope(n)
    return samples, {"freq": round(freq, 1)}


def _chord(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    root = float(rng.uniform(110, 440))
    # Major / minor / diminished / sus4 (semitone offsets from root).
    intervals_choices = [(0, 4, 7), (0, 3, 7), (0, 3, 6), (0, 5, 7)]
    intervals = intervals_choices[int(rng.integers(0, len(intervals_choices)))]
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    samples = np.zeros(n, dtype=np.float32)
    for semitones in intervals:
        f = root * (2 ** (semitones / 12.0))
        samples += np.sin(2 * math.pi * f * t)
    samples *= 0.3 / max(1, len(intervals))
    samples *= _envelope(n, attack=0.05, release=0.2)
    return samples, {"root": round(root, 1), "intervals": list(intervals)}


def _drum(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    """Series of enveloped noise hits - kick / snare / hihat patterns."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    bpm = float(rng.uniform(80, 160))
    beat_samples = int(SAMPLE_RATE * 60.0 / bpm)
    kind_choices = ["kick", "snare", "hihat"]
    kind = kind_choices[int(rng.integers(0, len(kind_choices)))]
    pos = 0
    while pos < n:
        hit_len = min(beat_samples // 2, n - pos)
        if hit_len <= 0:
            break
        if kind == "kick":
            t = np.arange(hit_len, dtype=np.float32) / SAMPLE_RATE
            freq = 60.0 * np.exp(-t * 25.0) + 30.0
            phase = 2 * math.pi * np.cumsum(freq) / SAMPLE_RATE
            hit = 0.8 * np.sin(phase) * np.exp(-t * 12.0)
        elif kind == "snare":
            noise = rng.standard_normal(hit_len).astype(np.float32)
            t = np.arange(hit_len, dtype=np.float32) / SAMPLE_RATE
            hit = 0.6 * noise * np.exp(-t * 18.0)
        else:
            noise = rng.standard_normal(hit_len).astype(np.float32)
            t = np.arange(hit_len, dtype=np.float32) / SAMPLE_RATE
            # crude high-pass: subtract a smoothed copy
            kernel = np.ones(8, dtype=np.float32) / 8.0
            smoothed = np.convolve(noise, kernel, mode="same")
            hit = 0.4 * (noise - smoothed) * np.exp(-t * 40.0)
        out[pos : pos + hit_len] += hit
        pos += beat_samples
    return out, {"kind": kind, "bpm": round(bpm, 1)}


def _rain(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    """Pink-ish filtered noise - broadband hiss with occasional droplets."""
    n = int(SAMPLE_RATE * duration)
    noise = rng.standard_normal(n).astype(np.float32)
    # Cheap low-pass via cumulative average.
    kernel_size = int(rng.integers(8, 32))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    filtered = np.convolve(noise, kernel, mode="same")
    samples = 0.35 * filtered
    # Sprinkle in droplet clicks.
    n_drops = int(duration * float(rng.uniform(2, 8)))
    for _ in range(n_drops):
        pos = int(rng.integers(0, max(1, n - 200)))
        drop_len = int(rng.integers(50, 200))
        t = np.arange(drop_len, dtype=np.float32) / SAMPLE_RATE
        drop = 0.4 * np.sin(2 * math.pi * float(rng.uniform(800, 3000)) * t) * np.exp(-t * 80.0)
        samples[pos : pos + drop_len] += drop
    samples *= _envelope(n, attack=0.1, release=0.2)
    return samples, {"kernel": kernel_size, "drops": n_drops}


def _wind(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    """Slow AM-modulated brown-ish noise."""
    n = int(SAMPLE_RATE * duration)
    noise = rng.standard_normal(n).astype(np.float32)
    # Brown-ish noise: integrate then high-pass slightly via subtracting mean.
    brown = np.cumsum(noise)
    brown -= np.mean(brown)
    brown /= max(1e-6, float(np.max(np.abs(brown))))
    # Slow amplitude modulation.
    mod_freq = float(rng.uniform(0.2, 1.5))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    am = 0.5 + 0.4 * np.sin(2 * math.pi * mod_freq * t)
    samples = 0.45 * brown * am * _envelope(n, attack=0.2, release=0.4)
    return samples, {"mod_freq": round(mod_freq, 2)}


def _bird(rng: np.random.Generator, duration: float) -> tuple[np.ndarray, dict]:
    """A handful of FM-swept chirps with silence between them."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    n_chirps = int(rng.integers(3, 8))
    for _ in range(n_chirps):
        chirp_len = int(SAMPLE_RATE * float(rng.uniform(0.08, 0.25)))
        chirp_len = min(chirp_len, n - 1)
        if chirp_len <= 0:
            continue
        start = int(rng.integers(0, n - chirp_len))
        t = np.arange(chirp_len, dtype=np.float32) / SAMPLE_RATE
        f0 = float(rng.uniform(1500, 3000))
        f1 = f0 + float(rng.uniform(-1500, 2500))
        freq = np.linspace(f0, f1, chirp_len, dtype=np.float32)
        phase = 2 * math.pi * np.cumsum(freq) / SAMPLE_RATE
        env = np.exp(-((t - chirp_len / SAMPLE_RATE / 2) ** 2) * 800.0)
        chirp = 0.5 * np.sin(phase) * env
        out[start : start + chirp_len] += chirp
    return out, {"chirps": n_chirps}


_GENERATORS = [
    ("tone", _tone),
    ("chord", _chord),
    ("drum", _drum),
    ("rain", _rain),
    ("wind", _wind),
    ("bird", _bird),
]


def generate_audio_dataset(
    output_dir: Path,
    count: int,
    seed: int = 42,
    on_progress: Optional[ProgressCallback] = None,
) -> list[Path]:
    """Generate ``count`` synthetic WAV files into ``output_dir``.

    Cycles through six ideas (tone, chord, drum, rain, wind, bird) so the
    resulting dataset has clear semantic clusters. Existing files are kept,
    so reloads are fast.

    If *on_progress* is supplied, it is called as
    ``on_progress(status, message, current, total)`` once per file (and
    once at the start and end) so the caller can drive a progress bar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    width = max(4, len(str(count)))
    if on_progress is not None:
        on_progress("downloading", f"Generating {count} synthetic audio clips…", 0, count)
    for i in range(count):
        idea, gen = _GENERATORS[i % len(_GENERATORS)]
        path = output_dir / f"{idea}_{i:0{width}d}.wav"
        paths.append(path)
        cached = path.exists()
        if on_progress is not None:
            verb = "Reusing cached" if cached else "Synthesising"
            on_progress(
                "downloading",
                f"{verb} synthetic audio {i + 1}/{count} ({idea})…",
                i,
                count,
            )
        if cached:
            continue
        rng = np.random.default_rng(seed + i)
        duration = float(rng.uniform(1.0, 2.5))
        samples, _meta = gen(rng, duration)
        _write_wav(path, samples)
    if on_progress is not None:
        on_progress("downloading", f"Generated {count} synthetic audio clips", count, count)
    return paths
