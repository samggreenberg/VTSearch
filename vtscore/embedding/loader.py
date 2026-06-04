"""Model loading and initialisation - delegates to the media type registry.

All embedding models are now owned by their respective
:class:`~vtscore.media.base.MediaType` instances and are loaded **lazily**
on first use (the first call to ``embed_media``, ``embed_text``, or a getter
function such as ``get_clap_model``).

This module keeps its original public API (``initialize_models``,
``get_clap_model``, etc.) as thin wrappers so that existing callers continue
to work unchanged.
"""

import gc
import os
import re
import sys
from typing import Any, cast

_TIME_SUFFIX_RE = re.compile(r"\s*\(\d+s(?:,\s*\d+\s+modules)?\)$")


def _strip_time_suffix(msg: str) -> str:
    return _TIME_SUFFIX_RE.sub("", msg) if msg else msg


from vtscore.config import MODELS_CACHE_DIR, resolve_device
from vtscore.media.torch_setup import ensure_torch_configured


def get_torch_device():
    """Return the preferred ``torch.device`` for MLP training / scoring.

    Resolves :data:`vtscore.config.DEVICE` (``VTSEARCH_DEVICE``, default
    ``"auto"``) to a concrete device - ``cuda`` when available, ``mps`` on
    Apple silicon, or ``cpu``.  Imports torch lazily.
    """
    import torch  # noqa: PLC0415

    return torch.device(resolve_device())


def _detect_cuda_devices() -> int:
    """Return the count of visible CUDA GPUs, or 0 if none / torch missing.

    Imports torch lazily so callers (e.g. settings default factories) can
    run before torch is loaded. All exceptions degrade to ``0`` - a missing
    or broken CUDA stack must not block startup.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def default_concurrent_downloads() -> int:
    """Default for ``max_concurrent_dataset_downloads`` derived from hardware.

    The download phase is bandwidth- and disk-bound. Allowing a handful of
    parallel downloads usually saturates a home connection without thrashing
    the disk; capped at 4 to keep memory and FD pressure reasonable on small
    boxes.
    """
    return max(1, min(4, os.cpu_count() or 1))


# A single CPU embed job holds an embedder model plus an N x D fp32 working
# set; budgeting ~4 GiB of total RAM per concurrent job keeps memory-starved
# boxes at one worker while letting roomy workstations run a few in parallel.
_RAM_BYTES_PER_CPU_EMBED = 4 * 1024 * 1024 * 1024

# Upper bound on the CPU embed default regardless of how big the box is; a
# hand override in ``data/settings.json`` can still go higher (clamped to 16).
_MAX_CPU_EMBED_DEFAULT = 4


def _total_memory_bytes() -> int:
    """Best-effort total physical RAM in bytes, or ``0`` if it can't be read.

    Uses ``MemTotal`` (Linux ``/proc/meminfo``) with an ``SC_PHYS_PAGES``
    sysconf fallback. Total (not *available*) RAM is the right signal for a
    startup default: it's stable, whereas free memory swings with whatever
    else happens to be running when the setting is first resolved.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (OSError, ValueError):
        pass
    return 0


def default_concurrent_embeddings() -> int:
    """Default for ``max_concurrent_dataset_embeddings`` derived from hardware.

    The embed phase is CPU/GPU- and RAM-bound, so the default scales with the
    scarcer of the two resources:

    * **GPU boxes** allow one task per visible CUDA device (capped at 2) so two
      datasets can embed in parallel on a multi-GPU rig without overcommitting a
      single device's VRAM.
    * **CPU-only boxes** allow roughly one job per 4 cores and one job per
      ``_RAM_BYTES_PER_CPU_EMBED`` of total RAM, whichever is smaller, capped at
      :data:`_MAX_CPU_EMBED_DEFAULT`. Constrained machines (few cores or little
      RAM) still resolve to 1 - preserving the old fully-serial behaviour where
      a second concurrent embed would thrash or OOM - while workstations get
      genuine parallel embedding with no config change. When total RAM can't be
      read we fall back to 1 rather than guess generously.
    """
    gpus = _detect_cuda_devices()
    if gpus > 0:
        return max(1, min(2, gpus))

    by_cpu = (os.cpu_count() or 1) // 4
    total_ram = _total_memory_bytes()
    by_ram = total_ram // _RAM_BYTES_PER_CPU_EMBED if total_ram else 1
    return max(1, min(_MAX_CPU_EMBED_DEFAULT, by_cpu, by_ram))


def initialize_models() -> None:
    """Prepare the runtime environment for embedding models.

    Creates the model cache directory and configures PyTorch thread count
    **if torch is already imported**.  When torch has not been imported yet
    (e.g. during fast test startup) the thread-count configuration is
    deferred until ``ensure_torch_configured`` is called by the first code
    path that actually imports torch.

    Models themselves are **not** loaded here.
    """
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_torch_configured()
    try:
        from vtsearch.logging_config import install_transformers_logging_bridge  # noqa: PLC0415

        install_transformers_logging_bridge()
    except Exception:
        pass
    gc.collect()


def _make_console_progress(original_callback):
    """Wrap *original_callback* to also print status to the terminal.

    Used during startup preloading so the user sees intermediate status
    messages and download progress bars in the console while models are
    being loaded.

    Consecutive progress events sharing a base message (the same text with
    any trailing ``(Ns)`` elapsed-time suffix stripped) overwrite the same
    terminal line, so ``timed_progress`` ticker updates animate in place
    instead of stacking new lines.  The progress line is terminated with a
    newline only when a different base message arrives, a phase message
    arrives, or the caller invokes ``cb.flush()``.
    """
    _last_msg: list[str | None] = [None]
    _last_base: list[str | None] = [None]
    _on_progress_line: list[bool] = [False]

    _FULL_BAR = "#" * 30

    def _complete_bar() -> None:
        """Overwrite the current progress line with a 100% bar."""
        if _last_base[0]:
            sys.stdout.write(f"\r    {_last_base[0]} [{_FULL_BAR}] 100%\033[K")

    def _flush() -> None:
        if _on_progress_line[0]:
            _complete_bar()
            sys.stdout.write("\n")
            sys.stdout.flush()
            _on_progress_line[0] = False
            _last_base[0] = None
            _last_msg[0] = None

    def _callback(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        original_callback(status, message, current, total)

        if total > 0:
            base = _strip_time_suffix(message)
            # If we're already on a progress line for a different task, complete it
            # at 100% before starting the new bar.
            if _on_progress_line[0] and _last_base[0] is not None and _last_base[0] != base:
                _complete_bar()
                sys.stdout.write("\n")
            pct = min(100, current * 100 // total)
            filled = pct * 30 // 100
            bar = "#" * filled + "." * (30 - filled)
            # \033[K clears from cursor to end of line so a shorter message
            # doesn't leave trailing chars from a longer prior render.
            line = f"\r    {message} [{bar}] {pct:>3}%\033[K"
            sys.stdout.write(line)
            sys.stdout.flush()
            _on_progress_line[0] = True
            _last_base[0] = base
            _last_msg[0] = message
        elif message and message != _last_msg[0]:
            # New phase message - print on its own line; complete any active bar first.
            if _on_progress_line[0]:
                _complete_bar()
                sys.stdout.write("\n")
                _on_progress_line[0] = False
                _last_base[0] = None
            sys.stdout.write(f"    {message}\n")
            sys.stdout.flush()
            _last_msg[0] = message

    _callback.flush = _flush  # type: ignore[attr-defined]
    return _callback


def predict_embedders_to_preload(
    extra_media_types: list[str] | None = None,
    extra_embedders: list[str] | None = None,
) -> list[str]:
    """Predict which embedders are likely to be needed next, from active metadata.

    Walks the dataset registry and detector registry and returns the unique
    list of embedder names the user is likely to need:

    - For each registered dataset and detector: ``entry["embedder"]`` if set
      and recognised, otherwise the default embedder for
      ``entry["media_type"]``.  A set-but-unrecognised ``embedder`` (e.g.
      the embedder was renamed or removed) also falls back to the media
      type's default rather than being silently dropped - losing the
      optimisation entirely on a typo is worse than warming the wrong
      embedder, which the user can still override.
    - For every media type in *extra_media_types*: the default embedder
      for that type. Used by the solo-mediaType streamlined mode (see
      :func:`vtsearch.settings.get_effective_solo_media_type`) so the
      user's chosen type has its default embedder warm at startup even
      when no datasets or detectors are registered yet.
    - For every embedder name in *extra_embedders*: the embedder itself
      (if it exists in the registry). Used by the solo-mediaEmbedder
      mode (``--solo-embedder TYPE=EMB``) so the CLI-pinned embedder is
      warm even when its mediaType is not in *extra_media_types*.

    Detector entries written before the ``embedder`` field existed have
    ``entry["embedder"] == ""``, so they fall through to the media type's
    default - matching the previous behaviour for unmigrated state.

    Order reflects discovery order (extras first so a solo-mode user
    isn't stuck behind unrelated dataset warmups, then datasets, then
    detectors), and is stable across runs.
    """
    from vtscore.datasets.registry import list_datasets
    from vtscore.detectors.registry import list_detectors
    from vtscore.media import all_embedders, embedders_for_type

    valid = {e.name for e in all_embedders()}

    def _default_for(media_type: str) -> str:
        if not media_type:
            return ""
        opts = embedders_for_type(media_type)
        return opts[0].name if opts else ""

    def _resolve(entry: dict) -> str:
        emb = (entry.get("embedder") or "").strip()
        if emb and emb in valid:
            return emb
        return _default_for(entry.get("media_type", "") or "")

    candidates: list[str] = []
    candidates.extend(extra_embedders or ())
    candidates.extend(_default_for(mt) for mt in (extra_media_types or ()))
    candidates.extend(_resolve(entry) for entry in list_datasets())
    candidates.extend(_resolve(entry) for entry in list_detectors())

    predictions: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if name and name in valid and name not in seen:
            seen.add(name)
            predictions.append(name)
    return predictions


def preload_predicted_embedders(
    extra_media_types: list[str] | None = None,
    extra_embedders: list[str] | None = None,
) -> list[str]:
    """Eagerly load embedding models predicted by :func:`predict_embedders_to_preload`.

    Calls :meth:`~vtscore.media.base.MediaEmbedder.load_models` on each
    predicted embedder so it is warm before the user opens the GUI.
    Prints intermediate status messages and download progress bars to
    the console while models load.

    *extra_media_types* is forwarded to :func:`predict_embedders_to_preload`
    so the caller (e.g. ``initialize_server`` honoring the
    ``--solo-media-type`` CLI fallback) can ensure a specific mediaType's
    default embedder is warm even with empty registries.

    Returns the list of embedder names that were preloaded.
    """
    from vtscore.media import get_embedder

    targets = predict_embedders_to_preload(
        extra_media_types=extra_media_types,
        extra_embedders=extra_embedders,
    )
    if not targets:
        return []

    print(f"  Predicted embedders to preload: {', '.join(targets)}", flush=True)
    preloaded: list[str] = []
    for emb_name in targets:
        try:
            emb = get_embedder(emb_name)
            print(f"  Preloading {emb_name} embedder...", flush=True)
            original_cb = emb._on_progress
            console_cb = _make_console_progress(original_cb)
            emb._on_progress = console_cb
            try:
                emb.load_models()
            finally:
                console_cb.flush()  # type: ignore[attr-defined]
                emb._on_progress = original_cb
            preloaded.append(emb_name)
        except Exception as exc:
            print(f"  Warning: failed to preload {emb_name}: {exc}", flush=True)
    return preloaded


def smart_preload_in_background() -> None:
    """Kick a daemon thread that warms any predicted embedders not yet loaded.

    Idempotent: embedders whose model is already in memory are skipped.
    Failures are swallowed because this is a best-effort optimisation -
    the real load path will retry on first use.
    """
    import threading

    def _run() -> None:
        from vtscore.media import get_embedder

        for emb_name in predict_embedders_to_preload():
            try:
                emb = get_embedder(emb_name)
                if getattr(emb, "_model", None) is not None:
                    continue
                emb.load_models()
            except Exception:
                pass

    threading.Thread(target=_run, name="smart-preload", daemon=True).start()


def predict_embedder_for_dataset(dataset_id: str) -> str:
    """Predict the embedder name needed by *dataset_id*.

    Mirrors the per-dataset half of :func:`predict_embedders_to_preload`
    for a single registry entry: returns ``entry["embedder"]`` if set and
    recognised, otherwise the default embedder for ``entry["media_type"]``
    (also the fallback when ``embedder`` is set but unrecognised).
    Returns ``""`` when the dataset is unknown or has no resolvable
    embedder.
    """
    from vtscore.datasets.registry import get_dataset
    from vtscore.media import all_embedders, embedders_for_type

    entry = get_dataset(dataset_id)
    if entry is None:
        return ""

    valid = {e.name for e in all_embedders()}
    emb = (entry.get("embedder") or "").strip()
    if emb and emb in valid:
        return emb
    media_type = entry.get("media_type", "") or ""
    if not media_type:
        return ""
    opts = embedders_for_type(media_type)
    return opts[0].name if opts else ""


def preload_embedder_for_dataset(dataset_id: str) -> str:
    """Warm the embedder needed by *dataset_id* in a background daemon thread.

    Used by the dashboard when the user selects a dataset row so the
    embedder is ready by the time they click Train. Idempotent: if the
    embedder is already loaded, the worker exits immediately. Returns
    the embedder name being warmed, or ``""`` when no embedder can be
    resolved for the given dataset.
    """
    import threading

    emb_name = predict_embedder_for_dataset(dataset_id)
    if not emb_name:
        return ""

    def _run() -> None:
        from vtscore.media import get_embedder

        try:
            emb = get_embedder(emb_name)
            if getattr(emb, "_model", None) is not None:
                return
            emb.load_models()
        except Exception:
            pass

    threading.Thread(target=_run, name=f"preload-ds-{dataset_id[:8]}", daemon=True).start()
    return emb_name


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
#
# These return the model instances held by their respective embedder objects.
# Existing callers that import these functions directly continue to work.
# ---------------------------------------------------------------------------


def get_clap_model():
    """Return ``(clap_model, clap_processor)`` from the CLAP embedder."""
    from vtscore.media import get_embedder

    emb = get_embedder("clap")
    # _get_model_and_processor is defined on the CLAP subclass, not the ABC.
    return cast(Any, emb)._get_model_and_processor()


def get_xclip_model():
    """Return ``(xclip_model, xclip_processor)`` from the X-CLIP embedder."""
    from vtscore.media import get_embedder

    emb = get_embedder("xclip")
    return cast(Any, emb)._get_model_and_processor()


def get_e5_model():
    """Return the E5 ``SentenceTransformer`` from the E5 embedder."""
    from vtscore.media import get_embedder

    emb = get_embedder("e5")
    return cast(Any, emb)._get_model()
