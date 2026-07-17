import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';

/**
 * Wire an ``<audio>`` element to play only a media's clip window
 * (``[clip_start, clip_end]``) when it carries one, instead of the whole
 * archive member the server serves. For non-windowed media (``clip_start ==
 * null``) it falls back to native full-file looping.
 *
 * Used by the two hover-preview players (the VTSBrowse canvas hexagons and the
 * bin popup's member grid), which — unlike the full {@link
 * ../components/center-panel/audio-player/audio-player.component} — drive
 * playback off inline element handlers rather than template bindings.
 *
 * The clip extents are read *lazily* from ``lookup`` inside the element's
 * ``loadedmetadata`` / ``timeupdate`` handlers rather than captured up front:
 * batch metadata hydration for the hovered item may still be in flight when
 * playback starts, but it will have landed by the time the audio has loaded.
 *
 * Call {@link clearClipWindow} on stop/cleanup to detach the handlers.
 */
export function applyClipWindow(
  el: HTMLAudioElement,
  lookup: () => MediaBatchResponse | undefined,
): void {
  el.onloadedmetadata = () => {
    const clipStart = lookup()?.clip_start;
    // Native loop only for non-windowed media; windowed clips loop manually via
    // the timeupdate handler below so they stay within [clip_start, clip_end].
    el.loop = clipStart == null;
    if (clipStart != null) el.currentTime = clipStart;
  };
  el.ontimeupdate = () => {
    const media = lookup();
    const clipStart = media?.clip_start;
    if (clipStart == null) return;
    const clipEnd = media?.clip_end;
    if (el.currentTime < clipStart || (clipEnd != null && el.currentTime >= clipEnd)) {
      el.currentTime = clipStart;
    }
  };
}

/** Detach the clip-window handlers wired by {@link applyClipWindow}. */
export function clearClipWindow(el: HTMLAudioElement): void {
  el.onloadedmetadata = null;
  el.ontimeupdate = null;
}

/**
 * Playback position of ``el`` as a fraction in ``[0, 1]`` of the clip the
 * now-playing waveform depicts, or ``null`` when no finite duration is known
 * yet (so the caller can hide the playhead until playback has a meaningful
 * position).
 *
 * The fraction is measured across the *window* the thumbnail shows, mirroring
 * {@link applyClipWindow} and the server's ``generate_waveform_thumbnail_window``
 * (which slices the PNG to ``[clip_start, clip_end]``): a windowed clip's
 * playhead sweeps ``[clip_start, clip_end]`` edge-to-edge, and a non-windowed
 * clip's sweeps ``[0, duration]``. This keeps the sweeping line aligned with the
 * waveform pixels the user actually sees.
 */
export function clipProgress(
  el: HTMLAudioElement,
  lookup: () => MediaBatchResponse | undefined,
): number | null {
  const duration = el.duration;
  if (!Number.isFinite(duration) || duration <= 0) return null;
  const media = lookup();
  const start = media?.clip_start ?? 0;
  const end = media?.clip_end ?? duration;
  const span = end - start;
  if (span <= 0) return null;
  return Math.max(0, Math.min(1, (el.currentTime - start) / span));
}
