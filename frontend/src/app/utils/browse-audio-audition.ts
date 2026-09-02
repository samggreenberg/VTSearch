import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';
import { applyClipWindow, clearClipWindow, clipProgress } from './clip-window';

/** A clip currently auditioning from a VTSBrowse hover — a canvas bin
 *  (`browse-hover-preview.component.ts`), a selection-panel entry
 *  (`browse-selection-panel.component.ts`), or a bin-popup member
 *  (`browse-bin-popup.component.ts`) — or ``null`` when nothing is playing.
 *  Drives the now-playing waveform anchored at the top-left of the canvas,
 *  alongside the volume control. */
export interface NowPlaying {
  mediaId: number;
  /** The clip's waveform PNG — the same thumbnail painted on its tile. */
  waveUrl: string;
  /** ``true`` while the clip is still fetching/decoding and hasn't begun to
   *  sound yet (or has stalled to rebuffer mid-play); drives the loading
   *  spinner on the now-playing widget. Flips to ``false`` once playback is
   *  actually audible. */
  loading: boolean;
  /** Playback position within the (possibly windowed) clip as a fraction in
   *  ``[0, 1]``, or ``null`` before a finite duration is known. Drives the
   *  sweeping playhead line on the now-playing waveform, updated ~60×/sec on a
   *  requestAnimationFrame loop while the clip sounds. */
  progress: number | null;
}

/** How long (ms) the cursor must rest on an audio item before its clip starts
 *  auditioning, so sweeping across bins (or down a long list) doesn't fire a
 *  burst of plays. The default for hover-driven surfaces; the bin popup passes
 *  ``0`` because opening it is already a deliberate, settled gesture. */
export const AUDIO_DWELL_MS = 200;

/** What the audition needs from its host component, as plain callbacks — this
 *  file is `utils/`, so it takes no Angular DI. */
export interface BrowseAudioAuditionOptions {
  /** Resolve an API path against the active dataset — `ActiveContextService.mediaUrl`. */
  mediaUrl: (path: string) => string;
  /** The cached clip metadata for a media, if it has landed —
   *  `MediaMetadataCacheService.get`. Read *lazily* (never captured), because
   *  hydration may still be in flight when playback starts. */
  lookup: (mediaId: number) => MediaBatchResponse | undefined;
  /** Ask for a media's metadata to be hydrated — `MediaMetadataCacheService.ensureLoaded`. */
  ensureLoaded: (mediaId: number) => void;
  /** Push the now-playing state at the host's `nowPlaying` output. **This is the
   *  zoneless notification path** (`docs/FRONTEND.md` §5): the indicator is
   *  rendered by an ancestor, so an edge that fails to emit leaves the UI
   *  showing stale playback state. */
  emit: (state: NowPlaying | null) => void;
  /** Dwell debounce before a hovered clip starts. Defaults to
   *  {@link AUDIO_DWELL_MS}; ``0`` plays synchronously on {@link
   *  BrowseAudioAudition.hover}. */
  dwellMs?: number;
}

/**
 * The VTSBrowse hover-audition state machine: dwell debounce, stale-event
 * guard, buffering tri-state and playhead sweep, over an audio element it owns.
 *
 * Three Browse surfaces audition clips on hover — the canvas hover preview, the
 * selection panel and the bin popup — and all three drive the *same* top-left
 * now-playing indicator through the shared {@link NowPlaying} output. They used
 * to carry a copy of this machine each; the copies diverged (issue #3436), so
 * the machine lives here and the components keep only their output wiring.
 *
 * The element is created here and **never mounted in the DOM**: browse audio is
 * heard, not shown — the top-left indicator is the only visual feedback — and a
 * media element plays perfectly well detached. Owning it is what lets the
 * source be set imperatively, which in turn is why starting a clip is
 * synchronous rather than deferred behind a template binding.
 *
 * Every path that stops playback emits ``null`` exactly once, and every path
 * that changes the buffering state or the playhead re-emits, so the indicator
 * can never be left asserting a clip that is no longer sounding.
 */
export class BrowseAudioAudition {
  /** The audition element. Detached from the document by design (see the class
   *  docstring); exposed so specs can dispatch its buffering events. */
  readonly element: HTMLAudioElement = new Audio();

  private readonly dwellMs: number;
  private dwellTimer: ReturnType<typeof setTimeout> | null = null;
  /** Media whose dwell is armed but has not yet started, or ``null``. */
  private pendingMediaId: number | null = null;
  /** Media currently auditioning, or ``null`` when silent. Doubles as the
   *  stale-event guard: the buffering listeners below are attached once to the
   *  reused element, so events that arrive after a stop must not resurrect the
   *  widget. */
  private playingMediaId: number | null = null;
  /** Waveform PNG of the clip auditioning, kept so the buffering listeners can
   *  re-emit without recomputing it. */
  private waveUrl = '';
  /** Last ``loading`` flag emitted, so the playhead sweep re-emits with the
   *  live buffering state rather than forcing it back to ``false``. */
  private loading = false;
  /** Handle of the in-flight requestAnimationFrame playhead sweep, or ``null``
   *  when the loop is stopped. */
  private sweepRaf: number | null = null;
  private volume = 1;

  constructor(private readonly opts: BrowseAudioAuditionOptions) {
    this.dwellMs = opts.dwellMs ?? AUDIO_DWELL_MS;
    // Buffering feedback: while the clip is fetching/decoding (or stalls to
    // rebuffer), the widget shows a spinner; once it's actually sounding, the
    // spinner clears.
    this.element.addEventListener('waiting', () => this.emitNowPlaying(true));
    this.element.addEventListener('playing', () => this.emitNowPlaying(false));
    this.element.addEventListener('canplay', () => this.emitNowPlaying(false));
  }

  /** Keep the element in step with the Browse toolbar's volume slider, including
   *  mid-playback. Also seeds the volume a later {@link hover} starts at. */
  setVolume(volume: number): void {
    this.volume = volume;
    this.element.volume = volume;
  }

  /** Whether `mediaId` is the clip auditioning, or the one whose dwell is armed.
   *  Callers use it to silence an item that is about to leave the DOM (and so
   *  will never fire its own mouseleave). */
  isTargeting(mediaId: number): boolean {
    return this.playingMediaId === mediaId || this.pendingMediaId === mediaId;
  }

  /**
   * The cursor settled on `mediaId`: (re)arm the dwell so a sweep across items
   * keeps resetting it and only the one the cursor rests on actually plays.
   * Already auditioning this exact clip is a no-op, so re-entering an item (or
   * a redundant re-summon) doesn't restart it.
   */
  hover(mediaId: number): void {
    if (this.playingMediaId === mediaId) return;
    this.pendingMediaId = mediaId;
    this.cancelDwell();
    if (this.dwellMs === 0) {
      this.pendingMediaId = null;
      this.play(mediaId);
      return;
    }
    this.dwellTimer = setTimeout(() => {
      this.dwellTimer = null;
      const id = this.pendingMediaId;
      this.pendingMediaId = null;
      if (id != null) this.play(id);
    }, this.dwellMs);
  }

  /**
   * Silence the audition: cancel a pending dwell, stop anything playing and
   * emit ``null`` so the indicator clears. Idempotent — a second call while
   * already silent emits nothing.
   *
   * This is the single retreat edge: hover-off, an item leaving the list, a
   * fresh bin replacing the one on show, and teardown all route through it.
   */
  stop(): void {
    this.cancelDwell();
    this.pendingMediaId = null;
    if (this.playingMediaId == null) return;
    this.playingMediaId = null;
    this.stopSweep();
    clearClipWindow(this.element);
    this.element.pause();
    this.element.currentTime = 0;
    this.opts.emit(null);
  }

  /** Tear down with the host component. */
  destroy(): void {
    this.stop();
    this.stopSweep();
  }

  private play(mediaId: number): void {
    this.playingMediaId = mediaId;
    this.waveUrl = this.opts.mediaUrl(`/api/medias/${mediaId}/thumbnail`);
    // Hydrate clip extents (clip_start/clip_end) so windowed clips loop within
    // their window; applyClipWindow reads them lazily as they land.
    this.opts.ensureLoaded(mediaId);
    this.element.volume = this.volume;
    applyClipWindow(this.element, () => this.opts.lookup(mediaId));
    this.element.src = this.opts.mediaUrl(`/api/medias/${mediaId}/audio`);
    this.element.load();
    this.element.play().catch(() => {});
    // Starts loading: show the spinner until a ``playing``/``canplay`` event
    // (wired in the constructor) clears it.
    this.emitNowPlaying(true);
    // Advance the playhead sweep while it sounds; self-cancels on pause/stop.
    this.startSweep();
  }

  /** Re-emit the now-playing state with a fresh ``loading`` flag and the current
   *  playhead position, from the buffering listeners and the sweep loop. A no-op
   *  once nothing is auditioning, so events that fire after a stop don't
   *  resurrect the widget. */
  private emitNowPlaying(loading: boolean): void {
    const mediaId = this.playingMediaId;
    if (mediaId == null) return;
    this.loading = loading;
    const progress = clipProgress(this.element, () => this.opts.lookup(mediaId));
    this.opts.emit({ mediaId, waveUrl: this.waveUrl, loading, progress });
  }

  // Re-emit the now-playing state ~60×/sec so the playhead sweeps smoothly:
  // (timeupdate) fires only ~4×/sec, too coarse for a fluid line. Self-cancels
  // when the clip pauses or stops; idempotent while a loop is already live.
  private startSweep(): void {
    if (this.sweepRaf !== null || typeof requestAnimationFrame !== 'function') return;
    const tick = (): void => {
      if (this.playingMediaId == null || this.element.paused) {
        this.sweepRaf = null;
        return;
      }
      this.emitNowPlaying(this.loading);
      this.sweepRaf = requestAnimationFrame(tick);
    };
    this.sweepRaf = requestAnimationFrame(tick);
  }

  private stopSweep(): void {
    if (this.sweepRaf !== null) {
      cancelAnimationFrame(this.sweepRaf);
      this.sweepRaf = null;
    }
  }

  private cancelDwell(): void {
    if (this.dwellTimer) {
      clearTimeout(this.dwellTimer);
      this.dwellTimer = null;
    }
  }
}
