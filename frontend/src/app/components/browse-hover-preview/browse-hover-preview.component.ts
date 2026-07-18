import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, output, signal, untracked } from '@angular/core';

import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { applyClipWindow, clearClipWindow, clipProgress } from '../../utils/clip-window';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';

/** A clip currently auditioning from a VTSBrowse hover (a canvas bin here, or a
 *  bin-popup member — see `browse-bin-popup.component.ts`), or ``null`` when
 *  nothing is playing. Drives the now-playing waveform anchored at the
 *  top-left of the canvas, alongside the volume control. */
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

/** How long (ms) the cursor must rest on an audio bin before its clip starts
 *  auditioning, so sweeping across bins doesn't fire a burst of plays. */
const AUDIO_DWELL_MS = 200;

/**
 * The VTSBrowse hover preview.
 *
 * For **audio** it starts auditioning the hovered bin's clip after a short
 * dwell. There is no on-canvas UI of its own: the bin itself lifts (the
 * canvas's hover-enlarge, shared with every thumbnail type — see
 * `usesThumbnails()`), and the now-playing waveform + volume control anchored
 * at the top-left of the canvas (`browse-view.component`) is the only visible
 * feedback that sound is playing. Hover clearing stops it immediately.
 *
 * For **text** it shows the small paragraph popup; **image/video** paint
 * their thumbnail on the tile, so nothing pops up.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-hover-preview',
  standalone: true,
  imports: [],
  templateUrl: './browse-hover-preview.component.html',
  styleUrl: './browse-hover-preview.component.scss',
})
export class BrowseHoverPreviewComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);
  private metadataCache = inject(MediaMetadataCacheService);

  readonly hover = input<HexHoverEvent | null>(null);
  readonly mediaType = input('');
  /** Preview-audio volume (0–1), driven by the Browse toolbar's volume control. */
  readonly volume = input(1);
  /** The clip now auditioning (for the top-left now-playing indicator), or
   *  ``null`` once the hover clears. */
  readonly nowPlaying = output<NowPlaying | null>();

  /** Never mounted in the DOM — audio here is heard, not shown; the top-left
   *  indicator is the only visual feedback that it's playing. */
  private readonly audioEl = new Audio();
  /** Waveform PNG of the clip currently auditioning, kept so the buffering
   *  listeners below can re-emit the now-playing state without recomputing it. */
  private nowPlayingWaveUrl = '';
  /** Last ``loading`` flag emitted, so the playhead sweep re-emits with the
   *  live buffering state rather than forcing it back to ``false``. */
  private nowPlayingLoading = false;
  /** Handle of the in-flight requestAnimationFrame playhead sweep, or ``null``
   *  when the loop is stopped. */
  private sweepRaf: number | null = null;

  constructor() {
    // The hover input drives the whole preview: show over a cell, hide on
    // null. (The old ngOnChanges arm — signal inputs don't fire it.) The body
    // runs untracked so its mediaType/metadata reads don't widen the trigger
    // beyond the hover itself.
    effect(() => {
      const hover = this.hover();
      untracked(() => (hover ? this.show(hover) : this.hide()));
    });
    // Keep the live element's volume in sync when the toolbar slider moves
    // mid-playback; starting a clip also seeds it.
    effect(() => {
      this.audioEl.volume = this.volume();
    });
    // Buffering feedback: while the clip is fetching/decoding (or stalls to
    // rebuffer), the widget shows a spinner; once it's actually sounding, the
    // spinner clears. Listeners are attached once to the reused element and
    // guarded by ``playingMediaId`` so stray events after a stop are ignored.
    this.audioEl.addEventListener('waiting', () => this.emitNowPlaying(true));
    this.audioEl.addEventListener('playing', () => this.emitNowPlaying(false));
    this.audioEl.addEventListener('canplay', () => this.emitNowPlaying(false));
  }

  // --- Text / count popup (text + any other non-thumbnail type) --------------
  visible = false;
  left = 0;
  top = 0;
  count = 0;
  // Written from the async paragraph `fetch().then()` continuation, so a signal
  // so the late write repaints the popup body under zoneless.
  readonly textContent = signal('');
  private textLoadAbort: AbortController | null = null;

  // --- Audio audition state machine -------------------------------------------
  private dwellTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingMediaId: number | null = null;
  private playingMediaId: number | null = null;

  ngOnDestroy(): void {
    this.cancelDwell();
    this.stopSweep();
    this.stopAudio();
    this.textLoadAbort?.abort();
  }

  private show(event: HexHoverEvent): void {
    const mediaType = this.mediaType();
    // Image and video paint their thumbnail directly onto the tile; nothing
    // pops up on hover.
    if (mediaType === 'image' || mediaType === 'video') {
      this.hide();
      return;
    }

    if (mediaType === 'audio') {
      this.scheduleAudio(event.cell.rep_id);
      return;
    }

    // text / other: the lightweight text-or-count popup (immediate, no dwell).
    this.visible = true;
    this.left = event.screenX + 16;
    this.top = event.screenY - 8;
    this.count = event.cell.count;
    const rep = event.cell.rep_id;
    if (mediaType === 'text') {
      this.loadText(rep);
    } else {
      this.textContent.set(`Item #${rep}`);
    }
  }

  private hide(): void {
    // Text popup: hide immediately.
    this.visible = false;
    this.textLoadAbort?.abort();
    this.textLoadAbort = null;
    this.textContent.set('');

    // Audio: cancel a pending audition and stop anything already playing. There
    // is no panel to bridge the cursor onto, so hover-off silences it at once.
    this.cancelDwell();
    this.stopAudio();
  }

  // --- Audio audition state machine -------------------------------------------

  private scheduleAudio(mediaId: number): void {
    // Already auditioning this exact clip: keep it (don't restart).
    if (this.playingMediaId === mediaId) return;

    // Debounce: (re)arm the dwell so a sweep across bins keeps resetting it and
    // only the bin the cursor settles on actually plays.
    this.pendingMediaId = mediaId;
    this.cancelDwell();
    this.dwellTimer = setTimeout(() => {
      this.dwellTimer = null;
      const id = this.pendingMediaId;
      this.pendingMediaId = null;
      if (id != null) this.playAudio(id);
    }, AUDIO_DWELL_MS);
  }

  private playAudio(mediaId: number): void {
    this.playingMediaId = mediaId;
    const waveUrl = this.activeContext.mediaUrl(`/api/medias/${mediaId}/thumbnail`);
    this.nowPlayingWaveUrl = waveUrl;
    // Hydrate clip extents (clip_start/clip_end) so windowed clips loop within
    // their window; applyClipWindow reads them lazily as they land.
    this.metadataCache.ensureLoaded([mediaId]);
    this.audioEl.volume = this.volume();
    applyClipWindow(this.audioEl, () => this.metadataCache.get(mediaId));
    this.audioEl.src = this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`);
    this.audioEl.load();
    this.audioEl.play().catch(() => {});
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
    this.nowPlayingLoading = loading;
    const progress = clipProgress(this.audioEl, () => this.metadataCache.get(mediaId));
    this.nowPlaying.emit({ mediaId, waveUrl: this.nowPlayingWaveUrl, loading, progress });
  }

  // Re-emit the now-playing state ~60×/sec so the playhead sweeps smoothly:
  // (timeupdate) fires only ~4×/sec, too coarse for a fluid line. Self-cancels
  // when the clip pauses or stops; idempotent while a loop is already live.
  private startSweep(): void {
    if (this.sweepRaf !== null || typeof requestAnimationFrame !== 'function') return;
    const tick = (): void => {
      if (this.playingMediaId == null || this.audioEl.paused) {
        this.sweepRaf = null;
        return;
      }
      this.emitNowPlaying(this.nowPlayingLoading);
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

  private stopAudio(): void {
    this.pendingMediaId = null;
    if (this.playingMediaId == null) return;
    this.playingMediaId = null;
    this.stopSweep();
    clearClipWindow(this.audioEl);
    this.audioEl.pause();
    this.audioEl.currentTime = 0;
    this.nowPlaying.emit(null);
  }

  private cancelDwell(): void {
    if (this.dwellTimer) {
      clearTimeout(this.dwellTimer);
      this.dwellTimer = null;
    }
  }

  private loadText(mediaId: number): void {
    this.textContent.set('Loading…');
    // Cancel any in-flight text load so a slow earlier response can't clobber
    // the preview after the cursor has moved to another hex.
    this.textLoadAbort?.abort();
    const abort = new AbortController();
    this.textLoadAbort = abort;
    const url = this.activeContext.mediaUrl(`/api/medias/${mediaId}/paragraph`);
    fetch(url, { signal: abort.signal })
      .then((r) => r.json())
      .then((data) => {
        if (this.hover()?.cell.rep_id === mediaId) {
          const text: string = data.content || '';
          this.textContent.set(text.length > 300 ? text.slice(0, 300) + '…' : text);
        }
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
        if (this.hover()?.cell.rep_id === mediaId) {
          this.textContent.set(`Item #${mediaId}`);
        }
      });
  }
}
