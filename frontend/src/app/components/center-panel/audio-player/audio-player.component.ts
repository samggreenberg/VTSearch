import { ChangeDetectionStrategy, Component, effect, ElementRef, inject, input, linkedSignal, OnDestroy, output, signal, untracked, viewChild } from '@angular/core';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import { decodeAudioBuffer } from '../../../utils/decode-audio';

/** Downsampled per-column waveform extrema for one clip at one canvas width. */
interface WaveformPeaks {
  min: Float32Array;
  max: Float32Array;
}

// Module-level LRU cache of downsampled waveform peaks, keyed by
// `${datasetId}:${mediaId}:${width}`. Peaks are a few KB (two Float32Arrays of
// `width` samples); the decoded AudioBuffer they derive from is megabytes, so
// we cache the peaks and re-derive nothing on re-selection -- the expensive
// `decodeAudioData` pass is skipped when a clip is revisited. The cache is
// module-level (not per-instance) so re-selecting a clip after navigating away
// -- which destroys and recreates the player -- still hits it. Bounded so a
// long voting session can't grow it without limit. The width is part of the
// key because peaks are computed at the canvas width; a resize simply recomputes.
const PEAKS_CACHE_LIMIT = 20;
const peaksCache = new Map<string, WaveformPeaks>();

function getCachedPeaks(key: string): WaveformPeaks | undefined {
  const hit = peaksCache.get(key);
  if (hit !== undefined) {
    // LRU touch: re-insert so this key is treated as most-recently-used.
    peaksCache.delete(key);
    peaksCache.set(key, hit);
  }
  return hit;
}

function putCachedPeaks(key: string, peaks: WaveformPeaks): void {
  peaksCache.delete(key);
  peaksCache.set(key, peaks);
  while (peaksCache.size > PEAKS_CACHE_LIMIT) {
    const oldest = peaksCache.keys().next().value;
    if (oldest === undefined) break;
    peaksCache.delete(oldest);
  }
}

/** Downsample one channel's samples to per-column (min, max) extrema. */
function computePeaks(channelData: Float32Array, width: number): WaveformPeaks {
  const min = new Float32Array(width);
  const max = new Float32Array(width);
  const step = Math.max(1, Math.ceil(channelData.length / width));
  for (let i = 0; i < width; i++) {
    let lo = 1.0;
    let hi = -1.0;
    const base = i * step;
    for (let j = 0; j < step; j++) {
      const datum = channelData[base + j];
      if (datum < lo) lo = datum;
      if (datum > hi) hi = datum;
    }
    min[i] = lo;
    max[i] = hi;
  }
  return { min, max };
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-audio-player',
  standalone: true,
  templateUrl: './audio-player.component.html',
  styleUrl: './audio-player.component.scss',
})
export class AudioPlayerComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);

  readonly media = input.required<Media>();
  readonly volume = input(1);
  readonly audioPlaying = input(true);
  readonly swipeClass = input('');
  readonly playingChanged = output<boolean>();

  private readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('waveformCanvas');
  // Public: CenterPanelComponent reaches through this to pause/resume the
  // native element on navigation / tab-visibility changes.
  readonly audioRef = viewChild<ElementRef<HTMLAudioElement>>('audioEl');

  // The volume the <audio> element should currently have. Resets to the
  // `volume` input whenever the parent pushes a new value; element-driven
  // changes (the native volume slider, keyboard adjustVolume) are mirrored
  // back into it so later loads restore the user's last-set volume.
  private readonly currentVolume = linkedSignal(() => this.volume());

  // Playhead sweep state. `playheadFraction` is currentTime/duration in [0, 1];
  // the template binds it to the overlay line's `left: %`. `playheadVisible`
  // gates rendering until a finite duration is known (the line has no meaningful
  // position before (loadedmetadata)). Both are driven imperatively: on a
  // requestAnimationFrame loop while playing (see startSweep) -- (timeupdate)
  // fires only ~4x/sec, too coarse for a smooth sweep -- and on demand after a
  // seek or pause.
  readonly playheadFraction = signal(0);
  readonly playheadVisible = signal(false);
  // Handle of the in-flight rAF sweep tick, or null when the loop is stopped.
  private sweepRaf: number | null = null;
  // True while a scrub gesture (pointerdown on the waveform) is in progress, so
  // (pointermove) seeks continuously until pointerup.
  private scrubbing = false;

  // Current object URL feeding the <audio> element (`blob:...`), or '' before
  // the first load. Set imperatively on the native element (not via a template
  // binding) because it's produced asynchronously after the fetch, and the app
  // is zoneless. Revoked before it's replaced and on destroy.
  audioSrc = '';

  private waveformAbort: AbortController | null = null;
  // See ImageViewerComponent.lastMediaId: the `media` input reference changes
  // whenever the metadata cache hydrates; without this guard, every cache
  // refresh would re-`loadAudio()` and snap playback back to t=0.
  private lastMediaId: number | null = null;
  // Active clip window bounds while enforcement is on, else null. Enforcement is
  // driven by the <audio> element's (timeupdate) event rather than a polling
  // timer, so it only runs while the clip is actually playing and progressing.
  private clipBounds: { start: number; end: number } | null = null;
  // Whether (loadedmetadata) has fired for the current audioSrc. Clip bounds
  // (clip_start/clip_end) often arrive via batch hydration *after* the audio
  // has already loaded, on a later media change with the same media id; in that
  // case (loadedmetadata) does not fire again, so we (re)apply the bounds here.
  // Mirrors VideoPlayerComponent: archive-member audio windows serve the whole
  // member and seek/loop within [clip_start, clip_end] (display-only).
  private metadataLoaded = false;

  constructor() {
    // (Re)load the clip when the media id changes. Tracks the view query so
    // the first load runs as soon as the <audio> element exists (view queries
    // resolve after the first render); until then the effect is a no-op and
    // `lastMediaId` stays null, so the pending media loads on resolution.
    effect(() => {
      const media = this.media();
      if (!this.audioRef()) return;
      if (media.id !== this.lastMediaId) {
        this.lastMediaId = media.id;
        this.metadataLoaded = false;
        this.stopClipEnforcement();
        untracked(() => void this.loadAudio(media));
      } else if (this.metadataLoaded) {
        // Same media id, but metadata (e.g. clip_start/clip_end) may have just
        // arrived via batch hydration. The audio already loaded, so
        // (loadedmetadata) won't fire again; apply the clip window now.
        untracked(() => this.applyClipBounds());
      }
    });

    // Push volume changes onto the element. The element is read untracked so
    // this runs only when the volume actually changes, mirroring the old
    // `changes['volume']` guard; the initial volume is applied by
    // (loadedmetadata) / loadAudio(), as before.
    effect(() => {
      const vol = this.currentVolume();
      const audio = untracked(this.audioRef)?.nativeElement;
      if (audio) audio.volume = vol;
    });

    // Play/pause the element when the parent toggles `audioPlaying`. Untracked
    // element read for the same reason: a media swap must not re-trigger this
    // (the old code's `!changes['media']` guard) — the post-load sync happens
    // in loadAudio()/(loadedmetadata) instead.
    effect(() => {
      this.audioPlaying();
      if (untracked(this.audioRef)) untracked(() => this.syncPlaybackState());
    });
  }

  ngOnDestroy(): void {
    this.stopClipEnforcement();
    this.stopSweep();
    this.waveformAbort?.abort();
    this.waveformAbort = null;
    const audio = this.audioRef()?.nativeElement;
    if (audio) {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    }
    this.revokeObjectUrl();
  }

  onLoadedMetadata(): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;
    this.metadataLoaded = true;
    audio.volume = this.currentVolume();
    this.applyClipBounds();
    this.syncPlaybackState();
    // Duration is now known: place the playhead at the current position and, if
    // playback is already running, start the sweep loop.
    this.updatePlayhead();
    if (!audio.paused) this.startSweep();
  }

  // Seek into the clip window and (re)start boundary enforcement when the media
  // carries clip extents; otherwise tear enforcement down. Safe to call both on
  // (loadedmetadata) and on later metadata-enrichment effect cycles.
  private applyClipBounds(): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;

    const media = this.media();
    if (media.clip_start != null) {
      const clipStart = media.clip_start;
      const clipEnd = media.clip_end;
      // Snap into the window only when currently outside it, so we don't yank
      // audio already looping correctly within its window.
      if (audio.currentTime < clipStart || (clipEnd != null && audio.currentTime >= clipEnd)) {
        audio.currentTime = clipStart;
      }
      this.startClipEnforcement();
    } else {
      this.stopClipEnforcement();
    }
  }

  private startClipEnforcement(): void {
    const media = this.media();
    if (media.clip_start == null || media.clip_end == null) {
      this.clipBounds = null;
      return;
    }
    // Arm enforcement; the actual boundary check runs in (timeupdate), which the
    // <audio> element fires as playback advances (no timer while paused).
    this.clipBounds = { start: media.clip_start, end: media.clip_end };
  }

  private stopClipEnforcement(): void {
    this.clipBounds = null;
  }

  // Enforce the clip window as playback advances: when the current time leaves
  // [start, end), loop back to the window start. Driven by the element's
  // (timeupdate) event instead of a 100ms polling interval.
  onTimeUpdate(): void {
    // Refresh the playhead first, unconditionally: (timeupdate) is the coarse
    // (~4x/sec) fallback that keeps the line moving when the rAF sweep isn't
    // running (headless env, or a background tab that throttles rAF), and it
    // must run whether or not this clip has a window to enforce.
    this.updatePlayhead();

    const bounds = this.clipBounds;
    if (!bounds) return;
    const audio = this.audioRef()?.nativeElement;
    if (!audio || audio.paused) return;
    if (audio.currentTime >= bounds.end || audio.currentTime < bounds.start) {
      audio.currentTime = bounds.start;
    }
  }

  onVolumeChange(): void {
    const audio = this.audioRef()?.nativeElement;
    if (audio) {
      this.currentVolume.set(audio.volume);
    }
  }

  onPlay(): void {
    this.startSweep();
    if (!this.audioPlaying()) {
      this.playingChanged.emit(true);
    }
  }

  onPause(): void {
    this.stopSweep();
    // Settle the playhead on the exact paused position (the last rAF tick may
    // have landed a frame short).
    this.updatePlayhead();
    if (this.audioPlaying()) {
      this.playingChanged.emit(false);
    }
  }

  togglePlayback(): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }

  // --- Playhead sweep + click/drag-to-seek --------------------------------

  // Recompute the playhead position from the element's clock. Hides the line
  // until a finite, positive duration is known (before (loadedmetadata) the
  // fraction is meaningless).
  private updatePlayhead(): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;
    const duration = audio.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      this.playheadVisible.set(false);
      return;
    }
    this.playheadVisible.set(true);
    this.playheadFraction.set(Math.max(0, Math.min(1, audio.currentTime / duration)));
  }

  // Run a requestAnimationFrame loop that advances the playhead ~60x/sec while
  // audio plays. Self-cancels when the element pauses (so a stray pause we
  // didn't route through onPause still stops it). Idempotent: a second call
  // while a loop is live is a no-op.
  private startSweep(): void {
    if (this.sweepRaf !== null) return;
    if (typeof requestAnimationFrame !== 'function') return;
    const tick = (): void => {
      this.updatePlayhead();
      const audio = this.audioRef()?.nativeElement;
      if (audio && !audio.paused) {
        this.sweepRaf = requestAnimationFrame(tick);
      } else {
        this.sweepRaf = null;
      }
    };
    this.sweepRaf = requestAnimationFrame(tick);
  }

  private stopSweep(): void {
    if (this.sweepRaf !== null) {
      cancelAnimationFrame(this.sweepRaf);
      this.sweepRaf = null;
    }
  }

  // Begin a scrub gesture: seek to the pressed position and capture the pointer
  // so drags outside the waveform keep seeking until release.
  onSeekPointerDown(event: PointerEvent): void {
    // Left button / primary pointer only; ignore right-clicks and secondary.
    if (event.button !== 0) return;
    this.scrubbing = true;
    const stage = event.currentTarget as HTMLElement | null;
    stage?.setPointerCapture?.(event.pointerId);
    this.seekToPointer(event.clientX);
    event.preventDefault();
  }

  onSeekPointerMove(event: PointerEvent): void {
    if (!this.scrubbing) return;
    this.seekToPointer(event.clientX);
  }

  onSeekPointerUp(event: PointerEvent): void {
    if (!this.scrubbing) return;
    this.scrubbing = false;
    const stage = event.currentTarget as HTMLElement | null;
    stage?.releasePointerCapture?.(event.pointerId);
  }

  // Map a viewport x-coordinate over the waveform to a playback time and seek
  // there. The waveform spans the whole decoded buffer, so the fraction across
  // the canvas maps linearly to [0, duration]; for a windowed clip the target
  // is clamped into [clip_start, clip_end] (the seekable region).
  private seekToPointer(clientX: number): void {
    const audio = this.audioRef()?.nativeElement;
    const canvas = this.canvasRef()?.nativeElement;
    if (!audio || !canvas) return;
    const duration = audio.duration;
    if (!Number.isFinite(duration) || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return;

    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    let target = fraction * duration;

    const media = this.media();
    if (media.clip_start != null) {
      const end = media.clip_end ?? duration;
      target = Math.max(media.clip_start, Math.min(end, target));
    }
    audio.currentTime = target;
    this.updatePlayhead();
  }

  adjustVolume(delta: number): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;
    audio.volume = Math.max(0, Math.min(1, audio.volume + delta));
    this.currentVolume.set(audio.volume);
  }

  // Download the clip once and feed those bytes to BOTH the <audio> element
  // (via an object URL) and the waveform renderer. Previously the <audio>
  // element streamed /audio while drawWaveform() fetched the identical URL a
  // second time and decoded it -- two downloads of the same bytes per selection.
  private async loadAudio(media: Media): Promise<void> {
    const mediaId = media.id;
    const datasetId = this.activeContext.datasetId;

    // Blank the old waveform immediately so a media switch doesn't leave the
    // previous clip's render on screen during the fetch. Retract the playhead
    // too -- its old position is meaningless for the incoming clip, and the new
    // duration is unknown until (loadedmetadata).
    this.clearCanvas();
    this.stopSweep();
    this.playheadVisible.set(false);
    this.playheadFraction.set(0);

    this.waveformAbort?.abort();
    const abort = new AbortController();
    this.waveformAbort = abort;

    let blob: Blob;
    try {
      const response = await fetch(this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`), {
        signal: abort.signal,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching audio for media ${mediaId}`);
      }
      blob = await response.blob();
    } catch (err) {
      if (!this.wasAborted(err, abort)) {
        console.warn('audio load failed', err);
        this.drawWaveformError();
      }
      this.clearAbort(abort);
      return;
    }
    if (abort.signal.aborted) {
      this.clearAbort(abort);
      return;
    }

    this.setAudioSource(blob);
    await this.renderWaveform(mediaId, datasetId, blob, abort);
    this.clearAbort(abort);

    const audio = this.audioRef()?.nativeElement;
    if (audio) {
      audio.volume = this.currentVolume();
      this.syncPlaybackState();
    }
  }

  // Point the <audio> element at the freshly downloaded bytes via an object
  // URL, revoking the previous one so blobs don't accumulate.
  private setAudioSource(blob: Blob): void {
    const url = URL.createObjectURL(blob);
    this.revokeObjectUrl();
    this.audioSrc = url;
    const audio = this.audioRef()?.nativeElement;
    if (audio) {
      audio.src = url;
      audio.load();
    }
  }

  private revokeObjectUrl(): void {
    if (this.audioSrc) {
      URL.revokeObjectURL(this.audioSrc);
      this.audioSrc = '';
    }
  }

  private syncPlaybackState(): void {
    const audio = this.audioRef()?.nativeElement;
    if (!audio) return;
    const audioPlaying = this.audioPlaying();
    if (audioPlaying && audio.paused) {
      audio.play().catch(() => {});
    } else if (!audioPlaying && !audio.paused) {
      audio.pause();
    }
  }

  // Prepare the canvas for drawing: size it to its rendered width and paint the
  // background. Returns the 2D context + dimensions, or null when no canvas /
  // 2D context is available (headless test env).
  private beginCanvas(): { ctx: CanvasRenderingContext2D; width: number; height: number } | null {
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0) canvas.width = Math.round(rect.width);
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = this.cssVar('--bg-surface', '#1a1d27');
    ctx.fillRect(0, 0, width, height);
    return { ctx, width, height };
  }

  private clearCanvas(): void {
    this.beginCanvas();
  }

  private async renderWaveform(
    mediaId: number,
    datasetId: string,
    blob: Blob,
    abort: AbortController,
  ): Promise<void> {
    const info = this.beginCanvas();
    if (!info) return;
    const { ctx, width, height } = info;

    const key = `${datasetId}:${mediaId}:${width}`;
    let peaks = getCachedPeaks(key);
    if (peaks === undefined) {
      let buffer: AudioBuffer;
      try {
        buffer = await decodeAudioBuffer(await blob.arrayBuffer());
      } catch (err) {
        if (!this.wasAborted(err, abort)) {
          console.warn('waveform decode failed', err);
          this.drawWaveformError();
        }
        return;
      }
      if (abort.signal.aborted) return;
      peaks = computePeaks(buffer.getChannelData(0), width);
      putCachedPeaks(key, peaks);
    }
    this.drawPeaks(ctx, peaks, width, height);
  }

  private drawPeaks(ctx: CanvasRenderingContext2D, peaks: WaveformPeaks, width: number, height: number): void {
    const amp = height / 2;

    ctx.strokeStyle = this.cssVar('--accent', '#7c8aff');
    ctx.lineWidth = 1;
    ctx.beginPath();
    const n = Math.min(width, peaks.min.length);
    for (let i = 0; i < n; i++) {
      const yMin = (1 + peaks.min[i]) * amp;
      const yMax = (1 + peaks.max[i]) * amp;
      if (i === 0) ctx.moveTo(i, yMin);
      ctx.lineTo(i, yMin);
      ctx.lineTo(i, yMax);
    }
    ctx.stroke();

    // Center line
    ctx.strokeStyle = this.cssVar('--border', '#2a2d3a');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }

  private drawWaveformError(): void {
    const info = this.beginCanvas();
    if (!info) return;
    const { ctx, width, height } = info;
    ctx.fillStyle = this.cssVar('--color-bad', '#f44336');
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Unable to load waveform', width / 2, height / 2);
  }

  private cssVar(name: string, fallback: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  private clearAbort(abort: AbortController): void {
    if (this.waveformAbort === abort) {
      this.waveformAbort = null;
    }
  }

  private wasAborted(err: unknown, abort: AbortController): boolean {
    return abort.signal.aborted || (err instanceof DOMException && err.name === 'AbortError');
  }
}
