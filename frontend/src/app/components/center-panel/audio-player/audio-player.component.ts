import { AfterViewInit, ChangeDetectionStrategy, Component, ElementRef, inject, Input, input, OnChanges, OnDestroy, output, SimpleChanges, ViewChild } from '@angular/core';
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
export class AudioPlayerComponent implements OnChanges, OnDestroy, AfterViewInit {
  private activeContext = inject(ActiveContextService);

  @Input() media!: Media;
  @Input() volume = 1;
  readonly audioPlaying = input(true);
  readonly swipeClass = input('');
  readonly playingChanged = output<boolean>();

  @ViewChild('waveformCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('audioEl') audioRef!: ElementRef<HTMLAudioElement>;

  // Current object URL feeding the <audio> element (`blob:...`), or '' before
  // the first load. Set imperatively on the native element (not via a template
  // binding) because it's produced asynchronously after the fetch, and the app
  // is zoneless. Revoked before it's replaced and on destroy.
  audioSrc = '';

  private waveformAbort: AbortController | null = null;
  private viewReady = false;
  // See ImageViewerComponent.lastMediaId: the `media` input reference changes
  // whenever the metadata cache hydrates; without this guard, every cache
  // refresh would re-`loadAudio()` and snap playback back to t=0.
  private lastMediaId: number | null = null;
  private clipCheckInterval: ReturnType<typeof setInterval> | null = null;
  // Whether (loadedmetadata) has fired for the current audioSrc. Clip bounds
  // (clip_start/clip_end) often arrive via batch hydration *after* the audio
  // has already loaded, on a later ngOnChanges with the same media id; in that
  // case (loadedmetadata) does not fire again, so we (re)apply the bounds here.
  // Mirrors VideoPlayerComponent: archive-member audio windows serve the whole
  // member and seek/loop within [clip_start, clip_end] (display-only).
  private metadataLoaded = false;

  ngAfterViewInit(): void {
    this.viewReady = true;
    if (this.media) this.lastMediaId = this.media.id;
    this.loadAudio();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      if (this.media.id !== this.lastMediaId) {
        this.lastMediaId = this.media.id;
        this.metadataLoaded = false;
        this.stopClipEnforcement();
        if (this.viewReady) {
          this.loadAudio();
        }
      } else if (this.metadataLoaded) {
        // Same media id, but metadata (e.g. clip_start/clip_end) may have just
        // arrived via batch hydration. The audio already loaded, so
        // (loadedmetadata) won't fire again; apply the clip window now.
        this.applyClipBounds();
      }
    }
    if (changes['volume'] && this.audioRef?.nativeElement) {
      this.audioRef.nativeElement.volume = this.volume;
    }
    if (changes['audioPlaying'] && !changes['media'] && this.audioRef?.nativeElement) {
      this.syncPlaybackState();
    }
  }

  ngOnDestroy(): void {
    this.stopClipEnforcement();
    this.waveformAbort?.abort();
    this.waveformAbort = null;
    const audio = this.audioRef?.nativeElement;
    if (audio) {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    }
    this.revokeObjectUrl();
  }

  onLoadedMetadata(): void {
    const audio = this.audioRef?.nativeElement;
    if (!audio) return;
    this.metadataLoaded = true;
    audio.volume = this.volume;
    this.applyClipBounds();
    this.syncPlaybackState();
  }

  // Seek into the clip window and (re)start boundary enforcement when the media
  // carries clip extents; otherwise tear enforcement down. Safe to call both on
  // (loadedmetadata) and on later metadata-enrichment ngOnChanges cycles.
  private applyClipBounds(): void {
    const audio = this.audioRef?.nativeElement;
    if (!audio) return;

    if (this.media?.clip_start != null) {
      const clipStart = this.media.clip_start;
      const clipEnd = this.media.clip_end;
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
    this.stopClipEnforcement();
    if (this.media?.clip_start == null || this.media?.clip_end == null) return;

    const clipStart = this.media.clip_start;
    const clipEnd = this.media.clip_end;

    // Poll every 100ms to enforce clip boundaries. When the audio reaches
    // clip_end, loop back to clip_start instead of continuing.
    this.clipCheckInterval = setInterval(() => {
      const audio = this.audioRef?.nativeElement;
      if (!audio || audio.paused) return;
      if (audio.currentTime >= clipEnd || audio.currentTime < clipStart) {
        audio.currentTime = clipStart;
      }
    }, 100);
  }

  private stopClipEnforcement(): void {
    if (this.clipCheckInterval != null) {
      clearInterval(this.clipCheckInterval);
      this.clipCheckInterval = null;
    }
  }

  onVolumeChange(): void {
    if (this.audioRef?.nativeElement) {
      this.volume = this.audioRef.nativeElement.volume;
    }
  }

  onPlay(): void {
    if (!this.audioPlaying()) {
      this.playingChanged.emit(true);
    }
  }

  onPause(): void {
    if (this.audioPlaying()) {
      this.playingChanged.emit(false);
    }
  }

  togglePlayback(): void {
    const audio = this.audioRef?.nativeElement;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }

  adjustVolume(delta: number): void {
    const audio = this.audioRef?.nativeElement;
    if (!audio) return;
    audio.volume = Math.max(0, Math.min(1, audio.volume + delta));
    this.volume = audio.volume;
  }

  // Download the clip once and feed those bytes to BOTH the <audio> element
  // (via an object URL) and the waveform renderer. Previously the <audio>
  // element streamed /audio while drawWaveform() fetched the identical URL a
  // second time and decoded it -- two downloads of the same bytes per selection.
  private async loadAudio(): Promise<void> {
    if (!this.media) return;
    const mediaId = this.media.id;
    const datasetId = this.activeContext.datasetId;

    // Blank the old waveform immediately so a media switch doesn't leave the
    // previous clip's render on screen during the fetch.
    this.clearCanvas();

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

    const audio = this.audioRef?.nativeElement;
    if (audio) {
      audio.volume = this.volume;
      this.syncPlaybackState();
    }
  }

  // Point the <audio> element at the freshly downloaded bytes via an object
  // URL, revoking the previous one so blobs don't accumulate.
  private setAudioSource(blob: Blob): void {
    const url = URL.createObjectURL(blob);
    this.revokeObjectUrl();
    this.audioSrc = url;
    const audio = this.audioRef?.nativeElement;
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
    const audio = this.audioRef?.nativeElement;
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
    const canvas = this.canvasRef?.nativeElement;
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
