import { AfterViewInit, ChangeDetectionStrategy, Component, ElementRef, inject, Input, input, OnChanges, OnDestroy, output, SimpleChanges, ViewChild } from '@angular/core';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import { decodeAudioBuffer } from '../../../utils/decode-audio';

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
        this.audioSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/audio`);
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

  private async loadAudio(): Promise<void> {
    if (!this.media || !this.canvasRef) return;
    await this.drawWaveform(this.media.id);
    if (this.audioRef?.nativeElement) {
      this.audioRef.nativeElement.volume = this.volume;
      this.syncPlaybackState();
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

  private async drawWaveform(mediaId: number): Promise<void> {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0) canvas.width = Math.round(rect.width);
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-surface').trim() || '#1a1d27';
    ctx.fillRect(0, 0, width, height);

    this.waveformAbort?.abort();
    const abort = new AbortController();
    this.waveformAbort = abort;

    try {
      const response = await fetch(this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`), {
        signal: abort.signal,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} fetching audio for media ${mediaId}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      if (abort.signal.aborted) return;

      const audioBuffer = await decodeAudioBuffer(arrayBuffer);
      if (abort.signal.aborted) return;

      const channelData = audioBuffer.getChannelData(0);
      const step = Math.ceil(channelData.length / width);
      const amp = height / 2;

      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#7c8aff';
      ctx.lineWidth = 1;
      ctx.beginPath();

      for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        for (let j = 0; j < step; j++) {
          const datum = channelData[i * step + j];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }
        const yMin = (1 + min) * amp;
        const yMax = (1 + max) * amp;
        if (i === 0) ctx.moveTo(i, yMin);
        ctx.lineTo(i, yMin);
        ctx.lineTo(i, yMax);
      }
      ctx.stroke();

      // Center line
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#2a2d3a';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    } catch (err) {
      if (abort.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
        return;
      }
      console.warn('waveform render failed', err);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-bad').trim() || '#f44336';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Unable to load waveform', width / 2, height / 2);
    } finally {
      if (this.waveformAbort === abort) {
        this.waveformAbort = null;
      }
    }
  }
}
