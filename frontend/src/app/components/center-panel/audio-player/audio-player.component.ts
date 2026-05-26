import {
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
  ViewChild,
  AfterViewInit,
} from '@angular/core';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-audio-player',
  standalone: true,
  templateUrl: './audio-player.component.html',
  styleUrl: './audio-player.component.scss',
})
export class AudioPlayerComponent implements OnChanges, OnDestroy, AfterViewInit {
  @Input() media!: Media;
  @Input() volume = 1;
  @Input() audioPlaying = true;
  @Input() swipeClass = '';
  @Output() playingChanged = new EventEmitter<boolean>();

  @ViewChild('waveformCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('audioEl') audioRef!: ElementRef<HTMLAudioElement>;

  audioSrc = '';

  private audioCtx: AudioContext | null = null;
  private waveformAbort: AbortController | null = null;
  private viewReady = false;
  // See ImageViewerComponent.lastMediaId: the `media` input reference changes
  // whenever the metadata cache hydrates; without this guard, every cache
  // refresh would re-`loadAudio()` and snap playback back to t=0.
  private lastMediaId: number | null = null;

  constructor(private activeContext: ActiveContextService) {}

  ngAfterViewInit(): void {
    this.viewReady = true;
    if (this.media) this.lastMediaId = this.media.id;
    this.loadAudio();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media && this.media.id !== this.lastMediaId) {
      this.lastMediaId = this.media.id;
      this.audioSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/audio`);
      if (this.viewReady) {
        this.loadAudio();
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
    this.waveformAbort?.abort();
    this.waveformAbort = null;
    const audio = this.audioRef?.nativeElement;
    if (audio) {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      this.audioCtx.close();
    }
  }

  onVolumeChange(): void {
    if (this.audioRef?.nativeElement) {
      this.volume = this.audioRef.nativeElement.volume;
    }
  }

  onPlay(): void {
    if (!this.audioPlaying) {
      this.playingChanged.emit(true);
    }
  }

  onPause(): void {
    if (this.audioPlaying) {
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
    if (this.audioPlaying && audio.paused) {
      audio.play().catch(() => {});
    } else if (!this.audioPlaying && !audio.paused) {
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

      if (!this.audioCtx || this.audioCtx.state === 'closed') {
        this.audioCtx = new AudioContext();
      }
      const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

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
