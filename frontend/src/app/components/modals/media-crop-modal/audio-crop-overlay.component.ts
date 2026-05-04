import {
  AfterViewInit,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnDestroy,
  Output,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';

export interface AudioCropResult {
  start: number;
  end: number;
}

type DragMode = 'none' | 'start' | 'end' | 'move';

const HANDLE_HIT_PX = 12;

@Component({
  selector: 'vt-audio-crop-overlay',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './audio-crop-overlay.component.html',
  styleUrl: './audio-crop-overlay.component.scss',
})
export class AudioCropOverlayComponent implements AfterViewInit, OnDestroy {
  @Input() audioUrl = '';
  /** Original audio file — used to decode the waveform. */
  @Input() audioFile?: File;
  @Output() applied = new EventEmitter<AudioCropResult>();
  @Output() cancelled = new EventEmitter<void>();

  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  duration = 0;
  start = 0;
  end = 0;
  loading = true;
  errorMsg = '';

  private dragMode: DragMode = 'none';
  private dragOffsetSec = 0;
  private peaks: number[] = [];

  async ngAfterViewInit(): Promise<void> {
    if (!this.audioFile) {
      this.loading = false;
      return;
    }
    try {
      const arrayBuffer = await this.audioFile.arrayBuffer();
      const AudioCtx = (window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
      const ctx = new AudioCtx();
      const buffer = await ctx.decodeAudioData(arrayBuffer);
      this.duration = buffer.duration;
      this.start = 0;
      this.end = this.duration;
      this.peaks = this.computePeaks(buffer, 600);
      this.loading = false;
      requestAnimationFrame(() => this.draw());
      ctx.close();
    } catch (e) {
      this.errorMsg = 'Could not decode audio waveform; you can still crop by dragging.';
      this.loading = false;
      // Fall back to a flat waveform with the audio element's reported duration.
      this.peaks = new Array(600).fill(0.05);
      requestAnimationFrame(() => this.draw());
    }
  }

  ngOnDestroy(): void {
    // No persistent resources beyond the AudioContext (already closed).
  }

  /** Set duration from the <audio> element when it becomes available. */
  onAudioLoadedMetadata(event: Event): void {
    const a = event.target as HTMLAudioElement;
    if (a.duration && Number.isFinite(a.duration)) {
      if (this.duration === 0) {
        this.duration = a.duration;
        this.end = a.duration;
        requestAnimationFrame(() => this.draw());
      }
    }
  }

  private computePeaks(buffer: AudioBuffer, n: number): number[] {
    const channelData = buffer.getChannelData(0);
    const samplesPerBin = Math.max(1, Math.floor(channelData.length / n));
    const peaks: number[] = [];
    for (let i = 0; i < n; i++) {
      let max = 0;
      const offset = i * samplesPerBin;
      for (let j = 0; j < samplesPerBin && offset + j < channelData.length; j++) {
        const v = Math.abs(channelData[offset + j]);
        if (v > max) max = v;
      }
      peaks.push(max);
    }
    return peaks;
  }

  private draw(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    // Waveform.
    if (this.peaks.length > 0) {
      const mid = height / 2;
      const barW = width / this.peaks.length;
      ctx.fillStyle = '#888';
      for (let i = 0; i < this.peaks.length; i++) {
        const h = this.peaks[i] * (height - 4);
        ctx.fillRect(i * barW, mid - h / 2, Math.max(1, barW - 0.5), h);
      }
    }

    if (this.duration <= 0) return;
    const x1 = (this.start / this.duration) * width;
    const x2 = (this.end / this.duration) * width;

    // Mask outside the selection.
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(0, 0, x1, height);
    ctx.fillRect(x2, 0, width - x2, height);

    // Selection border.
    ctx.strokeStyle = '#2962ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 + 1, 1, x2 - x1 - 2, height - 2);

    // Handles.
    ctx.fillStyle = '#2962ff';
    ctx.fillRect(x1 - 3, 0, 6, height);
    ctx.fillRect(x2 - 3, 0, 6, height);
  }

  onPointerDown(event: PointerEvent): void {
    if (this.duration <= 0) return;
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const sec = (px / rect.width) * this.duration;

    const x1 = (this.start / this.duration) * rect.width;
    const x2 = (this.end / this.duration) * rect.width;

    if (Math.abs(px - x1) < HANDLE_HIT_PX) {
      this.dragMode = 'start';
    } else if (Math.abs(px - x2) < HANDLE_HIT_PX) {
      this.dragMode = 'end';
    } else if (px > x1 && px < x2) {
      this.dragMode = 'move';
      this.dragOffsetSec = sec - this.start;
    } else {
      // Click outside the selection — reposition closer handle.
      const distToStart = Math.abs(px - x1);
      const distToEnd = Math.abs(px - x2);
      if (distToStart < distToEnd) {
        this.start = Math.max(0, Math.min(sec, this.end - 0.05));
        this.dragMode = 'start';
      } else {
        this.end = Math.max(this.start + 0.05, Math.min(sec, this.duration));
        this.dragMode = 'end';
      }
      this.draw();
    }

    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  onPointerMove(event: PointerEvent): void {
    if (this.dragMode === 'none' || this.duration <= 0) return;
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const sec = Math.max(0, Math.min((px / rect.width) * this.duration, this.duration));

    switch (this.dragMode) {
      case 'start':
        this.start = Math.max(0, Math.min(sec, this.end - 0.05));
        break;
      case 'end':
        this.end = Math.max(this.start + 0.05, Math.min(sec, this.duration));
        break;
      case 'move': {
        const width = this.end - this.start;
        let newStart = sec - this.dragOffsetSec;
        if (newStart < 0) newStart = 0;
        if (newStart + width > this.duration) newStart = this.duration - width;
        this.start = newStart;
        this.end = newStart + width;
        break;
      }
    }
    this.draw();
  }

  onPointerUp(event: PointerEvent): void {
    if (this.dragMode === 'none') return;
    this.dragMode = 'none';
    this.canvasRef.nativeElement.releasePointerCapture(event.pointerId);
  }

  apply(): void {
    if (this.duration <= 0 || this.end <= this.start) return;
    this.applied.emit({ start: this.start, end: this.end });
  }

  cancel(): void {
    this.cancelled.emit();
  }

  formatTime(sec: number): string {
    if (!Number.isFinite(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = sec - m * 60;
    return `${m}:${s.toFixed(2).padStart(5, '0')}`;
  }
}
