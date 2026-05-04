import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { ImageCropOverlayComponent, ImageCropResult } from './image-crop-overlay.component';
import { AudioCropOverlayComponent, AudioCropResult } from './audio-crop-overlay.component';

export type CropParams = { start: number; end: number } | { box: [number, number, number, number] };

export interface MediaCropResult {
  file: File;
  cropParams?: CropParams;
}

type View = 'confirm' | 'cropping';

@Component({
  selector: 'vt-media-crop-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, ImageCropOverlayComponent, AudioCropOverlayComponent],
  templateUrl: './media-crop-modal.component.html',
  styleUrl: './media-crop-modal.component.scss',
})
export class MediaCropModalComponent implements OnInit, OnDestroy {
  @Input() file!: File;
  @Input() mediaType = '';
  @Output() confirmed = new EventEmitter<MediaCropResult>();
  @Output() cancelled = new EventEmitter<void>();

  view: View = 'confirm';
  fileUrl = '';

  ngOnInit(): void {
    if (this.file) {
      this.fileUrl = URL.createObjectURL(this.file);
    }
  }

  ngOnDestroy(): void {
    if (this.fileUrl) {
      URL.revokeObjectURL(this.fileUrl);
    }
  }

  get cropSupported(): boolean {
    return this.mediaType === 'image' || this.mediaType === 'audio';
  }

  onOk(): void {
    this.confirmed.emit({ file: this.file });
  }

  onOkButCrop(): void {
    if (!this.cropSupported) {
      this.onOk();
      return;
    }
    this.view = 'cropping';
  }

  onCancel(): void {
    this.cancelled.emit();
  }

  onImageCropApplied(result: ImageCropResult): void {
    this.confirmed.emit({
      file: this.file,
      cropParams: { box: result.box },
    });
  }

  onAudioCropApplied(result: AudioCropResult): void {
    this.confirmed.emit({
      file: this.file,
      cropParams: { start: result.start, end: result.end },
    });
  }

  onCropOverlayCancelled(): void {
    this.view = 'confirm';
  }
}
