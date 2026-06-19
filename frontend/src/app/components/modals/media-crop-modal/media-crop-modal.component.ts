import { Component, OnDestroy, OnInit, input, output } from '@angular/core';

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
  imports: [ModalComponent, ImageCropOverlayComponent, AudioCropOverlayComponent],
  templateUrl: './media-crop-modal.component.html',
  styleUrl: './media-crop-modal.component.scss',
})
export class MediaCropModalComponent implements OnInit, OnDestroy {
  readonly file = input.required<File>();
  readonly mediaType = input('');
  readonly confirmed = output<MediaCropResult>();
  readonly cancelled = output<void>();

  view: View = 'confirm';
  fileUrl = '';

  ngOnInit(): void {
    const file = this.file();
    if (file) {
      this.fileUrl = URL.createObjectURL(file);
    }
  }

  ngOnDestroy(): void {
    if (this.fileUrl) {
      URL.revokeObjectURL(this.fileUrl);
    }
  }

  get cropSupported(): boolean {
    const mediaType = this.mediaType();
    return mediaType === 'image' || mediaType === 'audio';
  }

  onOk(): void {
    this.confirmed.emit({ file: this.file() });
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
      file: this.file(),
      cropParams: { box: result.box },
    });
  }

  onAudioCropApplied(result: AudioCropResult): void {
    this.confirmed.emit({
      file: this.file(),
      cropParams: { start: result.start, end: result.end },
    });
  }

  onCropOverlayCancelled(): void {
    this.view = 'confirm';
  }
}
