import {
  AfterViewInit,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ImageCropResult {
  /** Crop box in original-image pixel coordinates: [x1, y1, x2, y2]. */
  box: [number, number, number, number];
}

type DragMode = 'none' | 'move' | 'tl' | 'tr' | 'bl' | 'br' | 'l' | 'r' | 't' | 'b';

const HANDLE_HIT_RADIUS = 12;

@Component({
  selector: 'vt-image-crop-overlay',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-crop-overlay.component.html',
  styleUrl: './image-crop-overlay.component.scss',
})
export class ImageCropOverlayComponent implements AfterViewInit {
  @Input() imageUrl = '';
  @Output() applied = new EventEmitter<ImageCropResult>();
  @Output() cancelled = new EventEmitter<void>();

  @ViewChild('img') imgRef!: ElementRef<HTMLImageElement>;

  /** Crop box in *displayed* (CSS) pixel coordinates relative to the image element. */
  cropX = 0;
  cropY = 0;
  cropW = 0;
  cropH = 0;

  private naturalW = 0;
  private naturalH = 0;
  private displayW = 0;
  private displayH = 0;

  private dragMode: DragMode = 'none';
  private dragStartX = 0;
  private dragStartY = 0;
  private origCrop = { x: 0, y: 0, w: 0, h: 0 };

  ngAfterViewInit(): void {
    // Image may already be loaded if cached.
    const img = this.imgRef.nativeElement;
    if (img.complete && img.naturalWidth > 0) {
      this.onImageLoaded();
    }
  }

  onImageLoaded(): void {
    const img = this.imgRef.nativeElement;
    this.naturalW = img.naturalWidth;
    this.naturalH = img.naturalHeight;
    this.displayW = img.clientWidth;
    this.displayH = img.clientHeight;
    // Default crop: centred 60% box.
    this.cropW = Math.round(this.displayW * 0.6);
    this.cropH = Math.round(this.displayH * 0.6);
    this.cropX = Math.round((this.displayW - this.cropW) / 2);
    this.cropY = Math.round((this.displayH - this.cropH) / 2);
  }

  onPointerDown(event: PointerEvent): void {
    const rect = this.imgRef.nativeElement.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    this.dragMode = this.hitTest(px, py);
    if (this.dragMode === 'none') return;
    this.dragStartX = px;
    this.dragStartY = py;
    this.origCrop = { x: this.cropX, y: this.cropY, w: this.cropW, h: this.cropH };
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  onPointerMove(event: PointerEvent): void {
    if (this.dragMode === 'none') return;
    const rect = this.imgRef.nativeElement.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const dx = px - this.dragStartX;
    const dy = py - this.dragStartY;
    const o = this.origCrop;

    let x = o.x;
    let y = o.y;
    let w = o.w;
    let h = o.h;

    switch (this.dragMode) {
      case 'move':
        x = o.x + dx;
        y = o.y + dy;
        break;
      case 'tl':
        x = o.x + dx;
        y = o.y + dy;
        w = o.w - dx;
        h = o.h - dy;
        break;
      case 'tr':
        y = o.y + dy;
        w = o.w + dx;
        h = o.h - dy;
        break;
      case 'bl':
        x = o.x + dx;
        w = o.w - dx;
        h = o.h + dy;
        break;
      case 'br':
        w = o.w + dx;
        h = o.h + dy;
        break;
      case 'l':
        x = o.x + dx;
        w = o.w - dx;
        break;
      case 'r':
        w = o.w + dx;
        break;
      case 't':
        y = o.y + dy;
        h = o.h - dy;
        break;
      case 'b':
        h = o.h + dy;
        break;
    }

    // Enforce minimum size 5px.
    if (w < 5) {
      if (this.dragMode === 'l' || this.dragMode === 'tl' || this.dragMode === 'bl') {
        x = o.x + o.w - 5;
      }
      w = 5;
    }
    if (h < 5) {
      if (this.dragMode === 't' || this.dragMode === 'tl' || this.dragMode === 'tr') {
        y = o.y + o.h - 5;
      }
      h = 5;
    }
    // Clamp to image bounds.
    if (x < 0) {
      w += x;
      x = 0;
    }
    if (y < 0) {
      h += y;
      y = 0;
    }
    if (x + w > this.displayW) w = this.displayW - x;
    if (y + h > this.displayH) h = this.displayH - y;

    this.cropX = x;
    this.cropY = y;
    this.cropW = w;
    this.cropH = h;
  }

  onPointerUp(event: PointerEvent): void {
    if (this.dragMode === 'none') return;
    this.dragMode = 'none';
    (event.target as HTMLElement).releasePointerCapture(event.pointerId);
  }

  private hitTest(px: number, py: number): DragMode {
    const x1 = this.cropX;
    const y1 = this.cropY;
    const x2 = this.cropX + this.cropW;
    const y2 = this.cropY + this.cropH;
    const r = HANDLE_HIT_RADIUS;

    const nearTop = Math.abs(py - y1) < r;
    const nearBot = Math.abs(py - y2) < r;
    const nearLeft = Math.abs(px - x1) < r;
    const nearRight = Math.abs(px - x2) < r;

    if (nearTop && nearLeft) return 'tl';
    if (nearTop && nearRight) return 'tr';
    if (nearBot && nearLeft) return 'bl';
    if (nearBot && nearRight) return 'br';
    if (nearTop && px > x1 && px < x2) return 't';
    if (nearBot && px > x1 && px < x2) return 'b';
    if (nearLeft && py > y1 && py < y2) return 'l';
    if (nearRight && py > y1 && py < y2) return 'r';
    if (px >= x1 && px <= x2 && py >= y1 && py <= y2) return 'move';
    return 'none';
  }

  apply(): void {
    if (this.displayW === 0 || this.displayH === 0) return;
    const sx = this.naturalW / this.displayW;
    const sy = this.naturalH / this.displayH;
    const x1 = Math.max(0, Math.round(this.cropX * sx));
    const y1 = Math.max(0, Math.round(this.cropY * sy));
    const x2 = Math.min(this.naturalW, Math.round((this.cropX + this.cropW) * sx));
    const y2 = Math.min(this.naturalH, Math.round((this.cropY + this.cropH) * sy));
    this.applied.emit({ box: [x1, y1, x2, y2] });
  }

  cancel(): void {
    this.cancelled.emit();
  }
}
