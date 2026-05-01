import {
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { MediaItem } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  selector: 'vt-image-viewer',
  standalone: true,
  templateUrl: './image-viewer.component.html',
  styleUrl: './image-viewer.component.scss',
})
export class ImageViewerComponent implements OnChanges, OnDestroy {
  @Input() media!: MediaItem;

  @ViewChild('imageWrap') wrapRef!: ElementRef<HTMLDivElement>;
  @ViewChild('imageEl') imageRef!: ElementRef<HTMLImageElement>;

  imageSrc = '';

  constructor(private activeContext: ActiveContextService) {}
  imageReady = false;
  zoom = 1;
  rotation = 0;
  zoomLabel = '1\u00d7';

  private panX = 0;
  private panY = 0;
  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private panOriginX = 0;
  private panOriginY = 0;
  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private mouseUpHandler: (() => void) | null = null;

  readonly minZoom = 1;
  readonly maxZoom = 5;
  readonly zoomStep = 0.05;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media) {
      this.imageReady = false;
      this.imageSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/image`);
      this.resetView();
    }
  }

  onImageLoad(): void {
    this.imageReady = true;
  }

  onImageError(): void {
    this.imageReady = true;
  }

  ngOnDestroy(): void {
    this.removeWindowListeners();
  }

  onZoomInput(event: Event): void {
    this.zoom = parseFloat((event.target as HTMLInputElement).value);
    this.applyTransform();
  }

  rotateLeft(): void {
    this.rotation -= 90;
    this.applyTransform();
  }

  rotateRight(): void {
    this.rotation += 90;
    this.applyTransform();
  }

  zoomIn(): void {
    this.zoom = this.clampZoom(this.zoom + 0.15 * this.zoom);
    this.applyTransform();
  }

  zoomOut(): void {
    this.zoom = this.clampZoom(this.zoom - 0.15 * this.zoom);
    this.applyTransform();
  }

  resetView(): void {
    this.zoom = 1;
    this.rotation = 0;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
  }

  onWheel(event: WheelEvent): void {
    event.preventDefault();
    const oldZoom = this.zoom;
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    this.zoom = this.clampZoom(this.zoom + delta * this.zoom);

    const wrap = this.wrapRef?.nativeElement;
    if (wrap) {
      const rect = wrap.getBoundingClientRect();
      const cx = event.clientX - rect.left - rect.width / 2;
      const cy = event.clientY - rect.top - rect.height / 2;
      const ratio = this.zoom / oldZoom;
      this.panX = cx - ratio * (cx - this.panX);
      this.panY = cy - ratio * (cy - this.panY);
    }
    this.applyTransform();
  }

  onMouseDown(event: MouseEvent): void {
    const max = this.getMaxPan();
    if ((max.x <= 0 && max.y <= 0) || event.button !== 0) return;
    this.isPanning = true;
    this.panStartX = event.clientX;
    this.panStartY = event.clientY;
    this.panOriginX = this.panX;
    this.panOriginY = this.panY;
    event.preventDefault();
    this.setupWindowListeners();
  }

  get imageTransform(): string {
    return `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom}) rotate(${this.rotation}deg)`;
  }

  get wrapCursor(): string {
    const max = this.getMaxPan();
    return (max.x > 0 || max.y > 0) ? 'grab' : '';
  }

  private applyTransform(): void {
    const max = this.getMaxPan();
    this.panX = Math.max(-max.x, Math.min(max.x, this.panX));
    this.panY = Math.max(-max.y, Math.min(max.y, this.panY));
    const zoomVal = this.zoom;
    this.zoomLabel = (zoomVal === Math.floor(zoomVal) ? zoomVal.toFixed(0) : zoomVal.toFixed(1)) + '\u00d7';
  }

  private clampZoom(val: number): number {
    return Math.min(this.maxZoom, Math.max(this.minZoom, val));
  }

  private getMaxPan(): { x: number; y: number } {
    const img = this.imageRef?.nativeElement;
    const wrap = this.wrapRef?.nativeElement;
    if (!img || !wrap) return { x: 0, y: 0 };

    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    if (!natW || !natH) return { x: 0, y: 0 };

    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    if (!wrapW || !wrapH) return { x: 0, y: 0 };

    const imgAspect = natW / natH;
    const wrapAspect = wrapW / wrapH;
    let rendW: number, rendH: number;
    if (imgAspect > wrapAspect) {
      rendW = wrapW;
      rendH = wrapW / imgAspect;
    } else {
      rendH = wrapH;
      rendW = wrapH * imgAspect;
    }

    const rot = ((this.rotation % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
    const effW = swapped ? rendH : rendW;
    const effH = swapped ? rendW : rendH;

    return {
      x: Math.max(0, (effW * this.zoom - wrapW) / 2),
      y: Math.max(0, (effH * this.zoom - wrapH) / 2),
    };
  }

  private setupWindowListeners(): void {
    this.removeWindowListeners();
    this.mouseMoveHandler = (e: MouseEvent) => {
      if (!this.isPanning) return;
      this.panX = this.panOriginX + (e.clientX - this.panStartX);
      this.panY = this.panOriginY + (e.clientY - this.panStartY);
      this.applyTransform();
    };
    this.mouseUpHandler = () => {
      this.isPanning = false;
    };
    window.addEventListener('mousemove', this.mouseMoveHandler);
    window.addEventListener('mouseup', this.mouseUpHandler);
  }

  private removeWindowListeners(): void {
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler);
      this.mouseMoveHandler = null;
    }
    if (this.mouseUpHandler) {
      window.removeEventListener('mouseup', this.mouseUpHandler);
      this.mouseUpHandler = null;
    }
  }
}
