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
} from '@angular/core';
import { NgStyle } from '@angular/common';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

export type RegionBox = readonly [number, number, number, number];
export type ResizeHandle = 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se';

type DragMode =
  | { kind: 'pan'; startX: number; startY: number; originX: number; originY: number }
  | { kind: 'draw'; anchor: { x: number; y: number }; previousBox: RegionBox | null }
  | { kind: 'move'; startLocal: { x: number; y: number }; startBox: RegionBox }
  | { kind: 'resize'; handle: ResizeHandle; startBox: RegionBox };

const MIN_BOX_SIZE = 0.01; // 1% of the image; below this we treat a draw as a stray click

@Component({
  selector: 'vt-image-viewer',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './image-viewer.component.html',
  styleUrl: './image-viewer.component.scss',
})
export class ImageViewerComponent implements OnChanges, OnDestroy {
  @Input() media!: Media;
  /**
   * True while the parent is in the v2 "bad-vote-with-box discard confirm" armed state.
   * The viewer uses it to (a) render the box with a sticky red pulse, and (b) route Esc /
   * mouse-on-box back to the parent via `armedConfirmCanceled` instead of clearing the box.
   */
  @Input() pendingBadConfirm = false;
  @Output() regionBoxChange = new EventEmitter<RegionBox | null>();
  /**
   * Fired when the user does something that cancels the armed bad-vote-confirm without
   * voting (Esc while armed, or any mousedown on the box body/handles, or starting a
   * fresh Shift-drag). The parent clears its armed state but keeps the box.
   */
  @Output() armedConfirmCanceled = new EventEmitter<void>();

  @ViewChild('imageWrap') wrapRef!: ElementRef<HTMLDivElement>;
  @ViewChild('imageEl') imageRef!: ElementRef<HTMLImageElement>;

  imageSrc = '';
  imageReady = false;
  zoom = 1;
  rotation = 0;
  zoomLabel = '1×';
  // Track the id of the media we last reset for. The `media` input reference
  // changes whenever `MediaMetadataCacheService` hydrates richer metadata for
  // the same id; re-running ngOnChanges for those enrichments would clobber
  // `imageReady=true` back to false and — since `imageSrc` is the same string
  // — Angular wouldn't re-fire the `<img>` load event, leaving the canvas
  // permanently hidden behind `visibility: hidden`.
  private lastMediaId: number | null = null;

  // Region voting state (v2 of the patch-embedder plan, UI only — see docs/plans/patch-embedder.md).
  regionBox: RegionBox | null = null;
  regionBoxShake = false;
  shiftHeld = false;
  // Sticky toggle exposed by the Marquee button in .image-view-controls. While true the
  // viewer behaves as if Shift were held: cursor is a crosshair and a left-drag draws a
  // new region instead of panning. Shift+drag remains a power-user shortcut even when
  // the toggle is off; toggling the button just turns the gesture on without a modifier.
  marqueeMode = false;
  renderedW = 0;
  renderedH = 0;

  // panX/panY are not `private` so tests can drive screenToImageNormalized()
  // with non-zero pan without simulating a full wheel + drag sequence.
  panX = 0;
  panY = 0;
  private drag: DragMode | null = null;

  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private mouseUpHandler: (() => void) | null = null;
  private keyDownHandler: ((e: KeyboardEvent) => void) | null = null;
  private keyUpHandler: ((e: KeyboardEvent) => void) | null = null;
  private blurHandler: (() => void) | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private shakeTimer: ReturnType<typeof setTimeout> | null = null;

  readonly minZoom = 1;
  readonly maxZoom = 5;
  readonly zoomStep = 0.05;

  constructor(private activeContext: ActiveContextService) {
    this.setupWindowKeyListeners();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media'] && this.media && this.media.id !== this.lastMediaId) {
      this.lastMediaId = this.media.id;
      this.imageReady = false;
      this.imageSrc = this.activeContext.mediaUrl(`/api/medias/${this.media.id}/image`);
      this.resetView();
      this.clearRegionBox({ emit: true });
    }
  }

  onImageLoad(): void {
    this.imageReady = true;
    this.recomputeRenderedSize();
    this.attachWrapResizeObserver();
  }

  onImageError(): void {
    this.imageReady = true;
  }

  ngOnDestroy(): void {
    this.removeWindowMouseListeners();
    this.removeWindowKeyListeners();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
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

  /** True when a drag should draw a region (either Shift-held or Marquee toggle on). */
  get regionDrawActive(): boolean {
    return this.shiftHeld || this.marqueeMode;
  }

  toggleMarqueeMode(): void {
    this.marqueeMode = !this.marqueeMode;
  }

  onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;

    if (this.regionDrawActive && this.renderedW > 0 && this.renderedH > 0) {
      // Drag from anywhere on the canvas (Shift-held or marquee mode) starts a fresh
      // box. If an older box existed, discard it before recording the new anchor.
      const local = this.screenToImageNormalized(event);
      if (!local) return;
      const x = clamp01(local.x);
      const y = clamp01(local.y);
      if (this.pendingBadConfirm) this.armedConfirmCanceled.emit();
      // Remember the prior box so we can restore it on a zero-area release —
      // a stray Shift-click on empty space must not throw away real work.
      this.drag = { kind: 'draw', anchor: { x, y }, previousBox: this.regionBox };
      this.regionBox = [x, y, x, y];
      event.preventDefault();
      this.setupWindowMouseListeners();
      return;
    }

    // Default: pan-when-zoomed.
    const max = this.getMaxPan();
    if (max.x <= 0 && max.y <= 0) return;
    this.drag = {
      kind: 'pan',
      startX: event.clientX,
      startY: event.clientY,
      originX: this.panX,
      originY: this.panY,
    };
    event.preventDefault();
    this.setupWindowMouseListeners();
  }

  onRegionBodyMouseDown(event: MouseEvent): void {
    if (event.button !== 0 || !this.regionBox) return;
    event.stopPropagation();
    event.preventDefault();
    const local = this.screenToImageNormalized(event);
    if (!local) return;
    if (this.pendingBadConfirm) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'move', startLocal: local, startBox: this.regionBox };
    this.setupWindowMouseListeners();
  }

  onResizeHandleMouseDown(handle: ResizeHandle, event: MouseEvent): void {
    if (event.button !== 0 || !this.regionBox) return;
    event.stopPropagation();
    event.preventDefault();
    if (this.pendingBadConfirm) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'resize', handle, startBox: this.regionBox };
    this.setupWindowMouseListeners();
  }

  /** Clear the current region box and notify the parent. */
  clearRegionBox(opts: { emit: boolean } = { emit: true }): void {
    if (this.regionBox === null) return;
    this.regionBox = null;
    if (opts.emit) this.regionBoxChange.emit(null);
  }

  /** Visually flash the region box (used by bad-vote-confirm flow). */
  pulseRegionBox(): void {
    if (!this.regionBox) return;
    this.regionBoxShake = true;
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
    this.shakeTimer = setTimeout(() => (this.regionBoxShake = false), 500);
  }

  get imageTransform(): string {
    return `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom}) rotate(${this.rotation}deg)`;
  }

  get wrapCursor(): string {
    if (this.regionDrawActive) return 'crosshair';
    const max = this.getMaxPan();
    return max.x > 0 || max.y > 0 ? 'grab' : '';
  }

  get regionBoxStyle(): { [k: string]: string } | null {
    if (!this.regionBox) return null;
    const [x0, y0, x1, y1] = this.regionBox;
    return {
      left: pct(x0),
      top: pct(y0),
      width: pct(x1 - x0),
      height: pct(y1 - y0),
    };
  }

  private applyTransform(): void {
    const max = this.getMaxPan();
    this.panX = Math.max(-max.x, Math.min(max.x, this.panX));
    this.panY = Math.max(-max.y, Math.min(max.y, this.panY));
    const zoomVal = this.zoom;
    this.zoomLabel = (zoomVal === Math.floor(zoomVal) ? zoomVal.toFixed(0) : zoomVal.toFixed(1)) + '×';
  }

  private clampZoom(val: number): number {
    return Math.min(this.maxZoom, Math.max(this.minZoom, val));
  }

  private recomputeRenderedSize(): void {
    const img = this.imageRef?.nativeElement;
    const wrap = this.wrapRef?.nativeElement;
    if (!img || !wrap) return;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    if (!natW || !natH || !wrapW || !wrapH) {
      this.renderedW = 0;
      this.renderedH = 0;
      return;
    }
    const imgAspect = natW / natH;
    const wrapAspect = wrapW / wrapH;
    if (imgAspect > wrapAspect) {
      this.renderedW = wrapW;
      this.renderedH = wrapW / imgAspect;
    } else {
      this.renderedH = wrapH;
      this.renderedW = wrapH * imgAspect;
    }
  }

  private attachWrapResizeObserver(): void {
    const wrap = this.wrapRef?.nativeElement;
    if (!wrap || this.resizeObserver) return;
    if (typeof ResizeObserver === 'undefined') return;
    this.resizeObserver = new ResizeObserver(() => this.recomputeRenderedSize());
    this.resizeObserver.observe(wrap);
  }

  private getMaxPan(): { x: number; y: number } {
    const wrap = this.wrapRef?.nativeElement;
    if (!wrap || !this.renderedW || !this.renderedH) return { x: 0, y: 0 };
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    const rot = ((this.rotation % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
    const effW = swapped ? this.renderedH : this.renderedW;
    const effH = swapped ? this.renderedW : this.renderedH;
    return {
      x: Math.max(0, (effW * this.zoom - wrapW) / 2),
      y: Math.max(0, (effH * this.zoom - wrapH) / 2),
    };
  }

  /** Convert a screen-space mouse event to normalised image coords (pre-rotation).
   *  Returns null when the image isn't laid out yet. Public so tests can drive it
   *  with a mocked wrapRef + arbitrary pan/zoom/rotate state. */
  screenToImageNormalized(event: MouseEvent): { x: number; y: number } | null {
    const wrap = this.wrapRef?.nativeElement;
    if (!wrap || !this.renderedW || !this.renderedH) return null;
    const rect = wrap.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2) - this.panX;
    const dy = event.clientY - (rect.top + rect.height / 2) - this.panY;
    const sx = dx / this.zoom;
    const sy = dy / this.zoom;
    const rad = (-this.rotation * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const rx = sx * cos - sy * sin;
    const ry = sx * sin + sy * cos;
    return {
      x: (rx + this.renderedW / 2) / this.renderedW,
      y: (ry + this.renderedH / 2) / this.renderedH,
    };
  }

  private setupWindowMouseListeners(): void {
    this.removeWindowMouseListeners();
    this.mouseMoveHandler = (e: MouseEvent) => this.onWindowMouseMove(e);
    this.mouseUpHandler = () => this.onWindowMouseUp();
    window.addEventListener('mousemove', this.mouseMoveHandler);
    window.addEventListener('mouseup', this.mouseUpHandler);
  }

  private removeWindowMouseListeners(): void {
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler);
      this.mouseMoveHandler = null;
    }
    if (this.mouseUpHandler) {
      window.removeEventListener('mouseup', this.mouseUpHandler);
      this.mouseUpHandler = null;
    }
  }

  private onWindowMouseMove(e: MouseEvent): void {
    if (!this.drag) return;
    const d = this.drag;
    if (d.kind === 'pan') {
      this.panX = d.originX + (e.clientX - d.startX);
      this.panY = d.originY + (e.clientY - d.startY);
      this.applyTransform();
      return;
    }
    const local = this.screenToImageNormalized(e);
    if (!local) return;
    if (d.kind === 'draw') {
      const ax = d.anchor.x;
      const ay = d.anchor.y;
      const bx = clamp01(local.x);
      const by = clamp01(local.y);
      this.regionBox = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
      return;
    }
    if (d.kind === 'move') {
      const dx = local.x - d.startLocal.x;
      const dy = local.y - d.startLocal.y;
      const [sx0, sy0, sx1, sy1] = d.startBox;
      const w = sx1 - sx0;
      const h = sy1 - sy0;
      const x0 = clamp(sx0 + dx, 0, 1 - w);
      const y0 = clamp(sy0 + dy, 0, 1 - h);
      this.regionBox = [x0, y0, x0 + w, y0 + h];
      return;
    }
    // resize
    let [x0, y0, x1, y1] = d.startBox;
    const lx = clamp01(local.x);
    const ly = clamp01(local.y);
    if (d.handle.includes('n')) y0 = Math.min(ly, y1 - MIN_BOX_SIZE);
    if (d.handle.includes('s')) y1 = Math.max(ly, y0 + MIN_BOX_SIZE);
    if (d.handle.includes('w')) x0 = Math.min(lx, x1 - MIN_BOX_SIZE);
    if (d.handle.includes('e')) x1 = Math.max(lx, x0 + MIN_BOX_SIZE);
    this.regionBox = [x0, y0, x1, y1];
  }

  private onWindowMouseUp(): void {
    if (!this.drag) {
      this.removeWindowMouseListeners();
      return;
    }
    const drag = this.drag;
    this.drag = null;
    this.removeWindowMouseListeners();
    if (!this.regionBox) return;
    const [x0, y0, x1, y1] = this.regionBox;
    const tooSmall = x1 - x0 < MIN_BOX_SIZE || y1 - y0 < MIN_BOX_SIZE;
    if (drag.kind === 'draw' && tooSmall) {
      // Zero-area Shift-drag (a stray click without motion). Restore the
      // prior box rather than discarding it — drawing a box is real work.
      // Don't emit: the parent's last-known state was already previousBox
      // (the transient zero-area draw was never emitted).
      this.regionBox = drag.previousBox;
      return;
    }
    this.regionBoxChange.emit(this.regionBox);
  }

  private setupWindowKeyListeners(): void {
    this.keyDownHandler = (e: KeyboardEvent) => this.onWindowKeyDown(e);
    this.keyUpHandler = (e: KeyboardEvent) => this.onWindowKeyUp(e);
    this.blurHandler = () => {
      // Releasing focus (alt-tab, etc.) drops the Shift state — don't leave the
      // user stuck in region mode invisibly.
      this.shiftHeld = false;
    };
    window.addEventListener('keydown', this.keyDownHandler);
    window.addEventListener('keyup', this.keyUpHandler);
    window.addEventListener('blur', this.blurHandler);
  }

  private removeWindowKeyListeners(): void {
    if (this.keyDownHandler) {
      window.removeEventListener('keydown', this.keyDownHandler);
      this.keyDownHandler = null;
    }
    if (this.keyUpHandler) {
      window.removeEventListener('keyup', this.keyUpHandler);
      this.keyUpHandler = null;
    }
    if (this.blurHandler) {
      window.removeEventListener('blur', this.blurHandler);
      this.blurHandler = null;
    }
  }

  private onWindowKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Shift') {
      this.shiftHeld = true;
      return;
    }
    if (e.key !== 'Escape' || this.isTyping()) return;
    // Esc while a bad-vote-with-box discard is armed cancels the armed state but
    // keeps the box — per the v2 patch-embedder plan, drawing a box is real work
    // and Esc should be the "I changed my mind about voting no" out, not "throw
    // away the box". Only consume the key if we actually had an action to take.
    if (this.pendingBadConfirm) {
      e.preventDefault();
      this.armedConfirmCanceled.emit();
      return;
    }
    if (this.regionBox) {
      e.preventDefault();
      this.clearRegionBox({ emit: true });
    }
  }

  private onWindowKeyUp(e: KeyboardEvent): void {
    if (e.key === 'Shift') this.shiftHeld = false;
  }

  private isTyping(): boolean {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const type = (el as HTMLInputElement).type;
      if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
    }
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if ((el as HTMLElement).isContentEditable) return true;
    return false;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

function pct(v: number): string {
  return (v * 100).toFixed(3) + '%';
}
