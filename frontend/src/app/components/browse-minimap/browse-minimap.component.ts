import {
  Component,
  ElementRef,
  EventEmitter,
  Input,
  NgZone,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { Subscription } from 'rxjs';
import { TileCacheService } from '../../services/tile-cache.service';
import { BrowseViewportService, ViewportBounds } from '../../services/browse-viewport.service';
import { densityColor } from '../browse-canvas/hex-render.util';
import { binGeometry } from '../browse-canvas/bin-geometry';
import { IconComponent } from '../icon/icon.component';
import type { HexCellPayload, ProjectionMeta } from '../../models/projection.models';

/** Resize clamps; must mirror the backend ``browse_minimap_*`` setting ranges. */
export const MINIMAP_MIN_WIDTH = 120;
export const MINIMAP_MAX_WIDTH = 600;
export const MINIMAP_MIN_HEIGHT = 90;
export const MINIMAP_MAX_HEIGHT = 450;

/**
 * Lower-right overview for the browse canvas: a density heatmap of the whole
 * projection, with a rectangle marking the region the main canvas is currently
 * showing. Clicking/dragging the minimap recenters the main view; a corner
 * handle resizes it and a close button hides it.
 *
 * The heatmap picks the pyramid level whose hexes land near a small target
 * on-screen size, so the whole projection is shown as a fine-grained field of
 * many hexes (not the handful of level-0 bins). Tiles for that level are read
 * straight from the shared :class:`TileCacheService`, so it reuses whatever the
 * main canvas has already fetched and never holds its own copy.
 */
@Component({
  selector: 'vt-browse-minimap',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './browse-minimap.component.html',
  styleUrl: './browse-minimap.component.scss',
})
export class BrowseMinimapComponent implements OnInit, OnChanges, OnDestroy {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() meta: ProjectionMeta | null = null;
  @Input() width = 200;
  @Input() height = 150;
  /** Hide request from the close button. */
  @Output() closed = new EventEmitter<void>();
  /** Final size after a resize drag, for persistence. */
  @Output() resized = new EventEmitter<{ width: number; height: number }>();

  private ctx!: CanvasRenderingContext2D;
  private dpr = 1;
  private viewportBounds: ViewportBounds = null;

  private tileLoadSub: Subscription | null = null;
  private viewportSub: Subscription | null = null;
  private rafId = 0;
  private needsRedraw = false;

  private resizing = false;
  private resizeStartX = 0;
  private resizeStartY = 0;
  private resizeStartW = 0;
  private resizeStartH = 0;

  private navigating = false;

  private boundResizeMove = this.onResizeMove.bind(this);
  private boundResizeUp = this.onResizeUp.bind(this);
  private boundNavMove = this.onNavMove.bind(this);
  private boundNavUp = this.onNavUp.bind(this);

  constructor(
    private ngZone: NgZone,
    private tileCache: TileCacheService,
    private viewport: BrowseViewportService,
  ) {}

  ngOnInit(): void {
    this.ctx = this.canvasRef.nativeElement.getContext('2d')!;
    this.resizeCanvas();

    // Overview tiles arrive asynchronously; repaint as the cache fills. The
    // viewport box updates on every pan/zoom the main canvas publishes.
    this.tileLoadSub = this.tileCache.tileLoaded$.subscribe(() => this.requestRedraw());
    this.viewportSub = this.viewport.viewport$.subscribe((b) => {
      this.viewportBounds = b;
      this.requestRedraw();
    });

    this.requestOverviewTiles();
    this.requestRedraw();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!this.ctx) return;
    if (changes['width'] || changes['height']) {
      this.resizeCanvas();
      this.requestRedraw();
    }
    if (changes['meta'] && !changes['meta'].firstChange) {
      this.requestOverviewTiles();
      this.requestRedraw();
    }
  }

  ngOnDestroy(): void {
    this.tileLoadSub?.unsubscribe();
    this.viewportSub?.unsubscribe();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.detachResizeListeners();
    this.detachNavListeners();
  }

  private resizeCanvas(): void {
    this.dpr = window.devicePixelRatio || 1;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = Math.round(this.width * this.dpr);
    canvas.height = Math.round(this.height * this.dpr);
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  // --- Overview-tile coverage -----------------------------------------------

  /**
   * Target on-screen radius (minimap px) for the overview's hexes. Small enough
   * that the whole projection reads as a fine field of many bins rather than
   * the few large level-0 hexes, while still resolving structure at a glance.
   */
  private static readonly OVERVIEW_TARGET_HEX_PX = 5;

  /**
   * Pyramid level to render in the overview: the one whose hexes are nearest
   * the small target on-screen size for the current minimap scale. Deeper than
   * level 0 (so the map is finer-grained), clamped to the available levels.
   */
  private overviewLevel(f: { scale: number }): number {
    if (!this.meta || this.meta.levels.length === 0) return 0;
    const basePx = this.meta.base_radius * f.scale; // level-0 hex radius in px
    const ideal = Math.log2(basePx / BrowseMinimapComponent.OVERVIEW_TARGET_HEX_PX);
    return Math.max(0, Math.min(this.meta.levels.length - 1, Math.round(ideal)));
  }

  /** Fetch every tile of the overview level spanning the bounds (idempotent). */
  private requestOverviewTiles(): void {
    if (!this.meta || this.meta.point_count === 0) return;
    const f = this.fit();
    if (!f) return;
    const level = this.overviewLevel(f);
    for (const { tx, ty } of this.overviewTiles(level)) {
      this.tileCache.getTile(level, tx, ty)?.subscribe();
    }
  }

  private overviewTiles(level: number): { tx: number; ty: number }[] {
    if (!this.meta) return [];
    const radius = this.meta.base_radius / Math.pow(2, level);
    const geom = binGeometry(this.meta.bin_shape);
    const tileW = this.meta.tile_span * geom.dx(radius);
    const tileH = this.meta.tile_span * geom.dy(radius);
    const [xmin, ymin, xmax, ymax] = this.meta.bounds;
    const txMin = Math.floor(xmin / tileW) - 1;
    const txMax = Math.ceil(xmax / tileW) + 1;
    const tyMin = Math.floor(ymin / tileH) - 1;
    const tyMax = Math.ceil(ymax / tileH) + 1;
    const tiles: { tx: number; ty: number }[] = [];
    // Guard against a pathological extent producing a huge fan-out.
    if ((txMax - txMin) * (tyMax - tyMin) > 1024) return tiles;
    for (let tx = txMin; tx <= txMax; tx++) {
      for (let ty = tyMin; ty <= tyMax; ty++) tiles.push({ tx, ty });
    }
    return tiles;
  }

  // --- Rendering ------------------------------------------------------------

  private requestRedraw(): void {
    if (this.needsRedraw) return;
    this.needsRedraw = true;
    this.rafId = requestAnimationFrame(() => {
      this.needsRedraw = false;
      this.draw();
    });
  }

  /**
   * Projection→minimap scale (fits the whole extent inside an inset margin)
   * and the data centre, or ``null`` when there's nothing to map.
   */
  private fit(): { scale: number; cx: number; cy: number; margin: number } | null {
    if (!this.meta || this.meta.point_count === 0) return null;
    const [xmin, ymin, xmax, ymax] = this.meta.bounds;
    const dataW = xmax - xmin || 1;
    const dataH = ymax - ymin || 1;
    const margin = 4;
    const scale = Math.min(
      (this.width - margin * 2) / dataW,
      (this.height - margin * 2) / dataH,
    );
    return { scale, cx: (xmin + xmax) / 2, cy: (ymin + ymax) / 2, margin };
  }

  private projToMap(px: number, py: number, f: { scale: number; cx: number; cy: number }): [number, number] {
    return [this.width / 2 + (px - f.cx) * f.scale, this.height / 2 + (py - f.cy) * f.scale];
  }

  private mapToProj(mx: number, my: number, f: { scale: number; cx: number; cy: number }): [number, number] {
    return [f.cx + (mx - this.width / 2) / f.scale, f.cy + (my - this.height / 2) / f.scale];
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fillRect(0, 0, this.width, this.height);

    const f = this.fit();
    if (f) {
      const level = this.overviewLevel(f);
      const cells = this.overviewCells(level);
      let maxCount = 1;
      for (const c of cells) if (c.count > maxCount) maxCount = c.count;
      const geom = binGeometry(this.meta!.bin_shape);
      const cellR = (this.meta!.base_radius / Math.pow(2, level)) * f.scale;
      for (const cell of cells) {
        const [sx, sy] = this.projToMap(cell.cx, cell.cy, f);
        geom.traceCell(ctx, sx, sy, cellR, cell.count === 1);
        const t = Math.log(cell.count) / Math.log(maxCount || 2);
        ctx.fillStyle = densityColor(Math.max(0, Math.min(1, t)));
        ctx.fill();
      }
      this.drawViewportRect(ctx, f);
    }

    // Frame the minimap so it reads as a distinct panel over the canvas.
    ctx.strokeStyle = this.themeColor('--border');
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, this.width - 1, this.height - 1);
  }

  /** Pull all cached cells of the overview *level* covering the extent. */
  private overviewCells(level: number): HexCellPayload[] {
    const out: HexCellPayload[] = [];
    for (const { tx, ty } of this.overviewTiles(level)) {
      const tile = this.tileCache.getCached(level, tx, ty);
      if (tile) out.push(...tile.cells);
    }
    return out;
  }

  private drawViewportRect(
    ctx: CanvasRenderingContext2D,
    f: { scale: number; cx: number; cy: number },
  ): void {
    if (!this.viewportBounds) return;
    const [vxmin, vymin, vxmax, vymax] = this.viewportBounds;
    const [x0, y0] = this.projToMap(vxmin, vymin, f);
    const [x1, y1] = this.projToMap(vxmax, vymax, f);
    const rx = Math.min(x0, x1);
    const ry = Math.min(y0, y1);
    const rw = Math.abs(x1 - x0);
    const rh = Math.abs(y1 - y0);
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();
  }

  private themeColor(varName: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  // --- Click / drag to recenter --------------------------------------------

  onCanvasDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    event.preventDefault();
    this.navigating = true;
    this.recenterFromEvent(event);
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundNavMove);
      document.addEventListener('mouseup', this.boundNavUp);
    });
  }

  private onNavMove(event: MouseEvent): void {
    if (this.navigating) this.recenterFromEvent(event);
  }

  private onNavUp(): void {
    this.navigating = false;
    this.detachNavListeners();
  }

  private recenterFromEvent(event: MouseEvent): void {
    const f = this.fit();
    if (!f) return;
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const [px, py] = this.mapToProj(mx, my, f);
    this.viewport.requestRecenter(px, py);
  }

  private detachNavListeners(): void {
    document.removeEventListener('mousemove', this.boundNavMove);
    document.removeEventListener('mouseup', this.boundNavUp);
  }

  // --- Resize handle --------------------------------------------------------

  onResizeDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    this.resizing = true;
    this.resizeStartX = event.clientX;
    this.resizeStartY = event.clientY;
    this.resizeStartW = this.width;
    this.resizeStartH = this.height;
    document.addEventListener('mousemove', this.boundResizeMove);
    document.addEventListener('mouseup', this.boundResizeUp);
  }

  private onResizeMove(event: MouseEvent): void {
    if (!this.resizing) return;
    // Anchored bottom-right: dragging the top-left handle up/left grows it.
    const dw = this.resizeStartX - event.clientX;
    const dh = this.resizeStartY - event.clientY;
    this.width = Math.round(
      Math.max(MINIMAP_MIN_WIDTH, Math.min(MINIMAP_MAX_WIDTH, this.resizeStartW + dw)),
    );
    this.height = Math.round(
      Math.max(MINIMAP_MIN_HEIGHT, Math.min(MINIMAP_MAX_HEIGHT, this.resizeStartH + dh)),
    );
    this.resizeCanvas();
    this.requestRedraw();
  }

  private onResizeUp(): void {
    if (!this.resizing) return;
    this.resizing = false;
    this.detachResizeListeners();
    this.resized.emit({ width: this.width, height: this.height });
  }

  private detachResizeListeners(): void {
    document.removeEventListener('mousemove', this.boundResizeMove);
    document.removeEventListener('mouseup', this.boundResizeUp);
  }
}
