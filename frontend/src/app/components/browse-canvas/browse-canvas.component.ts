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
import { ActiveContextService } from '../../services/active-context.service';
import { BrowseViewportService } from '../../services/browse-viewport.service';
import {
  densityColor,
  resolveColormap,
  rgbString,
  type BrowseColormapId,
  type CanvasTheme,
  type ResolvedColormap,
} from './hex-render.util';
import { binGeometry, BinGeometry } from './bin-geometry';
import type {
  HexCellPayload,
  ProjectionMeta,
  TilePayload,
  ViewTransform,
} from '../../models/projection.models';

export interface HexHoverEvent {
  cell: HexCellPayload;
  screenX: number;
  screenY: number;
}

@Component({
  selector: 'vt-browse-canvas',
  standalone: true,
  templateUrl: './browse-canvas.component.html',
  styleUrl: './browse-canvas.component.scss',
})
export class BrowseCanvasComponent implements OnInit, OnChanges, OnDestroy {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() meta: ProjectionMeta | null = null;
  /**
   * Active dataset media type. For ``image`` and ``video`` the representative
   * item's thumbnail is painted directly onto each hex; other types keep the
   * flat density (darkred→yellow) shading.
   */
  @Input() mediaType = '';
  /**
   * User-controlled on-screen size multiplier for hexes. ``1`` is the default
   * fit. This scales rendering (positions + hex radius) only; it deliberately
   * does NOT feed level selection, so growing/shrinking the display never
   * changes the binning, i.e. which vectors land in a given hex.
   */
  @Input() displayScale = 1;
  /**
   * Density colormap preset for the flat (non-thumbnail) shading. ``auto``
   * follows the theme (Ocean in light mode, Heat in dark); the others lock to
   * a specific map. Resolved to concrete colours against the live theme at
   * draw time, so a theme switch repaints with the right ramp.
   */
  @Input() colormap: BrowseColormapId = 'auto';
  @Output() hexHover = new EventEmitter<HexHoverEvent | null>();
  /**
   * The densest visible cell's item count, emitted whenever it changes. Density
   * shading is renormalized to this per frame (yellow = this many items, the
   * darkest red = 1), so the legend reads it to label the ramp with live
   * numbers that track pan/zoom.
   */
  @Output() densityMaxChanged = new EventEmitter<number>();

  /** Loaded representative thumbnails, keyed by media id (insertion-ordered LRU). */
  private thumbCache = new Map<number, HTMLImageElement>();
  /** Media ids whose thumbnail failed to load, so we don't retry every frame. */
  private thumbFailed = new Set<number>();
  private readonly MAX_THUMBS = 2048;

  private ctx!: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  private transform: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private activeLevel = 0;
  private maxCount = 1;
  // Last maxCount pushed out via densityMaxChanged, so the legend is only
  // notified when the top of the scale actually moves (not every frame).
  private lastEmittedMax = 0;
  // The projection id the view was last framed for. Hex and square share one
  // projection id, so toggling bin shape re-bins without re-fitting to data —
  // the pan/zoom is preserved. Only a genuinely new projection re-frames.
  private lastProjectionId = '';

  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private panStartCenterX = 0;
  private panStartCenterY = 0;

  private hoveredCell: HexCellPayload | null = null;
  private hoverDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  // Last known cursor position over the canvas (canvas-relative mx/my plus the
  // viewport clientX/clientY) and whether the pointer is currently inside.
  // Used to re-resolve the hover after a zoom changes which hex sits under a
  // stationary cursor, so the preview/highlight don't go stale.
  private lastMouseX = 0;
  private lastMouseY = 0;
  private lastClientX = 0;
  private lastClientY = 0;
  private pointerInside = false;

  private tileLoadSub: Subscription | null = null;
  private rafId = 0;
  private needsRedraw = false;
  private resizeObserver: ResizeObserver | null = null;
  // Repaints when the document theme flips (explicit switch or an OS
  // dark/light change while on "system"), so the colormap and background
  // track the live theme without the parent having to feed it in.
  private themeObserver: MutationObserver | null = null;

  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);
  // Stable references for the canvas listeners so ngOnDestroy can remove them
  // (inline .bind(this) creates a fresh function each call, which
  // removeEventListener can never match).
  private boundMouseDown = this.onMouseDown.bind(this);
  private boundWheel = this.onWheel.bind(this);
  private boundCanvasMouseMove = this.onCanvasMouseMove.bind(this);
  private boundCanvasMouseLeave = this.onCanvasMouseLeave.bind(this);

  private recenterSub: Subscription | null = null;

  constructor(
    private ngZone: NgZone,
    private tileCache: TileCacheService,
    private activeContext: ActiveContextService,
    private viewport: BrowseViewportService,
  ) {}

  /** True when cells should be painted with the central item's thumbnail. */
  private get thumbnailMode(): boolean {
    return this.mediaType === 'image' || this.mediaType === 'video';
  }

  /** Geometry (hex or square) for the active projection's bin shape. */
  private get geom(): BinGeometry {
    return binGeometry(this.meta?.bin_shape);
  }

  /**
   * On-screen scale: the base zoom (driven by wheel/fit and feeding level
   * selection) times the user's display-size multiplier. All projection↔screen
   * conversions and the rendered hex radius use this; level selection uses the
   * base ``transform.zoom`` alone so the binning is invariant to display size.
   */
  private get effZoom(): number {
    return this.transform.zoom * this.displayScale;
  }

  ngOnInit(): void {
    this.ctx = this.canvasRef.nativeElement.getContext('2d')!;

    this.tileLoadSub = this.tileCache.tileLoaded$.subscribe(() => {
      this.requestRedraw();
    });

    // The minimap publishes recenter requests when the user clicks/drags it;
    // jump the viewport centre there (keeping zoom) and redraw.
    this.recenterSub = this.viewport.recenter$.subscribe(({ x, y }) => {
      this.transform.centerX = x;
      this.transform.centerY = y;
      this.requestRedraw();
    });

    this.resizeObserver = new ResizeObserver(() => {
      this.ngZone.runOutsideAngular(() => this.resize());
    });
    this.resizeObserver.observe(this.canvasRef.nativeElement.parentElement!);

    this.themeObserver = new MutationObserver(() => this.requestRedraw());
    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });

    this.ngZone.runOutsideAngular(() => {
      const el = this.canvasRef.nativeElement;
      el.addEventListener('mousedown', this.boundMouseDown);
      el.addEventListener('wheel', this.boundWheel, { passive: false });
      el.addEventListener('mousemove', this.boundCanvasMouseMove);
      el.addEventListener('mouseleave', this.boundCanvasMouseLeave);
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['meta'] && this.meta) {
      this.tileCache.setProjectionId(this.meta.projection_id);
      // A bin-shape toggle delivers fresh meta for the *same* projection id
      // (hex and square share one UMAP layout). In that case keep the current
      // pan/zoom and just re-bin visually; only a genuinely new projection
      // re-frames to data and drops stale representative thumbnails.
      if (this.meta.projection_id !== this.lastProjectionId) {
        this.lastProjectionId = this.meta.projection_id;
        this.thumbCache.clear();
        this.thumbFailed.clear();
        this.fitToData();
      } else {
        this.updateActiveLevel();
      }
      this.requestRedraw();
    }
    // Display-size changes only rescale rendering; the active level (binning)
    // is left untouched on purpose, so the same hexes are simply drawn larger
    // or smaller.
    if (changes['displayScale'] && !changes['displayScale'].firstChange) {
      this.requestRedraw();
    }
    // A colormap change only affects flat (non-thumbnail) shading; repaint.
    if (changes['colormap'] && !changes['colormap'].firstChange) {
      this.requestRedraw();
    }
  }

  ngOnDestroy(): void {
    this.tileLoadSub?.unsubscribe();
    this.recenterSub?.unsubscribe();
    this.viewport.setViewport(null);
    this.resizeObserver?.disconnect();
    this.themeObserver?.disconnect();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    const el = this.canvasRef.nativeElement;
    el.removeEventListener('mousedown', this.boundMouseDown);
    el.removeEventListener('wheel', this.boundWheel);
    el.removeEventListener('mousemove', this.boundCanvasMouseMove);
    el.removeEventListener('mouseleave', this.boundCanvasMouseLeave);
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    this.thumbCache.clear();
    this.thumbFailed.clear();
  }

  private resize(): void {
    const el = this.canvasRef.nativeElement.parentElement!;
    const rect = el.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.requestRedraw();
  }

  private fitToData(): void {
    if (!this.meta || this.meta.point_count === 0) return;
    const [xmin, ymin, xmax, ymax] = this.meta.bounds;
    const dataW = xmax - xmin || 1;
    const dataH = ymax - ymin || 1;
    const padding = 0.1;
    const padW = dataW * (1 + padding * 2);
    const padH = dataH * (1 + padding * 2);
    const w = this.width || 800;
    const h = this.height || 600;
    // Fit so the *effective* zoom fills the viewport: divide out displayScale so
    // a non-default display size still frames the whole projection on load.
    this.transform.zoom = Math.min(w / padW, h / padH) / this.displayScale;
    this.transform.centerX = (xmin + xmax) / 2;
    this.transform.centerY = (ymin + ymax) / 2;
    this.updateActiveLevel();
  }

  private updateActiveLevel(): void {
    if (!this.meta || this.meta.levels.length === 0) return;
    const targetScreenRadius = 28;
    const idealLevel = Math.log2(
      (this.meta.base_radius * this.transform.zoom) / targetScreenRadius,
    );
    this.activeLevel = Math.max(
      0,
      Math.min(this.meta.levels.length - 1, Math.round(idealLevel)),
    );
  }

  private projToScreen(px: number, py: number): [number, number] {
    const z = this.effZoom;
    const sx = (px - this.transform.centerX) * z + this.width / 2;
    const sy = (py - this.transform.centerY) * z + this.height / 2;
    return [sx, sy];
  }

  private screenToProj(sx: number, sy: number): [number, number] {
    const z = this.effZoom;
    const px = (sx - this.width / 2) / z + this.transform.centerX;
    const py = (sy - this.height / 2) / z + this.transform.centerY;
    return [px, py];
  }

  private getVisibleBounds(): [number, number, number, number] {
    const [xmin, ymin] = this.screenToProj(0, 0);
    const [xmax, ymax] = this.screenToProj(this.width, this.height);
    return [
      Math.min(xmin, xmax),
      Math.min(ymin, ymax),
      Math.max(xmin, xmax),
      Math.max(ymin, ymax),
    ];
  }

  private getVisibleTiles(): { tx: number; ty: number }[] {
    if (!this.meta) return [];
    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const geom = this.geom;
    const tileSpan = this.meta.tile_span;
    const tileW = tileSpan * geom.dx(radius);
    const tileH = tileSpan * geom.dy(radius);

    const [vxmin, vymin, vxmax, vymax] = this.getVisibleBounds();
    const txMin = Math.floor(vxmin / tileW - 1);
    const txMax = Math.ceil(vxmax / tileW + 1);
    const tyMin = Math.floor(vymin / tileH - 1);
    const tyMax = Math.ceil(vymax / tileH + 1);

    const tiles: { tx: number; ty: number }[] = [];
    for (let tx = txMin; tx <= txMax; tx++) {
      for (let ty = tyMin; ty <= tyMax; ty++) {
        tiles.push({ tx, ty });
      }
    }
    return tiles;
  }

  private requestRedraw(): void {
    if (this.needsRedraw) return;
    this.needsRedraw = true;
    this.rafId = requestAnimationFrame(() => {
      this.needsRedraw = false;
      this.draw();
    });
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx || this.width === 0) return;

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fillRect(0, 0, this.width, this.height);

    if (!this.meta || this.meta.point_count === 0) {
      ctx.fillStyle = this.themeColor('--text-muted');
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No projection data', this.width / 2, this.height / 2);
      return;
    }

    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const screenRadius = radius * this.effZoom;

    const visibleTiles = this.getVisibleTiles();
    let allCells: HexCellPayload[] = [];

    for (const { tx, ty } of visibleTiles) {
      const cached = this.tileCache.getCached(level, tx, ty);
      if (cached) {
        allCells = allCells.concat(cached.cells);
      } else {
        this.tileCache.getTile(level, tx, ty)?.subscribe();
      }
    }

    this.maxCount = 1;
    for (const cell of allCells) {
      if (cell.count > this.maxCount) this.maxCount = cell.count;
    }
    if (this.maxCount !== this.lastEmittedMax) {
      this.lastEmittedMax = this.maxCount;
      this.ngZone.run(() => this.densityMaxChanged.emit(this.maxCount));
    }

    // Resolve the colormap against the live theme once per frame, not per cell.
    const cmap = resolveColormap(this.colormap, this.effectiveTheme());

    for (const cell of allCells) {
      const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
      if (sx < -screenRadius * 2 || sx > this.width + screenRadius * 2) continue;
      if (sy < -screenRadius * 2 || sy > this.height + screenRadius * 2) continue;
      this.drawHex(ctx, sx, sy, screenRadius, cell, cmap);
    }

    // Publish the region now on screen so the minimap can draw its viewport box.
    this.viewport.setViewport(this.getVisibleBounds());

    this.prefetchNeighbors(visibleTiles);
  }

  private drawHex(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    cell: HexCellPayload,
    cmap: ResolvedColormap,
  ): void {
    // A cell with one item is drawn as a disc (slightly smaller than the cell);
    // multi-item cells keep their full shape so they tile the space.
    const single = cell.count === 1;
    this.geom.traceCell(ctx, cx, cy, radius, single);

    // Image / video: paint the central item's thumbnail clipped to the cell.
    // Until it loads, fall back to the density shading below so the cell is
    // never blank.
    const thumb = this.thumbnailMode ? this.getThumb(cell.rep_id) : null;
    if (thumb) {
      ctx.save();
      ctx.clip();
      this.drawImageCover(ctx, thumb, cx, cy, radius);
      ctx.restore();
    } else if (single) {
      // Singletons get the colormap's dedicated one-item colour, decoupled
      // from the density ramp so a lone dot reads as "exactly one".
      ctx.fillStyle = rgbString(cmap.single);
      ctx.fill();
    } else {
      const t = Math.log(cell.count) / Math.log(this.maxCount || 2);
      ctx.fillStyle = densityColor(Math.max(0, Math.min(1, t)), cmap.ramp);
      ctx.fill();
    }

    const isHovered =
      this.hoveredCell && this.hoveredCell.q === cell.q && this.hoveredCell.r === cell.r;
    if (isHovered) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      // Thumbnails read better with a faint dark separator than the body-bg
      // hairline used for flat density cells.
      ctx.strokeStyle = thumb ? 'rgba(0, 0, 0, 0.35)' : this.themeColor('--bg-body');
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  /** Cover-fit an image over the hex's 2*radius square (the path must be clipped). */
  private drawImageCover(
    ctx: CanvasRenderingContext2D,
    img: HTMLImageElement,
    cx: number,
    cy: number,
    radius: number,
  ): void {
    const size = radius * 2;
    const scale = Math.max(size / img.naturalWidth, size / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    ctx.drawImage(img, cx - dw / 2, cy - dh / 2, dw, dh);
  }

  /**
   * Return the loaded thumbnail for a representative media id, or null while it
   * loads / if it failed. Kicks off the fetch on first request and redraws when
   * the image arrives.
   */
  private getThumb(representativeId: number): HTMLImageElement | null {
    const cached = this.thumbCache.get(representativeId);
    if (cached) {
      return cached.complete && cached.naturalWidth > 0 ? cached : null;
    }
    if (this.thumbFailed.has(representativeId)) return null;

    if (this.thumbCache.size >= this.MAX_THUMBS) this.evictThumbs();

    const img = new Image();
    img.decoding = 'async';
    img.onload = () => this.requestRedraw();
    img.onerror = () => {
      this.thumbCache.delete(representativeId);
      this.thumbFailed.add(representativeId);
    };
    // The /image route serves the frame for video via its image_response hook.
    img.src = this.activeContext.mediaUrl(`/api/medias/${representativeId}/image`);
    this.thumbCache.set(representativeId, img);
    return null;
  }

  /** Drop the oldest quarter of cached thumbnails (insertion-ordered LRU). */
  private evictThumbs(): void {
    const target = Math.floor(this.MAX_THUMBS * 0.75);
    const toRemove = this.thumbCache.size - target;
    let i = 0;
    for (const key of this.thumbCache.keys()) {
      if (i++ >= toRemove) break;
      this.thumbCache.delete(key);
    }
  }

  private themeColor(varName: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  /** The effective theme in force, read from the document's ``data-theme``. */
  private effectiveTheme(): CanvasTheme {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'light' || t === 'highviz' ? t : 'dark';
  }

  private prefetchNeighbors(visibleTiles: { tx: number; ty: number }[]): void {
    if (!this.meta) return;
    const level = this.activeLevel;
    const seen = new Set(visibleTiles.map((t) => `${t.tx}:${t.ty}`));
    for (const { tx, ty } of visibleTiles) {
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const nx = tx + dx;
        const ny = ty + dy;
        if (!seen.has(`${nx}:${ny}`)) {
          this.tileCache.prefetch(level, nx, ny);
          seen.add(`${nx}:${ny}`);
        }
      }
    }
    if (level > 0) {
      this.prefetchLevel(level - 1, visibleTiles);
    }
    if (this.meta.levels.length > level + 1) {
      this.prefetchLevel(level + 1, visibleTiles);
    }
  }

  private prefetchLevel(targetLevel: number, sourceTiles: { tx: number; ty: number }[]): void {
    if (!this.meta) return;
    const sourceRadius = this.meta.base_radius / Math.pow(2, this.activeLevel);
    const targetRadius = this.meta.base_radius / Math.pow(2, targetLevel);
    const ratio = sourceRadius / targetRadius;
    const seen = new Set<string>();
    for (const { tx, ty } of sourceTiles) {
      const ttx = Math.floor(tx * ratio);
      const tty = Math.floor(ty * ratio);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const key = `${ttx + dx}:${tty + dy}`;
          if (!seen.has(key)) {
            seen.add(key);
            this.tileCache.prefetch(targetLevel, ttx + dx, tty + dy);
          }
        }
      }
    }
  }

  // --- Interaction handlers ---

  private onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    this.isPanning = true;
    this.panStartX = event.clientX;
    this.panStartY = event.clientY;
    this.panStartCenterX = this.transform.centerX;
    this.panStartCenterY = this.transform.centerY;
    document.addEventListener('mousemove', this.boundMouseMove);
    document.addEventListener('mouseup', this.boundMouseUp);
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isPanning) return;
    const dx = event.clientX - this.panStartX;
    const dy = event.clientY - this.panStartY;
    const z = this.effZoom;
    this.transform.centerX = this.panStartCenterX - dx / z;
    this.transform.centerY = this.panStartCenterY - dy / z;
    this.requestRedraw();
  }

  private onMouseUp(): void {
    this.isPanning = false;
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
  }

  /**
   * Zoom the base view by ``factor`` (>1 zooms in, narrowing the span shown),
   * keeping the projection point under screen coords ``(anchorX, anchorY)``
   * fixed. Defaults to the viewport centre, which is what the on-screen +/-
   * buttons use; the wheel passes the cursor position so it zooms toward the
   * pointer. Like the wheel path, this changes the base zoom only — level
   * selection re-runs so the hexes keep their ~28px display size while each
   * covers a narrower span.
   */
  /**
   * Frame the whole projection: pick a zoom and pan so the current data just
   * fits in the viewport (the same framing used on first load), then redraw.
   */
  zoomToFit(): void {
    this.fitToData();
    this.requestRedraw();
    this.refreshHoverAfterZoom();
  }

  zoomBy(factor: number, anchorX = this.width / 2, anchorY = this.height / 2): void {
    const [projX, projY] = this.screenToProj(anchorX, anchorY);
    const newZoom = Math.max(0.01, Math.min(100000, this.transform.zoom * factor));
    // Anchor the point under the cursor using the new *effective* zoom so the
    // display-size multiplier keeps that pixel fixed while zooming.
    const newEffZoom = newZoom * this.displayScale;

    this.transform.centerX = projX - (anchorX - this.width / 2) / newEffZoom;
    this.transform.centerY = projY - (anchorY - this.height / 2) / newEffZoom;
    this.transform.zoom = newZoom;

    this.updateActiveLevel();
    this.requestRedraw();
    this.refreshHoverAfterZoom();
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    this.zoomBy(factor, mx, my);
  }

  private onCanvasMouseMove(event: MouseEvent): void {
    if (this.isPanning) return;
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    this.lastMouseX = mx;
    this.lastMouseY = my;
    this.lastClientX = event.clientX;
    this.lastClientY = event.clientY;
    this.pointerInside = true;

    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    this.hoverDebounceTimer = setTimeout(() => {
      this.emitHoverHit(mx, my, event.clientX, event.clientY);
    }, 30);
  }

  /**
   * Resolve the hex under the canvas-relative point ``(mx, my)`` and emit a
   * hover event when it differs from the currently-hovered cell (clearing it
   * when the point now hits empty space). ``clientX/clientY`` anchor the
   * preview pop-up at the cursor. Shared by the mouse-move handler and the
   * post-zoom refresh.
   */
  private emitHoverHit(mx: number, my: number, clientX: number, clientY: number): void {
    const hit = this.hitTest(mx, my);
    const prevQ = this.hoveredCell?.q;
    const prevR = this.hoveredCell?.r;
    this.hoveredCell = hit;
    if (hit) {
      if (hit.q !== prevQ || hit.r !== prevR) {
        this.ngZone.run(() => {
          this.hexHover.emit({ cell: hit, screenX: clientX, screenY: clientY });
        });
        this.requestRedraw();
      }
    } else if (prevQ != null) {
      this.ngZone.run(() => this.hexHover.emit(null));
      this.requestRedraw();
    }
  }

  /**
   * After a zoom (which can re-bin to a different level), the hex under a
   * stationary cursor changes — re-resolve the hover so the preview and the
   * highlighted hex track the new cell instead of going stale. When the
   * pointer is off the canvas (e.g. the user clicked a +/- button), there is
   * nothing to hover, so clear any lingering preview.
   */
  private refreshHoverAfterZoom(): void {
    if (this.pointerInside) {
      this.emitHoverHit(this.lastMouseX, this.lastMouseY, this.lastClientX, this.lastClientY);
    } else if (this.hoveredCell) {
      this.hoveredCell = null;
      this.ngZone.run(() => this.hexHover.emit(null));
      this.requestRedraw();
    }
  }

  private onCanvasMouseLeave(): void {
    this.pointerInside = false;
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    if (this.hoveredCell) {
      this.hoveredCell = null;
      this.ngZone.run(() => this.hexHover.emit(null));
      this.requestRedraw();
    }
  }

  private hitTest(sx: number, sy: number): HexCellPayload | null {
    if (!this.meta) return null;
    const [px, py] = this.screenToProj(sx, sy);
    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const geom = this.geom;
    const tileW = this.meta.tile_span * geom.dx(radius);
    const tileH = this.meta.tile_span * geom.dy(radius);

    const txEst = Math.floor(px / tileW);
    const tyEst = Math.floor(py / tileH);

    let best: HexCellPayload | null = null;
    let bestDist = Infinity;

    for (let dtx = -1; dtx <= 1; dtx++) {
      for (let dty = -1; dty <= 1; dty++) {
        const tile = this.tileCache.getCached(level, txEst + dtx, tyEst + dty);
        if (!tile) continue;
        for (const cell of tile.cells) {
          const cdx = cell.cx - px;
          const cdy = cell.cy - py;
          const dist = cdx * cdx + cdy * cdy;
          if (dist < bestDist) {
            bestDist = dist;
            best = cell;
          }
        }
      }
    }

    if (best && geom.contains(best.cx - px, best.cy - py, radius)) {
      return best;
    }
    return null;
  }
}
