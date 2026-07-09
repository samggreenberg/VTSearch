import { ChangeDetectionStrategy, Component, effect, ElementRef, HostBinding, inject, Input, input, NgZone, OnChanges, OnDestroy, OnInit, output, SimpleChanges, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { TileCacheService } from '../../services/tile-cache.service';
import { BrowseViewportService, ViewportBounds } from '../../services/browse-viewport.service';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import {
  densityColor,
  resolveColormap,
  rgbString,
  type BrowseColormapId,
  type CanvasTheme,
} from '../browse-canvas/hex-render.util';
import { binGeometry } from '../browse-canvas/bin-geometry';
import { readRootZoom } from '../../utils/root-zoom';
import { onDevicePixelRatioChange } from '../../utils/device-pixel-ratio';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-minimap',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './browse-minimap.component.html',
  styleUrl: './browse-minimap.component.scss',
})
export class BrowseMinimapComponent implements OnInit, OnChanges, OnDestroy {
  private ngZone = inject(NgZone);
  private tileCache = inject(TileCacheService);
  private viewport = inject(BrowseViewportService);
  private selection = inject(BrowseSelectionService);
  private host = inject<ElementRef<HTMLElement>>(ElementRef);

  // Repaint whenever the selection mutates, so the overview's tristate tint
  // (unselected / partial / full) tracks the canvas. Reading the version
  // signal registers this effect as a dependency; the redraw is rAF-coalesced,
  // so a marquee drag that fires many mutations still collapses to one paint
  // per frame. Created in the injection context (field initializer).
  private selectionEffect = effect(() => {
    this.selection.version();
    this.requestRedraw();
  });

  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  readonly meta = input<ProjectionMeta | null>(null);
  @Input() width = 200;
  @Input() height = 150;
  /** Density colormap preset; mirrors the main canvas so the overview matches. */
  readonly colormap = input<BrowseColormapId>('auto');
  /**
   * Docked mode: the minimap fills its container (the browse side panel's
   * meta-row) and sizes its canvas to fit via a {@link ResizeObserver},
   * rather than floating over the canvas at an explicit size. In this mode
   * the close button and corner resize handle are hidden — the panel owns
   * the geometry — but click/drag-to-navigate stays live.
   */
  @Input() @HostBinding('class.dock') dock = false;
  /** Hide request from the close button (floating mode only). */
  readonly closed = output<void>();
  /** Final size after a resize drag, for persistence (floating mode only). */
  readonly resized = output<{
    width: number;
    height: number;
}>();

  private resizeObserver: ResizeObserver | null = null;
  // Teardown for the devicePixelRatio-change listener. A pure density change
  // (monitor-to-monitor drag) leaves the element box untouched, so neither the
  // dock ResizeObserver nor the floating width/height inputs fire; this re-runs
  // resizeCanvas() to rebuild the backing store at the new density.
  private dprListenerTeardown: (() => void) | null = null;

  private ctx!: CanvasRenderingContext2D;
  private dpr = 1;
  private viewportBounds: ViewportBounds = null;

  private tileLoadSub: Subscription | null = null;
  private viewportSub: Subscription | null = null;
  private rafId = 0;
  private needsRedraw = false;
  // Repaints when the document theme flips, matching the main canvas.
  private themeObserver: MutationObserver | null = null;

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

  ngOnInit(): void {
    this.ctx = this.canvasRef.nativeElement.getContext('2d')!;
    if (this.dock) this.startDockSizing();
    this.resizeCanvas();

    // A pure devicePixelRatio change doesn't resize the element, so re-run the
    // canvas sizing (and repaint) when the display density shifts.
    this.ngZone.runOutsideAngular(() => {
      this.dprListenerTeardown = onDevicePixelRatioChange(() => {
        this.resizeCanvas();
        this.requestRedraw();
      });
    });

    // Overview tiles arrive asynchronously; repaint as the cache fills. The
    // viewport box updates on every pan/zoom the main canvas publishes.
    this.tileLoadSub = this.tileCache.tileLoaded$.subscribe(() => this.requestRedraw());
    this.viewportSub = this.viewport.viewport$.subscribe((b) => {
      this.viewportBounds = b;
      this.requestRedraw();
    });

    this.themeObserver = new MutationObserver(() => this.requestRedraw());
    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
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
    if (changes['colormap'] && !changes['colormap'].firstChange) {
      this.requestRedraw();
    }
  }

  ngOnDestroy(): void {
    this.tileLoadSub?.unsubscribe();
    this.viewportSub?.unsubscribe();
    this.themeObserver?.disconnect();
    this.resizeObserver?.disconnect();
    this.dprListenerTeardown?.();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.detachResizeListeners();
    this.detachNavListeners();
  }

  /**
   * Docked mode: track the host element's box and resize the canvas to fill
   * it, so the overview grows/shrinks with the side panel's divider drag. The
   * observer fires outside Angular, so the resize never triggers change
   * detection — only a canvas repaint.
   */
  private startDockSizing(): void {
    const el = this.host.nativeElement;
    this.ngZone.runOutsideAngular(() => {
      this.resizeObserver = new ResizeObserver(() => {
        const w = Math.max(MINIMAP_MIN_WIDTH, Math.round(el.clientWidth));
        const h = Math.max(MINIMAP_MIN_HEIGHT, Math.round(el.clientHeight));
        if (w === this.width && h === this.height) return;
        this.width = w;
        this.height = h;
        this.resizeCanvas();
        // A large size change can shift the overview pyramid level, so make
        // sure the tiles for the new level are requested before repainting.
        this.requestOverviewTiles();
        this.requestRedraw();
      });
      this.resizeObserver.observe(el);
    });
  }

  private resizeCanvas(): void {
    this.dpr = window.devicePixelRatio || 1;
    // this.width/height are layout px, but the app-wide html{zoom} renders
    // the element zoom× larger on screen — bake the zoom into the backing
    // store (as the main canvas effectively does via its visual-px sizing)
    // so the bitmap isn't undersampled/blurry.
    const rootZoom = readRootZoom();
    const scale = this.dpr * rootZoom;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = Math.round(this.width * scale);
    canvas.height = Math.round(this.height * scale);
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
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
    const meta = this.meta();
    if (!meta || meta.levels.length === 0) return 0;
    const basePx = meta.base_radius * f.scale; // level-0 hex radius in px
    const ideal = Math.log2(basePx / BrowseMinimapComponent.OVERVIEW_TARGET_HEX_PX);
    return Math.max(0, Math.min(meta.levels.length - 1, Math.round(ideal)));
  }

  /** Fetch every tile of the overview level spanning the bounds (idempotent). */
  private requestOverviewTiles(): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    const f = this.fit();
    if (!f) return;
    const level = this.overviewLevel(f);
    for (const { tx, ty } of this.overviewTiles(level)) {
      this.tileCache.getTile(level, tx, ty)?.subscribe();
    }
  }

  private overviewTiles(level: number): { tx: number; ty: number }[] {
    const meta = this.meta();
    if (!meta) return [];
    const radius = meta.base_radius / Math.pow(2, level);
    const geom = binGeometry(meta.bin_shape);
    const tileW = meta.tile_span * geom.dx(radius);
    const tileH = meta.tile_span * geom.dy(radius);
    const [xmin, ymin, xmax, ymax] = meta.bounds;
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
   *
   * ``bounds`` is the extent of the bin *centres*, but each edge bin is drawn
   * out to its circumradius beyond its centre, so scaling on the centres alone
   * clips the edge bins off the minimap. We add the overview-level bin radius
   * (in projection units) as margin on every side. That radius depends on the
   * scale we're solving for (the overview level is chosen from it), so iterate
   * from the no-margin fit to a fixed point — the level is quantised, so this
   * settles immediately.
   */
  private fit(): { scale: number; cx: number; cy: number; margin: number } | null {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return null;
    const [xmin, ymin, xmax, ymax] = meta.bounds;
    const dataW = xmax - xmin || 1;
    const dataH = ymax - ymin || 1;
    const margin = 4;
    const availW = this.width - margin * 2;
    const availH = this.height - margin * 2;
    let scale = Math.min(availW / dataW, availH / dataH);
    for (let i = 0; i < 3; i++) {
      const level = this.overviewLevel({ scale });
      const r = meta.base_radius / Math.pow(2, level);
      scale = Math.min(availW / (dataW + 2 * r), availH / (dataH + 2 * r));
    }
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
      const geom = binGeometry(this.meta()!.bin_shape);
      const cellR = (this.meta()!.base_radius / Math.pow(2, level)) * f.scale;
      const cmap = resolveColormap(this.colormap(), this.effectiveTheme());
      const selActive = this.selection.size > 0;
      const selAccent = this.themeColor('--accent-color') || '#4f9dff';
      for (const cell of cells) {
        const [sx, sy] = this.projToMap(cell.cx, cell.cy, f);
        const single = cell.count === 1;
        geom.traceCell(ctx, sx, sy, cellR, single);
        let baseFill: string;
        if (single) {
          baseFill = rgbString(cmap.single);
        } else {
          const t = Math.log(cell.count) / Math.log(maxCount || 2);
          baseFill = densityColor(Math.max(0, Math.min(1, t)), cmap.ramp);
        }
        // Tristate selection tint (unselected / partial / full), the overview
        // analogue of the canvas's none/dashed-ring/solid-ring. At overview bin
        // size a border or glyph would be illegible, so selection is encoded in
        // the fill: full bins go solid accent, partial bins keep their density
        // colour under a translucent accent wash.
        const selState = selActive ? this.selStateFor(cell) : 0;
        if (selState === 2) {
          ctx.fillStyle = selAccent;
          ctx.fill();
        } else if (selState === 1) {
          ctx.fillStyle = baseFill;
          ctx.fill();
          ctx.save();
          ctx.globalAlpha = 0.45;
          ctx.fillStyle = selAccent;
          ctx.fill();
          ctx.restore();
        } else {
          ctx.fillStyle = baseFill;
          ctx.fill();
        }
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

  /**
   * Selection state of an overview cell: 0 = none, 1 = partial, 2 = full,
   * derived from how many of its members are selected — the same rule the
   * canvas uses, so the two views agree. Memoized on the cell against the
   * selection version (in minimap-private fields, distinct from the canvas's)
   * so a pan/zoom/theme repaint doesn't re-scan every overview bin's members
   * when the selection hasn't changed.
   */
  private selStateFor(cell: HexCellPayload): 0 | 1 | 2 {
    const memo = cell as HexCellPayload & { _mmSelVer?: number; _mmSelState?: 0 | 1 | 2 };
    if (memo._mmSelVer === this.selection.version() && memo._mmSelState !== undefined) {
      return memo._mmSelState;
    }
    const members = cell.member_ids && cell.member_ids.length > 0 ? cell.member_ids : [cell.rep_id];
    const sel = this.selection.selectedCountIn(members);
    const state: 0 | 1 | 2 = sel === 0 ? 0 : sel === members.length ? 2 : 1;
    memo._mmSelVer = this.selection.version();
    memo._mmSelState = state;
    return state;
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
    // The default white marker is invisible on the grayscale map, whose ramp
    // runs all the way through near-white — so against that achromatic field
    // use the chromatic accent (and its pre-baked translucent fill), which the
    // theme already adapts. Heat/Ocean never approach white, so white stays the
    // cleanest neutral marker for them.
    const gray = this.colormap() === 'gray';
    ctx.fillStyle = gray ? this.themeColor('--accent-highlight-bg') : 'rgba(255, 255, 255, 0.15)';
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeStyle = gray ? this.themeColor('--accent') : '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();
  }

  private themeColor(varName: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  /** The effective theme in force, read from the document's ``data-theme``. */
  private effectiveTheme(): CanvasTheme {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'light' || t === 'highviz' ? t : 'dark';
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
    // rect / clientX are visual px (CSS layout × the app-wide html{zoom})
    // while this.width/height and the map transform are layout px, so scale
    // the cursor offset by the rendered-size ratio. Without it every click
    // recenters skewed away from the map's top-left by the zoom factor.
    const mx = (event.clientX - rect.left) * (this.width / (rect.width || 1));
    const my = (event.clientY - rect.top) * (this.height / (rect.height || 1));
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
    // clientX/Y deltas are visual px but width/height are layout px, so
    // divide out the app-wide root zoom or the panel grows faster than the
    // cursor moves.
    const rootZoom = readRootZoom();
    const dw = (this.resizeStartX - event.clientX) / rootZoom;
    const dh = (this.resizeStartY - event.clientY) / rootZoom;
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
