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
import { BrowseSelectionService } from '../../services/browse-selection.service';
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

/** A right-click on the canvas, carrying what the view needs to open the bin
 *  popup over the spot under the cursor. */
export interface BrowseContextMenuEvent {
  /** Viewport coords (clientX/clientY) the popup anchors to. */
  clientX: number;
  clientY: number;
  /** Member media ids of the bin under the cursor; empty over blank space. */
  members: number[];
}

/** How much larger the hovered cell is drawn relative to its neighbours so it
 *  lifts off the grid. The border is reserved for selection state, so hover is
 *  signalled by this size bump + a soft drop shadow instead of a ring. */
const HOVER_RADIUS_SCALE = 1.18;

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
  /** On-screen bin radius (CSS px) the "M" thumbnail size targets, and the
   * default before any saved size is applied. See {@link targetRadius}. */
  static readonly DEFAULT_TARGET_RADIUS = 28;
  /**
   * Target on-screen bin radius in CSS px: the size each bin/thumbnail aims to
   * render at. Level selection picks the pyramid level whose bins land closest
   * to this, so it is the "thumbnail size" knob — bigger value ⇒ bigger, coarser
   * bins. It is set, not bound: {@link setThumbnailRadius} updates it (and, for
   * the +/- buttons, scales the view in lock-step so the *same* bins simply use
   * more pixels rather than re-binning). Default 28 = the "M" thumbnail size.
   */
  private targetRadius = BrowseCanvasComponent.DEFAULT_TARGET_RADIUS;
  /**
   * Density colormap preset for the flat (non-thumbnail) shading. ``auto``
   * follows the theme (Ocean in light mode, Heat in dark); the others lock to
   * a specific map. Resolved to concrete colours against the live theme at
   * draw time, so a theme switch repaints with the right ramp.
   */
  @Input() colormap: BrowseColormapId = 'auto';
  /**
   * Width (CSS px) of the colormap-coloured border painted around multi-item
   * ("pile") thumbnails. The band's colour is the density colour for the cell's
   * item count, so it reads as how tall the stack under the tile is. ``0``
   * disables it (cells fall back to the faint dark separator). Only takes effect
   * in {@link thumbnailMode} (image/video); singletons never get the band.
   */
  @Input() thumbnailBorder = 0;
  /**
   * GUI parallel to the Shift modifier: when on, a plain left-drag rubber-bands
   * a selection marquee instead of panning, so the region-select gesture is
   * discoverable without knowing the Shift+drag hotkey. Shift+drag keeps working
   * regardless. Toggled by the region-select button in the browse toolbar.
   */
  @Input() marqueeMode = false;
  @Output() hexHover = new EventEmitter<HexHoverEvent | null>();
  /** A right-click on the canvas; the view opens the bin popup in response. */
  @Output() contextMenu = new EventEmitter<BrowseContextMenuEvent>();
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
  // The transform the pixels currently on the canvas were painted at. After a
  // normal draw() this mirrors `transform`; during a zoom transition it tracks
  // the interpolated frame so a re-triggered transition can chain from what's
  // actually on screen. Seeds the "from" end of the zoom-in/out animation.
  private displayedTransform: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  // Set once the first real frame has been painted. The zoom transition needs a
  // prior frame to snapshot, so it stays disabled until this is true.
  private hasDrawn = false;
  private activeLevel = 0;
  private maxCount = 1;
  // Last maxCount pushed out via densityMaxChanged, so the legend is only
  // notified when the top of the scale actually moves (not every frame).
  private lastEmittedMax = 0;
  // The projection id the view was last framed for. Hex and square share one
  // projection id, so toggling bin shape re-bins without re-fitting to data —
  // the pan/zoom is preserved. Only a genuinely new projection re-frames.
  private lastProjectionId = '';
  // Whether the current framing was fit against the canvas's real measured size.
  // `meta` can arrive (and trigger the initial `fitToData`) before the canvas
  // has laid out, in which case the fit runs against the 800x600 fallback and
  // the published viewport bounds are wrong. We refit once on the first real
  // `resize()`; this flag stops later window resizes from clobbering the user's
  // pan/zoom.
  private fittedAgainstRealSize = false;

  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private panStartCenterX = 0;
  private panStartCenterY = 0;
  // Whether the current drag has moved past the click threshold. A mousedown +
  // mouseup with no real movement is treated as a click (toggle the bin under
  // the cursor) rather than a pan, so plain click selects without fighting pan.
  private dragMoved = false;
  private static readonly CLICK_MOVE_THRESHOLD = 4;

  // A single click toggles the bin under the cursor, but a double-click zooms in
  // there — so the toggle is deferred by the double-click window and dropped if a
  // second click lands. Without the defer, every double-click would also flip the
  // bin's selection on its way to zooming.
  private clickTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingToggleX = 0;
  private pendingToggleY = 0;
  private static readonly DBLCLICK_MS = 250;
  // How hard a double-click zooms in about the cursor. Larger than the wheel's
  // 1.15/tick so the gesture lands a decisive jump, matching the map idiom.
  private static readonly DOUBLE_CLICK_ZOOM = 2.0;

  // Shift+drag draws a marquee rectangle (canvas-relative screen coords) that
  // adds every bin whose centre falls inside it to the selection — the fast path
  // for grabbing a region, since plain drag is reserved for panning.
  private isMarquee = false;
  private marquee: { x0: number; y0: number; x1: number; y1: number } | null = null;

  // Accent colour resolved from the live theme once per frame, used for the
  // selection rings and the marquee rectangle.
  private selAccent = '#4f9dff';

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

  // --- Zoom transition (picture-in-picture) ---------------------------------
  // When a zoom crosses a pyramid-level boundary the bins re-lay-out, which used
  // to snap with no sense of "you zoomed *this* canvas". Instead we freeze the
  // current frame to an offscreen snapshot and, for ~ZOOM_ANIM_MS, blit it
  // scaled+translated so it grows (zoom-in) or shrinks (zoom-out) from where it
  // was to where the same region now sits — then paint the real, rebinned frame.
  // Because projection→screen is affine, the snapshot only needs a uniform
  // scale + offset per frame (see {@link zoomBlitRect}), no per-bin work.
  private static readonly ZOOM_ANIM_MS = 220;
  private animActive = false;
  // Offscreen copy of the canvas backing store taken when a transition starts;
  // reused across transitions to avoid reallocating.
  private animSnapshot: HTMLCanvasElement | null = null;
  private animFrom: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private animTo: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private animStartTs = 0;
  private animRafId = 0;
  // Background colour captured at transition start, so each frame doesn't pay a
  // getComputedStyle just to clear behind the shrinking/growing snapshot.
  private animBg = '';
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
  private boundDblClick = this.onDblClick.bind(this);
  private boundContextMenu = this.onContextMenu.bind(this);

  private recenterSub: Subscription | null = null;
  private selectionSub: Subscription | null = null;

  constructor(
    private ngZone: NgZone,
    private tileCache: TileCacheService,
    private activeContext: ActiveContextService,
    private viewport: BrowseViewportService,
    private selection: BrowseSelectionService,
  ) {}

  /** True when cells should be painted with the central item's thumbnail. */
  private get thumbnailMode(): boolean {
    return this.mediaType === 'image' || this.mediaType === 'video';
  }

  /** Geometry (hex or square) for the active projection's bin shape. */
  private get geom(): BinGeometry {
    return binGeometry(this.meta?.bin_shape);
  }

  /** On-screen scale (projection units → CSS px). Used for all projection↔screen
   * conversions and the rendered bin radius. Thumbnail size no longer folds in
   * here: it lives in {@link targetRadius} (level selection) and, for the +/-
   * buttons, in a matching {@link transform}.zoom change, so "Zoom" and
   * "thumbnail size" are now distinct operations rather than the same multiply. */
  private get effZoom(): number {
    return this.transform.zoom;
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

    // Repaint when the selection changes so the per-cell selection rings track
    // the live set. Unselected cells are left untouched — a selection elsewhere
    // never dims or otherwise alters them.
    this.selectionSub = this.selection.changed$.subscribe(() => this.requestRedraw());

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
      el.addEventListener('dblclick', this.boundDblClick);
      el.addEventListener('contextmenu', this.boundContextMenu);
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['meta'] && this.meta) {
      // Fresh meta re-bins (new projection, bin-shape toggle, cull). Any zoom
      // transition was easing the *old* data, so abandon it; the redraw below
      // paints the new state.
      this.cancelZoomAnim();
      this.tileCache.setProjectionId(this.meta.projection_id);
      // A bin-shape toggle delivers fresh meta for the *same* projection id
      // (hex and square share one UMAP layout). In that case keep the current
      // pan/zoom and just re-bin visually; only a genuinely new projection
      // re-frames to data and drops stale representative thumbnails.
      if (this.meta.projection_id !== this.lastProjectionId) {
        this.lastProjectionId = this.meta.projection_id;
        this.thumbCache.clear();
        this.thumbFailed.clear();
        // A new projection (media-type switch / rebuild) re-lays-out every item,
        // so the old selection no longer maps to what's on screen — drop it. A
        // bin-shape toggle keeps the same projection id and selection, since the
        // ids are shape-independent. A Re-project of the items already on screen
        // arms the survive mark instead: the ids are unchanged (only positions
        // move), so the id-based selection stays coherent and is kept.
        if (!this.selection.consumeSurviveProjectionChange()) {
          this.selection.clear();
        }
        this.fitToData();
      } else if (this.viewport.consumeFitOnNextMeta()) {
        // Same projection id, but the view asked for a re-frame (e.g. a
        // Remove-from-Good cull shrank the bounds): fit to what remains.
        this.fitToData();
      } else {
        this.updateActiveLevel();
      }
      this.requestRedraw();
    }
    // Entering region-select mode: drop any hover preview/highlight that was
    // showing, since hover is suppressed while the mode is on.
    if (changes['marqueeMode'] && this.marqueeMode) {
      this.clearHover();
    }
    // A colormap change only affects flat (non-thumbnail) shading; repaint.
    if (changes['colormap'] && !changes['colormap'].firstChange) {
      this.requestRedraw();
    }
    // The pile-thumbnail border width only changes how thumbnail cells are
    // stroked; a repaint picks it up without re-binning or re-fetching tiles.
    if (changes['thumbnailBorder'] && !changes['thumbnailBorder'].firstChange) {
      this.requestRedraw();
    }
  }

  ngOnDestroy(): void {
    this.tileLoadSub?.unsubscribe();
    this.recenterSub?.unsubscribe();
    this.selectionSub?.unsubscribe();
    this.viewport.setViewport(null);
    this.resizeObserver?.disconnect();
    this.themeObserver?.disconnect();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.animRafId) cancelAnimationFrame(this.animRafId);
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    if (this.clickTimer) clearTimeout(this.clickTimer);
    const el = this.canvasRef.nativeElement;
    el.removeEventListener('mousedown', this.boundMouseDown);
    el.removeEventListener('wheel', this.boundWheel);
    el.removeEventListener('mousemove', this.boundCanvasMouseMove);
    el.removeEventListener('mouseleave', this.boundCanvasMouseLeave);
    el.removeEventListener('dblclick', this.boundDblClick);
    el.removeEventListener('contextmenu', this.boundContextMenu);
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    this.thumbCache.clear();
    this.thumbFailed.clear();
  }

  private resize(): void {
    // A resize changes the backing store and framing; a snapshot blit sized to
    // the old canvas would be wrong, so drop any in-flight transition.
    this.cancelZoomAnim();
    const el = this.canvasRef.nativeElement.parentElement!;
    const rect = el.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    // getBoundingClientRect() returns viewport-coordinate px (CSS layout × root zoom).
    // canvas.style.width/height must be in CSS px, so divide out the root zoom to
    // avoid the canvas being visually double-scaled (once by html{zoom:N}, once by
    // the explicit style override), which would shift hit-test coordinates off the
    // cursor by the zoom factor.
    const rootZoom = parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    canvas.style.width = `${this.width / rootZoom}px`;
    canvas.style.height = `${this.height / rootZoom}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // The initial fit may have run against the 800x600 fallback (meta arrived
    // before layout). Now that the real size is known, refit to it so the
    // framing and the viewport bounds published to the minimap match the actual
    // canvas. Only on the first real measurement — a later window resize keeps
    // the user's pan/zoom.
    if (!this.fittedAgainstRealSize && this.meta && this.width > 0 && this.height > 0) {
      this.fitToData();
    }
    this.requestRedraw();
  }

  private fitToData(): void {
    if (!this.meta || this.meta.point_count === 0) return;
    const [xmin, ymin, xmax, ymax] = this.meta.bounds;
    const dataW = xmax - xmin || 1;
    const dataH = ymax - ymin || 1;
    // Small breathing room beyond the bins themselves.
    const padding = 0.05;
    const w = this.width || 800;
    const h = this.height || 600;
    // `bounds` is the extent of the bin *centres*, but each edge bin is drawn out
    // to its circumradius beyond its centre, so framing on the centres alone clips
    // the edge bins. Add the bin circumradius (in projection units) as margin. The
    // active level — and therefore the radius — depends on the zoom we're solving
    // for, so iterate a few times from the no-margin fit to a fixed point (the
    // level is quantised and clamps at 0, so this settles immediately).
    let zoom = Math.min(
      w / (dataW * (1 + padding * 2)),
      h / (dataH * (1 + padding * 2)),
    );
    for (let i = 0; i < 3; i++) {
      const level = this.levelForEffZoom(zoom);
      const r = this.meta.base_radius / Math.pow(2, level);
      const padW = dataW + 2 * (r + dataW * padding);
      const padH = dataH + 2 * (r + dataH * padding);
      zoom = Math.min(w / padW, h / padH);
    }
    this.transform.zoom = zoom;
    this.transform.centerX = (xmin + xmax) / 2;
    this.transform.centerY = (ymin + ymax) / 2;
    // Mark whether this fit used the real canvas size (vs the 800x600 fallback),
    // so `resize()` knows whether a corrective refit is still owed.
    this.fittedAgainstRealSize = this.width > 0 && this.height > 0;
    this.updateActiveLevel();
  }

  /** Pyramid level whose bins render closest to the current thumbnail size
   * ({@link targetRadius}) at the given on-screen zoom. Shared by live level
   * selection and fit framing. */
  private levelForEffZoom(effZoom: number): number {
    if (!this.meta || this.meta.levels.length === 0) return 0;
    const idealLevel = Math.log2(
      (this.meta.base_radius * effZoom) / this.targetRadius,
    );
    return Math.max(
      0,
      Math.min(this.meta.levels.length - 1, Math.round(idealLevel)),
    );
  }

  private updateActiveLevel(): void {
    if (!this.meta || this.meta.levels.length === 0) return;
    this.activeLevel = this.levelForEffZoom(this.effZoom);
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
    // A zoom transition owns the canvas while it runs; tile loads, hover and
    // selection repaints that arrive mid-transition are folded into the real
    // frame the transition paints when it lands (see {@link endZoomAnim}).
    if (this.animActive) return;
    if (this.needsRedraw) return;
    this.needsRedraw = true;
    this.rafId = requestAnimationFrame(() => {
      this.needsRedraw = false;
      this.draw();
    });
  }

  /** Honour the OS "reduce motion" setting: skip the zoom transition entirely. */
  private prefersReducedMotion(): boolean {
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  }

  /**
   * Commit a zoom change after `transform` and `activeLevel` have been updated.
   * Plays the picture-in-picture transition only when the pyramid level actually
   * flipped (the moment bins re-lay-out) — within a level the bins just rescale,
   * which already reads as a smooth zoom, so a plain redraw is enough. A zoom
   * that lands mid-transition without crossing a level retargets the in-flight
   * animation so it eases on to the latest view instead of snapping.
   *
   * @param prevLevel the active level *before* this zoom, for the cross check.
   */
  private commitZoomChange(prevLevel: number): void {
    const crossedLevel = this.activeLevel !== prevLevel;
    if (crossedLevel && this.hasDrawn && this.width > 0 && !this.prefersReducedMotion()) {
      this.startZoomAnim();
    } else if (this.animActive) {
      // Same level but the view moved while a transition runs: ease on to the
      // new target rather than stopping short at the old one.
      this.animTo = { ...this.transform };
    } else {
      this.requestRedraw();
    }
  }

  /**
   * Freeze the current frame and begin easing it toward the new transform. The
   * snapshot is the pixels on screen right now — the pre-zoom frame, or the
   * current blit frame when chaining off an in-flight transition (`animFrom`
   * tracks the live interpolated transform, so the hand-off is seamless).
   */
  private startZoomAnim(): void {
    const canvasEl = this.canvasRef.nativeElement;
    let snap = this.animSnapshot;
    if (!snap) snap = document.createElement('canvas');
    if (snap.width !== canvasEl.width || snap.height !== canvasEl.height) {
      snap.width = canvasEl.width;
      snap.height = canvasEl.height;
    }
    const sctx = snap.getContext('2d')!;
    sctx.clearRect(0, 0, snap.width, snap.height);
    sctx.drawImage(canvasEl, 0, 0);
    this.animSnapshot = snap;

    this.animFrom = { ...this.displayedTransform };
    this.animTo = { ...this.transform };
    this.animStartTs = performance.now();
    this.animBg = this.themeColor('--bg-body');

    // The animation owns the canvas now; drop any pending plain redraw.
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.needsRedraw = false;
    if (this.animRafId) cancelAnimationFrame(this.animRafId);
    this.animActive = true;
    this.ngZone.runOutsideAngular(() => {
      this.animRafId = requestAnimationFrame(this.stepZoomAnim);
    });
  }

  /**
   * Where to blit the frozen snapshot so the proj region it covers lands where
   * transform `to` (interpolated by eased fraction `e` from `animFrom`) would
   * put it. proj→screen is affine and the zoom is uniform, so the snapshot maps
   * by a single scale + offset: a point at snapshot-pixel `s` goes to
   * `scale*s + offset`. At e=0 this is the identity (overlays the live frame);
   * at e=1 it matches the destination transform exactly, so the real rebinned
   * frame can take over without a jump. Returns CSS-px destination rect.
   */
  private zoomBlitRect(e: number): { x: number; y: number; w: number; h: number } {
    const from = this.animFrom;
    const z0 = from.zoom;
    // Geometric zoom interpolation (perceptually even), linear centre pan.
    const zu = z0 * Math.pow(this.animTo.zoom / z0, e);
    const cux = from.centerX + (this.animTo.centerX - from.centerX) * e;
    const cuy = from.centerY + (this.animTo.centerY - from.centerY) * e;
    const scale = zu / z0;
    const x = (this.width / 2) * (1 - scale) + (from.centerX - cux) * zu;
    const y = (this.height / 2) * (1 - scale) + (from.centerY - cuy) * zu;
    return { x, y, w: scale * this.width, h: scale * this.height };
  }

  /** One frame of the zoom transition: clear, blit the scaled snapshot, repeat
   *  until the duration elapses, then paint the real rebinned frame. */
  private readonly stepZoomAnim = (now: number): void => {
    const ctx = this.ctx;
    const snap = this.animSnapshot;
    if (!this.animActive || !snap || !ctx) return;

    const t = Math.min(1, Math.max(0, (now - this.animStartTs) / BrowseCanvasComponent.ZOOM_ANIM_MS));
    const e = 1 - Math.pow(1 - t, 3); // easeOutCubic: quick out, settle into the rebin
    const rect = this.zoomBlitRect(e);

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.animBg;
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(snap, 0, 0, snap.width, snap.height, rect.x, rect.y, rect.w, rect.h);

    // Track what the canvas shows so a re-trigger chains from this exact frame.
    this.displayedTransform = {
      centerX: this.animFrom.centerX + (this.animTo.centerX - this.animFrom.centerX) * e,
      centerY: this.animFrom.centerY + (this.animTo.centerY - this.animFrom.centerY) * e,
      zoom: this.animFrom.zoom * Math.pow(this.animTo.zoom / this.animFrom.zoom, e),
    };

    if (t < 1) {
      this.animRafId = requestAnimationFrame(this.stepZoomAnim);
    } else {
      this.endZoomAnim();
    }
  };

  /** Land the transition: paint the real, rebinned frame at the destination. */
  private endZoomAnim(): void {
    this.animActive = false;
    if (this.animRafId) {
      cancelAnimationFrame(this.animRafId);
      this.animRafId = 0;
    }
    this.draw();
  }

  /** Abandon any in-flight transition without painting (the caller repaints).
   *  Used when the projection/bin-shape/size changes out from under it. */
  private cancelZoomAnim(): void {
    if (!this.animActive) return;
    this.animActive = false;
    if (this.animRafId) {
      cancelAnimationFrame(this.animRafId);
      this.animRafId = 0;
    }
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
    // Accent for selection rings + marquee, also resolved once per frame.
    this.selAccent = this.themeColor('--accent-color') || '#4f9dff';
    const selectionActive = this.selection.size > 0;

    // The hovered cell is deferred and redrawn last (enlarged, on top of its
    // neighbours) so the hover read-out is a size bump rather than a border —
    // leaving the border free to encode selection state.
    let hovered: { cell: HexCellPayload; sx: number; sy: number } | null = null;
    for (const cell of allCells) {
      const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
      if (sx < -screenRadius * 2 || sx > this.width + screenRadius * 2) continue;
      if (sy < -screenRadius * 2 || sy > this.height + screenRadius * 2) continue;
      if (
        this.hoveredCell &&
        this.hoveredCell.q === cell.q &&
        this.hoveredCell.r === cell.r
      ) {
        hovered = { cell, sx, sy };
        continue;
      }
      this.drawHex(ctx, sx, sy, screenRadius, cell, cmap, selectionActive);
    }
    if (hovered) {
      this.drawHoveredHex(
        ctx,
        hovered.sx,
        hovered.sy,
        screenRadius,
        hovered.cell,
        cmap,
        selectionActive,
      );
    }

    if (this.marquee) this.drawMarquee(ctx);

    // Publish the region now on screen so the minimap can draw its viewport box.
    this.viewport.setViewport(this.getVisibleBounds());

    this.prefetchNeighbors(visibleTiles);

    // Record what's now on screen so a zoom transition can grow/shrink this
    // exact frame the next time a zoom crosses a level boundary.
    this.displayedTransform = {
      centerX: this.transform.centerX,
      centerY: this.transform.centerY,
      zoom: this.transform.zoom,
    };
    this.hasDrawn = true;
  }

  /** Selection state of a cell: 0 = none, 1 = partial, 2 = full. Memoized on
   *  the cell against the selection version so a steady-state pan doesn't
   *  re-scan every bin's members each frame. */
  private selStateFor(cell: HexCellPayload): 0 | 1 | 2 {
    const memo = cell as HexCellPayload & { _selVer?: number; _selState?: 0 | 1 | 2 };
    if (memo._selVer === this.selection.version && memo._selState !== undefined) {
      return memo._selState;
    }
    const members = this.cellMembers(cell);
    const sel = this.selection.selectedCountIn(members);
    const state: 0 | 1 | 2 = sel === 0 ? 0 : sel === members.length ? 2 : 1;
    memo._selVer = this.selection.version;
    memo._selState = state;
    return state;
  }

  /** Translucent fill + dashed accent border for the in-progress marquee. */
  private drawMarquee(ctx: CanvasRenderingContext2D): void {
    const m = this.marquee!;
    const x = Math.min(m.x0, m.x1);
    const y = Math.min(m.y0, m.y1);
    const w = Math.abs(m.x1 - m.x0);
    const h = Math.abs(m.y1 - m.y0);
    ctx.save();
    ctx.fillStyle = this.selAccent;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = this.selAccent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  private drawHex(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    cell: HexCellPayload,
    cmap: ResolvedColormap,
    selectionActive: boolean,
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

    // The border encodes selection state only: an inset accent ring, solid when
    // every member is selected and dashed when only some are. Unselected bins
    // keep their plain border regardless of what's selected elsewhere, so the
    // grid never re-shades when the selection changes.
    const selState = selectionActive ? this.selStateFor(cell) : 0;
    if (selState > 0) {
      // Selected bin: an inset accent ring (solid when every member is
      // selected, dashed when only some are — the "partial" state). Clipped so
      // the band sits just inside the cell rather than bleeding onto neighbours.
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = this.selAccent;
      ctx.lineWidth = 5;
      if (selState === 1) ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.restore();
    } else if (thumb && !single && this.thumbnailBorder > 0) {
      // Pile thumbnail: a band whose colormap colour encodes how many items are
      // stacked under this tile. Clipped to the cell so the full width sits just
      // inside the thumbnail edge rather than bleeding onto neighbours (a
      // centred stroke would spill half its width outward).
      const t = Math.log(cell.count) / Math.log(this.maxCount || 2);
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = densityColor(Math.max(0, Math.min(1, t)), cmap.ramp);
      ctx.lineWidth = this.thumbnailBorder * 2;
      ctx.stroke();
      ctx.restore();
    } else {
      // Thumbnails read better with a faint dark separator than the body-bg
      // hairline used for flat density cells.
      ctx.strokeStyle = thumb ? 'rgba(0, 0, 0, 0.35)' : this.themeColor('--bg-body');
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  /** Redraw the hovered cell on top of its neighbours, enlarged and with a soft
   *  drop shadow so it lifts off the grid. Hover is signalled this way (not by a
   *  border) so the cell's border can stay dedicated to selection state. */
  private drawHoveredHex(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    cell: HexCellPayload,
    cmap: ResolvedColormap,
    selectionActive: boolean,
  ): void {
    const bumped = radius * HOVER_RADIUS_SCALE;
    // Cast a single clean drop shadow from an opaque base shape first, then
    // paint the real (shadow-free) cell on top so the fill/border don't each
    // stack their own shadow.
    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = Math.max(4, radius * 0.3);
    ctx.shadowOffsetY = Math.max(1, radius * 0.1);
    this.geom.traceCell(ctx, cx, cy, bumped, cell.count === 1);
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fill();
    ctx.restore();

    this.drawHex(ctx, cx, cy, bumped, cell, cmap, selectionActive);
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
    // Downscaled tile, not full-res /image: a browse projection can hold
    // thousands of points, so painting full-size bitmaps onto the hexes would
    // exhaust memory. The /thumbnail route serves the frame for video via the
    // same image_response hook, then downscales it.
    img.src = this.activeContext.mediaUrl(`/api/medias/${representativeId}/thumbnail`);
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
    // A new drag/marquee takes over from any zoom transition: settle it to the
    // real frame now so the pan/marquee starts from a correct, crisp view.
    if (this.animActive) this.endZoomAnim();
    // A fresh press settles any single-click toggle still waiting out the
    // double-click window (so quick clicks on different bins each register). The
    // second press of a double-click (detail >= 2) is exempt: flushing there
    // would commit the very toggle the double-click means to drop.
    if (event.detail < 2) this.flushPendingToggle();
    this.panStartX = event.clientX;
    this.panStartY = event.clientY;
    this.dragMoved = false;

    if (event.shiftKey || this.marqueeMode) {
      // Shift+drag (or the region-select toggle): rubber-band a region to add to
      // the selection. Suppress any hover preview while marqueeing so it doesn't
      // flicker over the rectangle.
      event.preventDefault();
      const [mx, my] = this.canvasXY(event);
      this.isMarquee = true;
      this.marquee = { x0: mx, y0: my, x1: mx, y1: my };
      document.addEventListener('mousemove', this.boundMouseMove);
      document.addEventListener('mouseup', this.boundMouseUp);
      return;
    }

    this.isPanning = true;
    this.panStartCenterX = this.transform.centerX;
    this.panStartCenterY = this.transform.centerY;
    document.addEventListener('mousemove', this.boundMouseMove);
    document.addEventListener('mouseup', this.boundMouseUp);
  }

  private onMouseMove(event: MouseEvent): void {
    if (this.isMarquee && this.marquee) {
      const [mx, my] = this.canvasXY(event);
      this.marquee.x1 = mx;
      this.marquee.y1 = my;
      this.dragMoved = true;
      this.requestRedraw();
      return;
    }
    if (!this.isPanning) return;
    const dx = event.clientX - this.panStartX;
    const dy = event.clientY - this.panStartY;
    if (
      Math.abs(dx) > BrowseCanvasComponent.CLICK_MOVE_THRESHOLD ||
      Math.abs(dy) > BrowseCanvasComponent.CLICK_MOVE_THRESHOLD
    ) {
      this.dragMoved = true;
    }
    const z = this.effZoom;
    this.transform.centerX = this.panStartCenterX - dx / z;
    this.transform.centerY = this.panStartCenterY - dy / z;
    this.requestRedraw();
  }

  private onMouseUp(event: MouseEvent): void {
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);

    if (this.isMarquee) {
      this.isMarquee = false;
      this.commitMarquee();
      this.marquee = null;
      this.requestRedraw();
      return;
    }

    const wasPanning = this.isPanning;
    this.isPanning = false;
    // A press that never crossed the move threshold is a click: toggle the bin
    // under the cursor (no modifier — Shift is reserved for the marquee). The
    // toggle is deferred so a double-click (which zooms) doesn't also select.
    if (wasPanning && !this.dragMoved && !event.shiftKey) {
      if (event.detail >= 2) {
        // Second release of a double-click: the dblclick handler zooms, so drop
        // the pending single-click toggle rather than flipping the bin.
        this.cancelPendingToggle();
      } else {
        const [mx, my] = this.canvasXY(event);
        this.scheduleToggle(mx, my);
      }
    }
  }

  /**
   * Schedule a single-click bin toggle, deferred by the double-click window so a
   * double-click (which zooms in) doesn't also flip the bin's selection. Any
   * pending toggle is committed first, so two quick clicks on *different* bins
   * each register rather than the second cancelling the first.
   */
  private scheduleToggle(mx: number, my: number): void {
    this.flushPendingToggle();
    this.pendingToggleX = mx;
    this.pendingToggleY = my;
    this.clickTimer = setTimeout(() => {
      this.clickTimer = null;
      this.toggleCellAt(mx, my);
    }, BrowseCanvasComponent.DBLCLICK_MS);
  }

  /** Run a pending single-click toggle now (if any) and clear the timer. */
  private flushPendingToggle(): void {
    if (this.clickTimer === null) return;
    clearTimeout(this.clickTimer);
    this.clickTimer = null;
    this.toggleCellAt(this.pendingToggleX, this.pendingToggleY);
  }

  /** Drop a pending single-click toggle without running it (double-click path). */
  private cancelPendingToggle(): void {
    if (this.clickTimer === null) return;
    clearTimeout(this.clickTimer);
    this.clickTimer = null;
  }

  /** Double-click zooms in about the cursor (map idiom). Shift/region-select are
   *  reserved for the marquee, so they don't zoom. */
  private onDblClick(event: MouseEvent): void {
    if (event.shiftKey || this.marqueeMode) return;
    this.cancelPendingToggle();
    const [mx, my] = this.canvasXY(event);
    this.zoomBy(BrowseCanvasComponent.DOUBLE_CLICK_ZOOM, mx, my);
  }

  /** Right-click: suppress the native menu and ask the view to open the bin
   *  popup, carrying the cursor anchor and the bin (if any) under it. */
  private onContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const [mx, my] = this.canvasXY(event);
    // Close any hover preview so it doesn't sit under the popup.
    this.clearHover();
    const cell = this.hitTest(mx, my);
    const members = cell ? this.cellMembers(cell) : [];
    this.ngZone.run(() =>
      this.contextMenu.emit({
        clientX: event.clientX,
        clientY: event.clientY,
        members,
      }),
    );
  }

  /** Canvas-relative ``[x, y]`` for a mouse event. */
  private canvasXY(event: MouseEvent): [number, number] {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  /** Member media ids of a cell, falling back to its representative. */
  private cellMembers(cell: HexCellPayload): number[] {
    return cell.member_ids && cell.member_ids.length > 0 ? cell.member_ids : [cell.rep_id];
  }

  /** Toggle the selection of the bin under canvas point ``(mx, my)``. */
  private toggleCellAt(mx: number, my: number): void {
    const cell = this.hitTest(mx, my);
    if (!cell) return;
    const members = this.cellMembers(cell);
    // Mutate inside the zone so the selection panel's count updates.
    this.ngZone.run(() => this.selection.toggleBin(members));
    this.requestRedraw();
  }

  /** Add every bin whose centre falls inside the marquee rectangle. */
  private commitMarquee(): void {
    if (!this.marquee || !this.meta) return;
    const [px0, py0] = this.screenToProj(this.marquee.x0, this.marquee.y0);
    const [px1, py1] = this.screenToProj(this.marquee.x1, this.marquee.y1);
    const minX = Math.min(px0, px1);
    const maxX = Math.max(px0, px1);
    const minY = Math.min(py0, py1);
    const maxY = Math.max(py0, py1);

    const level = this.activeLevel;
    const ids: number[] = [];
    for (const { tx, ty } of this.getVisibleTiles()) {
      const tile = this.tileCache.getCached(level, tx, ty);
      if (!tile) continue;
      for (const cell of tile.cells) {
        if (cell.cx >= minX && cell.cx <= maxX && cell.cy >= minY && cell.cy <= maxY) {
          for (const id of this.cellMembers(cell)) ids.push(id);
        }
      }
    }
    if (ids.length > 0) {
      this.ngZone.run(() => this.selection.addAll(ids));
    }
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
    const prevLevel = this.activeLevel;
    this.fitToData();
    this.commitZoomChange(prevLevel);
    this.refreshHoverAfterZoom();
  }

  zoomBy(factor: number, anchorX = this.width / 2, anchorY = this.height / 2): void {
    const prevLevel = this.activeLevel;
    const [projX, projY] = this.screenToProj(anchorX, anchorY);
    const newZoom = Math.max(0.01, Math.min(100000, this.transform.zoom * factor));
    // Keep the point under the cursor fixed while zooming.
    this.transform.centerX = projX - (anchorX - this.width / 2) / newZoom;
    this.transform.centerY = projY - (anchorY - this.height / 2) / newZoom;
    this.transform.zoom = newZoom;

    // Zoom holds the thumbnail size (targetRadius) and re-selects the level, so
    // a smaller region is re-binned more finely while bins stay ~the same size.
    // When that re-selection crosses a level, the picture-in-picture transition
    // grows/shrinks the current frame into place before the rebin lands.
    this.updateActiveLevel();
    this.commitZoomChange(prevLevel);
    this.refreshHoverAfterZoom();
  }

  /**
   * Set the thumbnail size — the on-screen radius each bin aims to render at.
   *
   * `reframe` picks between the two callers:
   *  - `true` (the +/- thumbnail buttons): scale the view by the same factor as
   *    the size change so the level is held (level selection divides
   *    `base_radius * zoom` by `targetRadius`, and scaling both by the same
   *    factor leaves the quotient — hence the chosen level — unchanged). The
   *    *same bins* therefore just use more/fewer pixels and the visible region
   *    shrinks/grows. This is "make the thumbnails bigger", never a re-bin.
   *  - `false` (initial load / settings sync): only record the size and
   *    re-select the level at the current framing, so a saved size sets the
   *    overview granularity without yanking the viewport.
   */
  setThumbnailRadius(radius: number, reframe: boolean): void {
    if (radius <= 0 || radius === this.targetRadius) return;
    if (reframe && this.meta && this.fittedAgainstRealSize) {
      this.transform.zoom *= radius / this.targetRadius;
    }
    this.targetRadius = radius;
    this.updateActiveLevel();
    this.requestRedraw();
    if (reframe) this.refreshHoverAfterZoom();
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
    // No hover while panning or marqueeing (mid-drag), and none at all in
    // region-select mode: the cursor is a crosshair for drawing a box, so a
    // hover preview/highlight popping up under it would just be noise.
    if (this.isPanning || this.isMarquee || this.marqueeMode) return;
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
    this.clearHover();
  }

  /** Drop any pending/active hover: cancel the debounce, clear the highlighted
   *  cell, and tell the preview to close. Safe to call when nothing is hovered. */
  private clearHover(): void {
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
