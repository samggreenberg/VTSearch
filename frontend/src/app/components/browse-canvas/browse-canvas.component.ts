import { AfterViewInit, ChangeDetectionStrategy, Component, effect, ElementRef, inject, input, NgZone, OnDestroy, output, untracked, viewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { BrowseViewportService } from '../../services/browse-viewport.service';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import {
  densityColor,
  resolveColormap,
  rgbString,
  type BrowseColormapId,
  type CanvasTheme,
  type ResolvedColormap,
} from './hex-render.util';
import {
  binGeometry,
  BinGeometry,
  hoverThumbHalfExtents,
  imageTileFitDimensions,
  pickCell,
  sameBin,
  type TileFit,
} from './bin-geometry';
import {
  clampedTransform,
  computeFitZoom,
  getVisibleBounds,
  levelForEffZoom,
  projToScreen,
  screenToProj,
  softCeilZoom,
  softClampPan,
  softFloorZoom,
  type ViewGeomContext,
} from './view-transform';
import {
  layoutSigns,
  SIGN_FONT_FAMILY,
  signShadow,
  viewLevelForZoom,
} from './sign-layout';
import { prefersReducedMotion } from '../../utils/reduced-motion';
import {
  detectSoftwareRenderer,
  FramePerfMonitor,
  resolveLowEffects,
  type BrowseGraphicsMode,
} from './render-perf';
import { onDevicePixelRatioChange } from '../../utils/device-pixel-ratio';
import type {
  HexCellPayload,
  ProjectionMeta,
  RegionLabelPayload,
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
  /** The bin's representative id — the centroid clip whose thumbnail is drawn
   *  on the canvas (see ``rep_id``). The popup opens on this item and scrolls
   *  its 1-D member list to it, so the detail view starts on the same image the
   *  user right-clicked. Null over blank space. */
  repId: number | null;
  /** The canvas's bounding rect (viewport coords); the popup clamps inside it
   *  so it never spills onto the side panel or past the canvas edges. */
  bounds: DOMRect;
}

/** How much larger a hovered *flat-density* cell (text — no thumbnail) is
 *  drawn relative to its neighbours so it lifts off the grid. The border is
 *  reserved for selection state, so hover is signalled by this size bump + a
 *  soft drop shadow instead of a ring. Thumbnail cells ignore this and size
 *  their break-out rectangle by the neighbour-centre rule (`hoverThumbRect`). */
const HOVER_RADIUS_SCALE = 1.38;

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-canvas',
  standalone: true,
  templateUrl: './browse-canvas.component.html',
  styleUrl: './browse-canvas.component.scss',
})
export class BrowseCanvasComponent implements AfterViewInit, OnDestroy {
  private ngZone = inject(NgZone);
  private tileCache = inject(TileCacheService);
  private activeContext = inject(ActiveContextService);
  private viewport = inject(BrowseViewportService);
  private selection = inject(BrowseSelectionService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);

  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  readonly meta = input<ProjectionMeta | null>(null);
  /**
   * Active dataset media type. For thumbnail types (``usesThumbnails`` —
   * image / video / document / audio, audio via its waveform PNG) the
   * representative item's thumbnail is painted directly onto each tile; other
   * types (text) keep the flat density (darkred→yellow) shading.
   */
  readonly mediaType = input('');
  /** On-screen bin radius (CSS px) the "M" thumbnail size targets, and the
   * default before any saved size is applied. See {@link targetRadius}. */
  static readonly DEFAULT_TARGET_RADIUS = 28;
  /** Longest side (px) the ``/thumbnail`` route caps images at (mirrors
   * ``vtscore`` ``DEFAULT_MAX_DIM``). Once a cell is drawn wider than this the
   * capped thumbnail would upscale, so at those zoom levels the canvas fetches
   * the full-res ``/image`` instead. See {@link useFullResThumbs}. */
  static readonly THUMB_NATIVE_MAX_DIM = 384;
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
  readonly colormap = input<BrowseColormapId>('auto');
  /**
   * Width (CSS px) of the colormap-coloured border painted around thumbnails.
   * A multi-item ("pile") tile's band is the density colour for its item count,
   * so it reads as how tall the stack under the tile is; a singleton's band is
   * the colormap's dedicated one-item colour (`single`) painted as a hard-edged
   * (sharp-cornered) rectangle, so a lone item reads as "exactly one" and stands
   * out against the background. ``0`` disables both (cells fall back to the faint
   * dark separator). Only takes effect in {@link thumbnailMode} (image/video).
   */
  readonly thumbnailBorder = input(0);
  /**
   * GUI parallel to the Shift modifier: when on, a plain left-drag rubber-bands
   * a selection marquee instead of panning, so the region-select gesture is
   * discoverable without knowing the Shift+drag hotkey. Shift+drag keeps working
   * regardless. Toggled by the region-select button in the browse toolbar.
   */
  readonly marqueeMode = input(false);
  /**
   * Whether Shift is currently held (tracked by the parent view). Shift+drag
   * draws a region marquee, so while Shift is down the canvas shows the same
   * crosshair cursor as {@link marqueeMode} to preview the gesture. The actual
   * Shift+drag is detected from `event.shiftKey` at mousedown; this input only
   * drives the cursor affordance.
   */
  readonly shiftHeld = input(false);
  /**
   * Region signpost labels for the active projection — the named regions the
   * canvas letters over the map (see ``docs/plans/vtsbrowse-toponymy.md``).
   * Fetched once per projection by the parent view; empty until a labeler has
   * run for this dataset. Which signs show at a given moment is a function of
   * the zoom: see ``sign-layout.ts`` for the visibility/size bands.
   */
  readonly labels = input<RegionLabelPayload[]>([]);
  /**
   * Whether the signpost layer is drawn at all — the per-media
   * ``browse_signposts`` setting, surfaced as (and toggled by) the signpost
   * button in the browse toolbar. Off hides the signs without discarding the
   * fetched labels, so toggling back on repaints them instantly.
   */
  readonly signposts = input(true);
  /**
   * How many wheel notches cross one pyramid level (a full 2x of zoom) — the
   * per-media ``browse_mouse_zooms_per_level`` setting. The per-notch width
   * factor is ``2 ** (1 / n)``, so 1 ⇒ 2x (one notch per level), 2 ⇒ √2 (the
   * default, two notches per level), 3 ⇒ ∛2. Clamped to 1..3 by
   * {@link wheelZoomFactor}; the +/- buttons in the parent use the same factor
   * so the two gestures stay in lock-step.
   */
  readonly zoomsPerLevel = input(2);
  /**
   * Canvas rendering effort — the ``browse_graphics`` setting, surfaced as the
   * "Graphics" pulldown at the top of Settings → Browser. ``auto`` (the
   * default) lets {@link perf} decide per client; ``full`` and ``reduced`` pin
   * the pipeline. See {@link lowFx} and `render-perf.ts`.
   */
  readonly graphics = input<BrowseGraphicsMode>('auto');
  readonly hexHover = output<HexHoverEvent | null>();
  /** A right-click on the canvas; the view opens the bin popup in response. */
  readonly contextMenu = output<BrowseContextMenuEvent>();
  /**
   * The densest visible cell's item count, emitted whenever it changes. Density
   * shading is renormalized to this per frame (yellow = this many items, the
   * darkest red = 1), so the legend reads it to label the ramp with live
   * numbers that track pan/zoom.
   */
  readonly densityMaxChanged = output<number>();
  /**
   * Emitted once, after the opening view is fully painted with real content:
   * the fit has run against the real canvas size, every density tile under the
   * viewport is in, and — for image/video datasets — every on-screen
   * representative thumbnail has decoded (or failed). The browse view keeps a
   * cover over the canvas until this fires, so the user is shown the finished
   * view rather than a thumbnail-less grey grid that then fills in. Fires
   * exactly once per canvas instance (a later pan/zoom loads on demand as
   * before). See {@link maybeReportFirstView}.
   */
  readonly firstViewReady = output<void>();

  /** Loaded representative thumbnails, keyed by media id (insertion-ordered LRU). */
  private thumbCache = new Map<number, HTMLImageElement>();
  /** Media ids whose thumbnail failed to load, so we don't retry every frame. */
  private thumbFailed = new Set<number>();
  private readonly MAX_THUMBS = 2048;
  /** Audio waveform thumbnails are theme-agnostic alpha masks (issue #2369);
   *  this caches each one tinted to the current theme's accent colour so the
   *  {@link ImageBitmap}-style ``source-in`` fill runs once per (clip, theme),
   *  not per frame. Cleared whenever the raw {@link thumbCache} is (dataset /
   *  level switch) and — via {@link retintWaveforms} — on a theme flip. */
  private tintedThumbCache = new Map<number, HTMLCanvasElement>();
  /** Resolution tier the {@link thumbCache} is currently filled at. Flips to
   * ``true`` once the zoom is large enough that {@link getThumb} fetches the
   * full-res ``/image`` instead of the capped ``/thumbnail``; crossing the
   * threshold drops the cache so cells reload at the matching resolution. */
  private thumbsAreFullRes = false;
  /** Cap on new thumbnail fetches kicked off per idle preload pass, so a fast
   * pan can warm the just-revealed ring without flooding the network or filling
   * the thumbnail cache in a single burst. */
  private static readonly PRELOAD_MAX_PER_PASS = 64;
  /** Fraction of {@link PRELOAD_MAX_PER_PASS} reserved for warming the *finer*
   * level a zoom-in would reveal, so a wide pan ring can't starve the zoom path.
   * Pan warms the rest first; whichever side has nothing left to warm hands its
   * unused budget back to the other, so the split only bites when both compete. */
  private static readonly THUMB_PRELOAD_ZOOM_SHARE = 0.5;
  /** Handle for the pending idle thumbnail-preload pass (at most one in flight),
   * with a flag recording whether it was scheduled via ``setTimeout`` (the
   * fallback when ``requestIdleCallback`` is unavailable) so destroy cancels it
   * the matching way. */
  private thumbPrefetchHandle: number | null = null;
  private thumbPrefetchIsTimeout = false;
  /** Smoothed pan direction (projection units, magnitude < 1) and the view
   * centre it was last measured from. Biases which off-view tiles warm first:
   * the preload ring extends one tile further on the leading edges so a sustained
   * pan stays ahead of the motion. Decays toward zero when the view holds still,
   * so a stationary view warms its ring symmetrically. */
  private panDirX = 0;
  private panDirY = 0;
  private lastDrawCenterX = Number.NaN;
  private lastDrawCenterY = Number.NaN;

  private ctx!: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  // --- Low-effects mode for software-rendered environments (issue #2695) ----
  // Some deployments run the browser with hardware acceleration permanently
  // off, so every canvas paint rasterizes on the CPU. The animation pipeline's
  // GPU-cheap staples — full-canvas smoothed blits, overscanned snapshots up
  // to 4× the viewport, shadow blurs — each cost tens of ms per frame there,
  // making zooming near-unusable. Rather than forcing those users to switch
  // animations off, `lowFx` keeps every animation running but strips the
  // per-frame cost: viewport-sized snapshots only (no overscan ring),
  // nearest-neighbour blits, no shadow blurs, and a dpr capped at 1. Seeded
  // upfront by the WebGL renderer probe (SwiftShader et al.) and latched at
  // runtime by measured frame costs — see `render-perf.ts` for both. The
  // {@link graphics} setting overrides the detection in either direction.
  private readonly perf = new FramePerfMonitor(detectSoftwareRenderer());

  /** Whether to paint frames in the cheap, software-rasterizer-friendly way. */
  private get lowFx(): boolean {
    return resolveLowEffects(this.graphics(), this.perf.slow);
  }

  /**
   * Feed the monitor one frame's paint duration (from `startTs`, a
   * `performance.now()` stamp taken before painting). When this very sample
   * latches low-effects mode on a hiDPI display, schedule a one-time
   * {@link resize} so the backing store shrinks to the capped dpr — the single
   * biggest per-frame saving on such displays. Scheduled (not run inline)
   * because this is called from inside paint paths; the resize cancels any
   * in-flight animation, which is an acceptable one-time snap.
   */
  private recordFrameCost(startTs: number): void {
    const wasLow = this.lowFx;
    this.perf.record(performance.now() - startTs);
    // Only an *effective* flip matters: with the setting pinned to full or
    // reduced the latch changes nothing on screen, so there is nothing to
    // re-measure the backing store for.
    if (!wasLow && this.lowFx) this.syncBackingStoreToEffects();
  }

  /**
   * Re-run {@link resize} when the effective effects level changes, so the
   * backing store picks up (or drops) the low-effects dpr cap — the single
   * biggest per-frame saving on a hiDPI display. Deferred rather than run
   * inline because callers are mid-paint, and the resize cancels any in-flight
   * animation; that one-time snap is the cost of switching pipelines. A no-op
   * where the cap can't bite (the device is already at dpr 1).
   */
  private syncBackingStoreToEffects(): void {
    if (window.devicePixelRatio <= 1) return;
    setTimeout(() => {
      if (!this.destroyed) this.resize();
    });
  }

  private transform: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  // The transform the pixels currently on the canvas were painted at. After a
  // normal draw() this mirrors `transform`; during a zoom transition it tracks
  // the interpolated frame so a re-triggered transition can chain from what's
  // actually on screen. Seeds the "from" end of the zoom-in/out animation.
  private displayedTransform: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  // The pyramid level the pixels currently on the canvas were binned at. Paired
  // with `displayedTransform` so a zoom-out can re-render the off-screen margin
  // (revealed as the frame shrinks) using the *source* level's bins, keeping the
  // overscan continuous with the frozen centre. See {@link renderSnapshotBorder}.
  private displayedLevel = 0;
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
  // Guards {@link firstViewReady} so the opening-view signal fires exactly once
  // per canvas instance.
  private firstViewReported = false;
  // Backstop so a representative thumbnail that never resolves (a hung request)
  // can't strand the reveal cover: armed the first time we begin waiting on the
  // opening view's thumbnails, it releases the view after this long regardless.
  private firstViewTimer: ReturnType<typeof setTimeout> | null = null;
  private static readonly FIRST_VIEW_MAX_WAIT_MS = 12000;
  // Whether the user has taken over the framing (panned, zoomed, or resized a
  // thumbnail). Until then the view stays auto-fit: applying a saved cell size
  // (which changes the bin radius and thus how far edge bins reach) re-fits so
  // the projection opens fully in view, rather than leaving the initial fit —
  // computed for the default radius — clipping the now-larger edge bins.
  private framedByUser = false;

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
  // The bin the pending single-click toggle will flip, resolved at click time
  // (not when the timer fires). The defer is only there to let a double-click
  // preempt the toggle; it must not defer the *hit-test*, because a wheel notch
  // or arrow-key glide inside the double-click window moves the transform, so a
  // late hit-test of the captured screen point would land on a different bin.
  // Binding the cell here keeps the toggle on the bin the user actually clicked.
  private pendingToggleCell: HexCellPayload | null = null;
  private static readonly DBLCLICK_MS = 250;
  // `event.timeStamp` of the last right-click that landed on empty canvas (no
  // bin under the cursor), used to detect a double-right-click there; see
  // {@link onContextMenu}. Reset to 0 whenever a right-click lands on a bin
  // or a double is consumed, so only two *consecutive empty* right-clicks
  // within DBLCLICK_MS pair up.
  private lastEmptyContextMenuAt = 0;
  // Per-notch wheel zoom factor. A pyramid level spans a full 2x of zoom, so the
  // {@link zoomsPerLevel} setting (n) sets how many notches cross a bin layer:
  // the factor is 2 ** (1 / n). At the default n=2 that is exactly √2, so two
  // notches = exactly one level (and exactly one DOUBLE_CLICK_ZOOM). An exact
  // power of 2 matters here: an earlier 1.4 was a hair under √2, making a level
  // cost log_1.4(2) ≈ 2.06 notches; the 0.06 surplus accumulates, so when the
  // whole-projection fit happens to land near the bottom edge of a level band
  // the *first* flip from that overview rounds up to a 3rd notch (then ~2
  // thereafter) — felt as an inconsistent "three at the top, two after". A clean
  // 2 ** (1 / n) makes every flip cost exactly n notches.
  private wheelZoomFactor(): number {
    const n = Math.max(1, Math.min(3, Math.round(this.zoomsPerLevel())));
    return Math.pow(2, 1 / n);
  }
  // How hard a double-click zooms in about the cursor. Larger than the wheel's
  // per-notch factor so the gesture lands a decisive jump, matching the map idiom.
  private static readonly DOUBLE_CLICK_ZOOM = 2.0;

  // Shift+drag draws a marquee rectangle (canvas-relative screen coords) that
  // adds every bin whose centre falls inside it to the selection — the fast path
  // for grabbing a region, since plain drag is reserved for panning.
  private isMarquee = false;
  private marquee: { x0: number; y0: number; x1: number; y1: number } | null = null;

  // Accent colour resolved from the live theme once per frame, used for the
  // selection rings and the marquee rectangle.
  private selAccent = '#4f9dff';

  // Waveform tint colours resolved from the live theme once per frame. Audio
  // datasets paint each tile as a themed surface (``waveSurface``) with the
  // wave masked in the accent (``waveAccent``); see {@link drawCell} and
  // issue #2369. ``waveformTint`` is the per-frame "this dataset is audio" flag.
  private waveformTint = false;
  private waveAccent = '#4f9dff';
  private waveSurface = '#1a1d27';

  // Per-frame "this dataset is images" flag. Image grid tiles use a balanced
  // "half crop, half pad" fit (see {@link drawImageFit}) instead of the cover
  // crop used for video/audio, leaving a background-coloured gap inside the
  // bin's border; hovering still breaks the whole image out unchanged.
  private imageTiles = false;

  private hoveredCell: HexCellPayload | null = null;
  /**
   * Pyramid level {@link hoveredCell} was resolved at, or -1 when nothing is
   * hovered. A cell's axial ``(q, r)`` is only unique *within* a level, so the
   * level travels alongside the payload (which carries no level of its own) and
   * every identity check goes through {@link sameBin}. Without it a zoom that
   * crosses a level boundary matches the stale hover against a different,
   * finer same-``(q, r)`` cell — see issue #2967.
   */
  private hoveredLevel = -1;
  /**
   * A bin "pinned" enlarged because its detail popup (right-click) is open.
   * Independent of the live hover: it stays enlarged as long as the popup is up,
   * so the user can tell which bin the details belong to. While a cell is pinned,
   * hover on other bins is suppressed — the detail popup has precedence — until
   * the popup is dismissed ({@link unpinCell}) or right-clicking another bin
   * pins that one instead. Rendered enlarged via the same {@link drawHoveredHex}
   * path as a hover; see the ``pinnedCell ?? hoveredCell`` pick in {@link draw}.
   */
  private pinnedCell: HexCellPayload | null = null;
  /** Pyramid level {@link pinnedCell} was pinned at, or -1 when nothing is
   *  pinned. Same role as {@link hoveredLevel}: the pin survives wheel zooms, so
   *  after a level-crossing zoom the pinned bin no longer exists among the drawn
   *  cells and nothing is enlarged — rather than the wrong bin being enlarged
   *  while the details panel still describes the original. */
  private pinnedLevel = -1;
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
  /** Set in ngOnDestroy: late async callbacks (thumbnail loads, tile
   *  responses) must not schedule new rAF / idle work on a dead component. */
  private destroyed = false;
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
  // CSS-px footprint of `animSnapshot`. Normally the viewport (width×height),
  // but a zoom-out overscans (snapshot bigger than the viewport) so the shrunk
  // blit still covers the canvas instead of leaving black margins where the
  // newly-revealed area sits. See {@link startZoomAnim} / {@link zoomBlitRect}.
  private snapW = 0;
  private snapH = 0;
  // How far a zoom-out snapshot overscans the viewport, per axis. Caps the
  // overscan buffer at 2×2 the canvas (4× the area / memory); a deeper zoom-out
  // than this just falls back to a black falloff at the far edge.
  private static readonly SNAP_OVERSCAN_MAX = 2;
  private animFrom: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private animTo: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private animStartTs = 0;
  private animRafId = 0;
  // Background colour captured at transition start, so each frame doesn't pay a
  // getComputedStyle just to clear behind the shrinking/growing snapshot.
  private animBg = '';

  // --- Boundary settle (rubber-band snap-back) ------------------------------
  // After a pan/zoom leaves the view past its hard bounds (rubber-band
  // overshoot), this eases the *real* transform back to the clamp. Distinct from
  // the picture-in-picture zoom transition above: it walks the live transform
  // and repaints each frame, rather than blitting a frozen snapshot. The give /
  // overshoot-cap constants for the rubber-band curve itself live next to
  // `rubber()` in `view-transform.ts`.
  /** Snap-back duration. */
  private static readonly SETTLE_MS = 320;
  private settleActive = false;
  private settleRafId = 0;
  private settleFrom: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private settleTo: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private settleStartTs = 0;
  // Wheel zoom arrives as a burst of discrete events with no "gesture end", so
  // the snap-back is debounced: it fires only once the wheel has gone quiet.
  private settleTimer: ReturnType<typeof setTimeout> | null = null;

  // --- Directional (arrow-key) pan glide ------------------------------------
  // A drag pan tracks the pointer (naturally smooth) and a zoom plays the
  // picture-in-picture transition, but an arrow-key push used to snap the centre
  // in one frame, which reads as a jarring jump. Instead glide there over
  // ~PAN_ANIM_MS with the same easeOutCubic the zoom/settle use, so a N/E/S/W
  // push reads like a zoom does.
  //
  // A pan is the zoom=1 (pure-translation) case of the picture-in-picture zoom
  // transition, so it uses the same trick: freeze the current frame to an
  // offscreen snapshot and blit it translated each step, then paint the real,
  // rebinned frame once when the glide lands. An earlier version walked the live
  // transform and ran a full {@link draw} every frame — which re-painted every
  // visible thumbnail, so a glide over a dense screen stuttered (worst at the
  // start, where easeOutCubic moves fastest and the leading edge is decoding
  // freshly-revealed thumbnails). The blit costs one snapshot plus cheap copies
  // instead, independent of how many thumbnails are in view. The snapshot
  // overscans toward the move so the revealed edge shows cached content as it
  // slides in (falling off to background past the cache, like a zoom-out). A
  // side benefit: because the glide never calls {@link draw}, it never republishes
  // the viewport, so the minimap rectangle jumps once at the end rather than
  // animating along — matching the zoom transition.
  private static readonly PAN_ANIM_MS = 220;
  private panAnimActive = false;
  private panAnimRafId = 0;
  private panAnimFrom: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private panAnimTo: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private panAnimStartTs = 0;
  // Offscreen frozen frame for the glide (reused across glides), its CSS-px
  // footprint (viewport + overscan margins), and the background captured at
  // glide start so each blit frame clears without a getComputedStyle.
  private panSnapshot: HTMLCanvasElement | null = null;
  private panSnapW = 0;
  private panSnapH = 0;
  private panAnimBg = '';

  private resizeObserver: ResizeObserver | null = null;
  // Teardown for the devicePixelRatio-change listener. A pure density change
  // (dragging to a different-density monitor) leaves the CSS box unchanged, so
  // the ResizeObserver never fires; this re-runs `resize()` to refresh the
  // backing store and the thumbnail-resolution tier.
  private dprListenerTeardown: (() => void) | null = null;
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

  // Repaint when the selection changes so the per-cell selection rings track
  // the live set. An effect (not a subscription) so a signal write — including
  // one from a raw canvas event handler — schedules the redraw under zoneless
  // without an NgZone.run re-entry. The first run (post-ngOnInit) just queues a
  // harmless redraw. Unselected cells are left untouched: a selection elsewhere
  // never dims or otherwise alters them.
  private readonly selectionRedraw = effect(() => {
    this.selection.version();
    this.requestRedraw();
  });

  // The input→redraw dispatch that used to live in ngOnChanges (signal inputs
  // don't fire it). Each effect tracks exactly the inputs its old ngOnChanges
  // arm keyed on and runs the body untracked, so incidental signal reads inside
  // (selection state, meta re-reads, service calls) can't widen its triggers.

  /** Fresh meta re-bins (new projection, bin-shape toggle, cull). */
  private readonly metaChanged = effect(() => {
    const meta = this.meta();
    if (!meta) return;
    untracked(() => {
      // Any zoom transition was easing the *old* data, so abandon it; the
      // redraw below paints the new state. A boundary settle would spring
      // toward the old bounds, so stop it too — as would a directional glide.
      this.cancelZoomAnim();
      this.cancelSettle();
      this.cancelPanAnim();
      this.tileCache.setProjectionId(meta.projection_id);
      // A bin-shape toggle delivers fresh meta for the *same* projection id
      // (hex and square share one UMAP layout). In that case keep the current
      // pan/zoom and just re-bin visually; only a genuinely new projection
      // re-frames to data and drops stale representative thumbnails.
      if (meta.projection_id !== this.lastProjectionId) {
        this.lastProjectionId = meta.projection_id;
        // A brand-new projection opens auto-fit: clear the user-framed flag so a
        // saved cell size applied right after load re-fits to keep it all in view.
        this.framedByUser = false;
        this.thumbCache.clear();
        this.thumbFailed.clear();
        this.tintedThumbCache.clear();
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
      } else {
        this.updateActiveLevel();
      }
      this.requestRedraw();
    });
  });

  /** Entering region-select mode: drop any hover preview/highlight that was
   *  showing, since hover is suppressed while the mode is on. */
  private readonly marqueeModeChanged = effect(() => {
    if (this.marqueeMode()) untracked(() => this.clearHover());
  });

  /** Switching the "Graphics" setting changes what a frame costs to paint, so
   *  the backing store's dpr cap may need to come or go; the resize repaints.
   *  When the cap can't bite (dpr 1) a plain repaint picks up the new effects.
   *  Skips the first run: `ngAfterViewInit`'s initial `resize` already sizes
   *  the backing store against the starting mode. */
  private readonly graphicsChanged = effect(() => {
    this.graphics();
    untracked(() => {
      if (!this.hasDrawn) return;
      this.syncBackingStoreToEffects();
      this.requestRedraw();
    });
  });

  /** Repaint-only inputs: a colormap change only affects flat (non-thumbnail)
   *  shading; the pile-thumbnail border only changes how thumbnail cells are
   *  stroked; labels arriving (they load async, after the tiles) or the
   *  signpost layer toggling only changes the sign overlay. A coalesced repaint
   *  picks any of them up without re-binning or re-fetching tiles. */
  private readonly repaintInputsChanged = effect(() => {
    this.colormap();
    this.thumbnailBorder();
    this.labels();
    this.signposts();
    untracked(() => this.requestRedraw());
  });

  /** True when cells should be painted with the central item's thumbnail. */
  private get thumbnailMode(): boolean {
    return this.mediaTypeCaps.usesThumbnails(this.mediaType());
  }

  /** At the largest zoom levels a cell is drawn wider (in device px) than the
   * thumbnail's native longest side, so painting the capped ``/thumbnail`` would
   * just upscale a blurry bitmap. Past that point fetch the full-res ``/image``
   * instead. Only a handful of such giant cells fit on screen at once, so the
   * LRU still bounds memory. */
  private get useFullResThumbs(): boolean {
    return 2 * this.targetRadius * this.dpr > BrowseCanvasComponent.THUMB_NATIVE_MAX_DIM;
  }

  /** Drop the thumbnail cache when the zoom crosses the full-res threshold so
   * cells reload at the resolution matching the new tier. Cheap no-op while the
   * tier is unchanged. */
  private syncThumbResolutionTier(): void {
    if (this.useFullResThumbs === this.thumbsAreFullRes) return;
    this.thumbsAreFullRes = this.useFullResThumbs;
    this.thumbCache.clear();
    this.thumbFailed.clear();
    this.tintedThumbCache.clear();
  }

  /** Geometry (hex or square) for the active projection's bin shape. */
  private get geom(): BinGeometry {
    return binGeometry(this.meta()?.bin_shape);
  }

  /** On-screen scale (projection units → CSS px). Used for all projection↔screen
   * conversions and the rendered bin radius. Thumbnail size no longer folds in
   * here: it lives in {@link targetRadius} (level selection) and, for the +/-
   * buttons, in a matching {@link transform}.zoom change, so "Zoom" and
   * "thumbnail size" are now distinct operations rather than the same multiply. */
  private get effZoom(): number {
    return this.transform.zoom;
  }

  ngAfterViewInit(): void {
    this.ctx = this.canvasRef().nativeElement.getContext('2d')!;

    this.tileLoadSub = this.tileCache.tileLoaded$.subscribe(() => {
      this.requestRedraw();
    });

    // The minimap publishes recenter requests when the user clicks/drags it;
    // jump the viewport centre there (keeping zoom) and redraw.
    this.recenterSub = this.viewport.recenter$.subscribe(({ x, y }) => {
      // A minimap jump is a programmatic move, not an elastic gesture: cancel any
      // snap-back / directional glide and hard-clamp straight to the bounds.
      this.cancelSettle();
      this.cancelPanAnim();
      this.transform.centerX = x;
      this.transform.centerY = y;
      // A minimap click/drag can target a content edge; keep the viewport inside
      // the useful bounds just as a direct pan does.
      this.clampView();
      this.requestRedraw();
    });

    this.resizeObserver = new ResizeObserver(() => {
      this.ngZone.runOutsideAngular(() => this.resize());
    });
    this.resizeObserver.observe(this.canvasRef().nativeElement.parentElement!);

    // A pure devicePixelRatio change (monitor-to-monitor drag, browser zoom, OS
    // scaling) doesn't touch the element box, so the ResizeObserver stays quiet;
    // re-run resize() to rebuild the backing store at the new density.
    this.dprListenerTeardown = onDevicePixelRatioChange(() => {
      this.ngZone.runOutsideAngular(() => this.resize());
    });

    this.themeObserver = new MutationObserver(() => {
      // A theme flip changes the accent the waveform masks are tinted with, so
      // drop the tinted cache (issue #2369); each visible wave re-tints on the
      // redraw below. The raw mask cache is untouched — the masks are
      // theme-agnostic, so nothing needs re-fetching.
      this.tintedThumbCache.clear();
      this.requestRedraw();
    });
    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });

    this.ngZone.runOutsideAngular(() => {
      const el = this.canvasRef().nativeElement;
      el.addEventListener('mousedown', this.boundMouseDown);
      el.addEventListener('wheel', this.boundWheel, { passive: false });
      el.addEventListener('mousemove', this.boundCanvasMouseMove);
      el.addEventListener('mouseleave', this.boundCanvasMouseLeave);
      el.addEventListener('dblclick', this.boundDblClick);
      el.addEventListener('contextmenu', this.boundContextMenu);
    });
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.tileLoadSub?.unsubscribe();
    this.recenterSub?.unsubscribe();
    this.viewport.setViewport(null);
    this.resizeObserver?.disconnect();
    this.dprListenerTeardown?.();
    this.themeObserver?.disconnect();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.animRafId) cancelAnimationFrame(this.animRafId);
    if (this.settleRafId) cancelAnimationFrame(this.settleRafId);
    if (this.panAnimRafId) cancelAnimationFrame(this.panAnimRafId);
    if (this.settleTimer) clearTimeout(this.settleTimer);
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    if (this.clickTimer) clearTimeout(this.clickTimer);
    if (this.thumbPrefetchHandle !== null) {
      if (this.thumbPrefetchIsTimeout) clearTimeout(this.thumbPrefetchHandle);
      else if (typeof cancelIdleCallback === 'function') cancelIdleCallback(this.thumbPrefetchHandle);
    }
    if (this.firstViewTimer !== null) clearTimeout(this.firstViewTimer);
    // `ctx` is only assigned in ngAfterViewInit — the same place the canvas
    // listeners are attached — so it doubles as "the view query resolved" and
    // keeps a destroy-before-render from reading the required query.
    if (this.ctx) {
      const el = this.canvasRef().nativeElement;
      el.removeEventListener('mousedown', this.boundMouseDown);
      el.removeEventListener('wheel', this.boundWheel);
      el.removeEventListener('mousemove', this.boundCanvasMouseMove);
      el.removeEventListener('mouseleave', this.boundCanvasMouseLeave);
      el.removeEventListener('dblclick', this.boundDblClick);
      el.removeEventListener('contextmenu', this.boundContextMenu);
    }
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    this.thumbCache.clear();
    this.thumbFailed.clear();
    this.tintedThumbCache.clear();
  }

  private resize(): void {
    // A resize changes the backing store and framing; a snapshot blit sized to
    // the old canvas would be wrong, so drop any in-flight transition. A settle
    // would spring toward stale bounds, so stop it too — the refit/clamp below
    // re-establishes the correct framing.
    this.cancelZoomAnim();
    this.cancelSettle();
    this.cancelPanAnim();
    const el = this.canvasRef().nativeElement.parentElement!;
    const rect = el.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    // Software rasterization pays per backing-store pixel, so on a hiDPI
    // display low-effects mode renders at CSS resolution (dpr 1): a quarter
    // of the pixels for a slight softness — the standard degraded-mode trade
    // (issue #2695). Hardware-accelerated environments keep the native dpr.
    if (this.lowFx && this.dpr > 1) this.dpr = 1;
    this.width = rect.width;
    this.height = rect.height;
    const canvas = this.canvasRef().nativeElement;
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // A devicePixelRatio change (e.g. dragging to a different-density display)
    // can also cross the full-res threshold, so re-evaluate the tier here.
    this.syncThumbResolutionTier();
    // The initial fit may have run against the 800x600 fallback (meta arrived
    // before layout). Now that the real size is known, refit to it so the
    // framing and the viewport bounds published to the minimap match the actual
    // canvas. Only on the first real measurement — a later window resize keeps
    // the user's pan/zoom.
    if (!this.fittedAgainstRealSize && this.meta() && this.width > 0 && this.height > 0) {
      this.fitToData();
    } else {
      // The viewport changed size: a shrunk canvas raises the fit floor and a
      // grown one extends past the content, so re-clamp the kept pan/zoom.
      this.clampView();
      this.updateActiveLevel();
    }
    this.requestRedraw();
  }

  private fitToData(): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    const [xmin, ymin, xmax, ymax] = meta.bounds;
    this.transform.zoom = this.computeFitZoom();
    this.transform.centerX = (xmin + xmax) / 2;
    this.transform.centerY = (ymin + ymax) / 2;
    // Mark whether this fit used the real canvas size (vs the 800x600 fallback),
    // so `resize()` knows whether a corrective refit is still owed.
    this.fittedAgainstRealSize = this.width > 0 && this.height > 0;
    this.updateActiveLevel();
  }

  // --- Pan/zoom/clamp/rubber-band geometry -----------------------------------
  // The pure coordinate-transform and boundary-clamping math (fit/max zoom,
  // hard + rubber-band clamping, level selection, proj<->screen conversion)
  // lives in the framework-free `view-transform.ts` module so it's independently
  // unit-testable. These are thin delegating wrappers that supply the live
  // `meta`/`width`/`height`/`targetRadius`/`transform` state; see that module
  // for the actual logic and the invariants behind it (why `[lo,hi]` clamping
  // doesn't pin to centre, why rubber-band overshoot works in log-space for
  // zoom, why level selection floors at the fit-zoom level, etc.).

  /** Bundles the live state {@link computeFitZoom} and friends need. */
  private geomCtx(): ViewGeomContext {
    return { meta: this.meta(), width: this.width, height: this.height, targetRadius: this.targetRadius };
  }

  /** See `computeFitZoom` in `view-transform.ts`: the whole-projection-fit zoom,
   *  the floor {@link clampView} holds the user to. */
  private computeFitZoom(): number {
    return computeFitZoom(this.geomCtx(), this.transform.zoom);
  }

  /**
   * Keep the view within the useful bounds: never zoomed out past the
   * whole-projection fit, never panned so the viewport *centre* leaves the
   * content. Mutates {@link transform} in place; callers re-select the level and
   * redraw. Safe to call repeatedly (idempotent once inside the bounds). See
   * `clampedTransform` in `view-transform.ts` for the actual clamping logic.
   */
  private clampView(): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    if (this.width <= 0 || this.height <= 0) return;
    const c = this.clampedTransform(this.transform);
    this.transform.zoom = c.zoom;
    this.transform.centerX = c.centerX;
    this.transform.centerY = c.centerY;
  }

  /** The hard-clamped form of `t`; see `clampedTransform` in `view-transform.ts`.
   *  Pure: returns a fresh transform without touching {@link transform}, so the
   *  boundary-settle animation can compute its destination while the live view
   *  is still mid-overshoot. Caller is responsible for the meta/size guards. */
  private clampedTransform(t: ViewTransform): ViewTransform {
    return clampedTransform(t, this.geomCtx());
  }

  /** Soft analogue of the pan half of {@link clampView}: rather than pin the
   *  centre at the content edge, let it drift past with rubber-band resistance,
   *  so a drag into the wall still moves a little. Mutates {@link transform}.
   *  See `softClampPan` in `view-transform.ts`. */
  private softClampPan(z: number): void {
    const p = softClampPan(this.transform, this.geomCtx(), z);
    this.transform.centerX = p.centerX;
    this.transform.centerY = p.centerY;
  }

  /** Soft analogue of the zoom floor in {@link clampView}; see `softFloorZoom`
   *  in `view-transform.ts`. */
  private softFloorZoom(rawZoom: number): number {
    return softFloorZoom(rawZoom, this.geomCtx(), this.transform.zoom);
  }

  /** Soft analogue of the zoom ceiling in {@link clampView}; see `softCeilZoom`
   *  in `view-transform.ts`. Mirror of {@link softFloorZoom}. */
  private softCeilZoom(rawZoom: number): number {
    return softCeilZoom(rawZoom, this.geomCtx(), this.transform.zoom);
  }

  /** Pyramid level whose bins render closest to the current thumbnail size
   * ({@link targetRadius}) at the given on-screen zoom. Shared by live level
   * selection and fit framing. See `levelForEffZoom` in `view-transform.ts`. */
  private levelForEffZoom(effZoom: number): number {
    return levelForEffZoom(this.geomCtx(), effZoom);
  }

  private updateActiveLevel(): void {
    const meta = this.meta();
    if (!meta || meta.levels.length === 0) return;
    // Floor the level at the coarsest one reachable in "normal space" (the
    // level the fit zoom lands on). Zooming out past the whole-projection fit
    // only happens in the rubber-band overshoot zone, and that overshoot is
    // always sprung back by the settle — so re-binning to a coarser level out
    // there would shift the bins, then shift them straight back when the view
    // snaps home. That double re-lay-out reads as a glitch. Holding the level
    // at the floor keeps the bins put while the elastic edge does its bounce.
    const floorLevel = this.levelForEffZoom(this.computeFitZoom());
    this.activeLevel = Math.max(floorLevel, this.levelForEffZoom(this.effZoom));
  }

  private projToScreen(px: number, py: number): [number, number] {
    return projToScreen(px, py, this.transform, this.width, this.height);
  }

  private screenToProj(sx: number, sy: number): [number, number] {
    return screenToProj(sx, sy, this.transform, this.width, this.height);
  }

  private getVisibleBounds(): [number, number, number, number] {
    return getVisibleBounds(this.transform, this.width, this.height);
  }

  private getVisibleTiles(): { tx: number; ty: number }[] {
    const meta = this.meta();
    if (!meta) return [];
    const level = this.activeLevel;
    const radius = meta.base_radius / Math.pow(2, level);
    const geom = this.geom;
    const tileSpan = meta.tile_span;
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
    // Late async callbacks (a thumbnail Image finishing its download, a tile
    // response) can land after destroy; ngOnDestroy cancels the *current*
    // rAF handle, so a redraw scheduled here afterwards would run draw() on
    // the destroyed component — emitting on destroyed outputs, republishing
    // a non-null viewport over the teardown's null, and re-arming the idle
    // thumb-prefetch loop.
    if (this.destroyed) return;
    // A zoom transition or boundary snap-back owns the canvas while it runs;
    // tile loads, hover and selection repaints that arrive mid-animation are
    // folded into the real frame it paints each step / when it lands (see
    // {@link endZoomAnim} and {@link stepSettle}).
    if (this.animActive || this.settleActive || this.panAnimActive) return;
    if (this.needsRedraw) return;
    this.needsRedraw = true;
    this.rafId = requestAnimationFrame(() => {
      this.needsRedraw = false;
      this.draw();
    });
  }

  /**
   * Commit a zoom change after `transform` and `activeLevel` have been updated.
   * Plays the picture-in-picture transition on every zoom, whether or not the
   * pyramid level flipped: across a level the snapshot blit hides the re-lay-out
   * of the bins, and within a level it smoothly scales the same bins to their new
   * size — either way the eased grow/shrink reads as one consistent zoom gesture
   * instead of a snap. A zoom that lands while a transition is already running
   * retargets it so it eases on to the latest view instead of stopping short.
   */
  private commitZoomChange(): void {
    if (this.animActive) {
      // A transition is already running (e.g. a burst of wheel notches): ease on
      // to the new target rather than stopping short at the old one.
      this.animTo = { ...this.transform };
    } else if (this.hasDrawn && this.width > 0 && !prefersReducedMotion()) {
      this.startZoomAnim();
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
    const canvasEl = this.canvasRef().nativeElement;
    this.animFrom = { ...this.displayedTransform };
    this.animTo = { ...this.transform };

    // Zoom-out shrinks the frozen frame, exposing the area that was just off
    // screen as a margin around it. With a plain viewport-sized snapshot that
    // margin is bare background (the black border). Instead overscan: snapshot a
    // buffer larger than the viewport — the frozen centre is the already-rendered
    // canvas (free), and the extra ring is re-rendered from the source level's
    // bins, painting whatever the off-view prefetch already warmed: cached
    // thumbnails in thumbnail mode, colour-mapped hexes in flat-density mode
    // (audio/text). Both come straight from the tile cache, so the ring fills the
    // margin for every media type rather than only thumbnail ones. The shrunk
    // blit then covers the canvas with real content. Zoom-in needs no overscan
    // (the frame grows past the viewport), so it stays the cheap copy.
    // Low-effects mode skips the overscan entirely: the buffer is up to 4× the
    // viewport's pixels, and both filling it (renderSnapshotBorder walks every
    // ring bin) and blitting it each frame are what software rasterization
    // chokes on. The margin falls back to bare background during the shrink,
    // which is far preferable to a stuttering ease.
    const overscan = this.lowFx
      ? 1
      : Math.min(BrowseCanvasComponent.SNAP_OVERSCAN_MAX, this.animFrom.zoom / this.animTo.zoom);
    const doOverscan = overscan > 1.01;
    this.snapW = doOverscan ? Math.ceil(this.width * overscan) : this.width;
    this.snapH = doOverscan ? Math.ceil(this.height * overscan) : this.height;

    let snap = this.animSnapshot;
    if (!snap) snap = document.createElement('canvas');
    // Match the live canvas exactly on the non-overscan path so the plain copy
    // stays pixel-for-pixel (the device size may floor differently from snapW×dpr).
    const wantW = doOverscan ? Math.round(this.snapW * this.dpr) : canvasEl.width;
    const wantH = doOverscan ? Math.round(this.snapH * this.dpr) : canvasEl.height;
    if (snap.width !== wantW || snap.height !== wantH) {
      snap.width = wantW;
      snap.height = wantH;
    }
    const sctx = snap.getContext('2d')!;
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.clearRect(0, 0, snap.width, snap.height);
    if (doOverscan) {
      // Margin first (in CSS px via the dpr transform), then drop the frozen
      // viewport copy into the centre so it overwrites any seam bins exactly.
      sctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.renderSnapshotBorder(sctx, this.animFrom, this.snapW, this.snapH, this.displayedLevel);
      sctx.setTransform(1, 0, 0, 1, 0, 0);
      const ox = Math.round((this.snapW - this.width) * this.dpr * 0.5);
      const oy = Math.round((this.snapH - this.height) * this.dpr * 0.5);
      sctx.drawImage(canvasEl, ox, oy);
    } else {
      sctx.drawImage(canvasEl, 0, 0);
    }
    this.animSnapshot = snap;

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
   * Paint the overscan margin of a zoom-out snapshot: the source-level bins that
   * sit just outside the frozen viewport. `sctx` is the snapshot context with a
   * CSS-px (dpr) transform already applied and the buffer centred on
   * {@link animFrom}'s centre. Only cells whose tile is already cached are drawn
   * (and `getThumb` paints whatever the prefetch warmed), so the fill reaches as
   * far as the cache does and falls off to background past it — never a network
   * wait. The centre is left for the frozen viewport copy to overwrite.
   */
  private renderSnapshotBorder(
    sctx: CanvasRenderingContext2D,
    from: ViewTransform,
    snapW: number,
    snapH: number,
    level: number,
    innerHalfW = 0,
    innerHalfH = 0,
  ): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    const radius = meta.base_radius / Math.pow(2, level);
    const screenRadius = radius * from.zoom;
    const geom = this.geom;
    const tileSpan = meta.tile_span;
    const tileW = tileSpan * geom.dx(radius);
    const tileH = tileSpan * geom.dy(radius);

    const halfWx = snapW / 2 / from.zoom;
    const halfWy = snapH / 2 / from.zoom;
    const txMin = Math.floor((from.centerX - halfWx) / tileW - 1);
    const txMax = Math.ceil((from.centerX + halfWx) / tileW + 1);
    const tyMin = Math.floor((from.centerY - halfWy) / tileH - 1);
    const tyMax = Math.ceil((from.centerY + halfWy) / tileH + 1);

    const cmap = resolveColormap(this.colormap(), this.effectiveTheme());
    this.selAccent = this.themeColor('--accent') || '#4f9dff';
    const selectionActive = this.selection.size > 0;
    const cull = screenRadius * 2;
    // The frozen viewport copy is dropped into the centre afterwards, so a cell
    // whose whole silhouette lands inside that centred rect is overwritten anyway
    // — skip it and paint only the overscan ring. innerHalf* = 0 (the zoom-out
    // caller) disables the skip and draws the full buffer, as before.
    const cx = snapW / 2;
    const cy = snapH / 2;

    for (let tx = txMin; tx <= txMax; tx++) {
      for (let ty = tyMin; ty <= tyMax; ty++) {
        const cached = this.tileCache.getCached(level, tx, ty);
        if (!cached) continue;
        for (const cell of cached.cells) {
          const bcx = (cell.cx - from.centerX) * from.zoom + cx;
          const bcy = (cell.cy - from.centerY) * from.zoom + cy;
          if (bcx < -cull || bcx > snapW + cull) continue;
          if (bcy < -cull || bcy > snapH + cull) continue;
          if (
            innerHalfW > 0 &&
            bcx - screenRadius >= cx - innerHalfW &&
            bcx + screenRadius <= cx + innerHalfW &&
            bcy - screenRadius >= cy - innerHalfH &&
            bcy + screenRadius <= cy + innerHalfH
          ) {
            continue;
          }
          this.drawHex(sctx, bcx, bcy, screenRadius, cell, cmap, selectionActive);
        }
      }
    }
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
    // The snapshot is `snapW×snapH` CSS px centred on `from.center` (snapW/snapH
    // equal the viewport unless a zoom-out overscanned them). Place that centre
    // where the interpolated transform puts `from.center`, then size the blit by
    // the snapshot's own footprint so the overscanned ring lands outside the
    // viewport edges. With no overscan (snapW=width) this is the old identity.
    const cxScreen = this.width / 2 + (from.centerX - cux) * zu;
    const cyScreen = this.height / 2 + (from.centerY - cuy) * zu;
    const w = scale * this.snapW;
    const h = scale * this.snapH;
    return { x: cxScreen - w / 2, y: cyScreen - h / 2, w, h };
  }

  /** One frame of the zoom transition: clear, blit the scaled snapshot, repeat
   *  until the duration elapses, then paint the real rebinned frame. */
  private readonly stepZoomAnim = (now: number): void => {
    const ctx = this.ctx;
    const snap = this.animSnapshot;
    if (!this.animActive || !snap || !ctx) return;

    const frameStart = performance.now();
    const t = Math.min(1, Math.max(0, (now - this.animStartTs) / BrowseCanvasComponent.ZOOM_ANIM_MS));
    const e = 1 - Math.pow(1 - t, 3); // easeOutCubic: quick out, settle into the rebin
    const rect = this.zoomBlitRect(e);

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.animBg;
    ctx.fillRect(0, 0, this.width, this.height);
    // Bilinear filtering of a full-canvas scaled blit is the single most
    // expensive part of a software-rasterized frame; low-effects mode drops to
    // nearest-neighbour, whose brief shimmer is invisible mid-motion. draw()
    // restores smoothing for the crisp landed frame.
    ctx.imageSmoothingEnabled = !this.lowFx;
    ctx.drawImage(snap, 0, 0, snap.width, snap.height, rect.x, rect.y, rect.w, rect.h);
    this.recordFrameCost(frameStart);

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

  /**
   * If the view sits past its hard bounds (rubber-band overshoot), ease it back
   * to the clamped position; otherwise do nothing. Cancels any pending debounced
   * settle. Called when a pan ends and (debounced) once wheel zoom goes quiet.
   */
  private settleToBounds(): void {
    if (this.settleTimer) {
      clearTimeout(this.settleTimer);
      this.settleTimer = null;
    }
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    if (this.width <= 0 || this.height <= 0) return;
    // A zoom transition or pan glide still owns the canvas: both its rAF loop and
    // the settle's write `displayedTransform` and repaint, so running them at once
    // truncates the transition (visible jank). Re-arm the debounce and let the
    // animation land first; the next tick finds the canvas free and springs back.
    if (this.animActive || this.panAnimActive) {
      this.scheduleSettle();
      return;
    }
    const dest = this.clampedTransform(this.transform);
    const cur = this.transform;
    const settled =
      Math.abs(dest.centerX - cur.centerX) < 1e-3 &&
      Math.abs(dest.centerY - cur.centerY) < 1e-3 &&
      Math.abs(dest.zoom - cur.zoom) <= 1e-4 * cur.zoom;
    if (settled) return;
    // Honour reduced-motion: jump straight to the clamp instead of springing.
    if (prefersReducedMotion()) {
      this.transform = dest;
      this.updateActiveLevel();
      this.requestRedraw();
      this.refreshHoverAfterZoom();
      return;
    }
    this.settleFrom = { ...cur };
    this.settleTo = dest;
    this.settleStartTs = performance.now();
    if (this.settleRafId) cancelAnimationFrame(this.settleRafId);
    this.settleActive = true;
    this.ngZone.runOutsideAngular(() => {
      this.settleRafId = requestAnimationFrame(this.stepSettle);
    });
  }

  /** One frame of the boundary snap-back: interpolate the live transform from
   *  the overshoot toward the clamp (easeOutCubic, geometric on zoom) and
   *  repaint, landing exactly on the clamped transform. */
  private readonly stepSettle = (now: number): void => {
    if (!this.settleActive) return;
    // Clamp to [0, 1]; the lower clamp guards the same frame-start vs. `performance.now()`
    // timing skew described in {@link stepPanAnim} (here the rAF is scheduled from the
    // mouseup/wheel-quiet handler), which would otherwise overshoot the snap-back backwards
    // on the first frame. Matches the zoom transition's guard.
    const t = Math.min(1, Math.max(0, (now - this.settleStartTs) / BrowseCanvasComponent.SETTLE_MS));
    const e = 1 - Math.pow(1 - t, 3);
    const from = this.settleFrom;
    const to = this.settleTo;
    this.transform.centerX = from.centerX + (to.centerX - from.centerX) * e;
    this.transform.centerY = from.centerY + (to.centerY - from.centerY) * e;
    this.transform.zoom = from.zoom * Math.pow(to.zoom / from.zoom, e);
    this.updateActiveLevel();
    this.draw();
    if (t < 1) {
      this.settleRafId = requestAnimationFrame(this.stepSettle);
    } else {
      this.settleActive = false;
      this.settleRafId = 0;
      this.transform.centerX = to.centerX;
      this.transform.centerY = to.centerY;
      this.transform.zoom = to.zoom;
      this.updateActiveLevel();
      this.draw();
      this.refreshHoverAfterZoom();
    }
  };

  /** Stop a snap-back in flight (without snapping to the clamp) and drop any
   *  pending debounced settle — used when a new gesture takes over, so it
   *  continues from wherever the view visually is. */
  private cancelSettle(): void {
    if (this.settleTimer) {
      clearTimeout(this.settleTimer);
      this.settleTimer = null;
    }
    if (!this.settleActive) return;
    this.settleActive = false;
    if (this.settleRafId) {
      cancelAnimationFrame(this.settleRafId);
      this.settleRafId = 0;
    }
  }

  /** Debounced settle for wheel zoom, which has no discrete "end": fires once
   *  the wheel has been quiet for a beat. */
  private scheduleSettle(): void {
    if (this.settleTimer) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      this.settleToBounds();
    }, 90);
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx || this.width === 0) return;

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fillRect(0, 0, this.width, this.height);

    const meta = this.meta();
    if (!meta || meta.point_count === 0) {
      ctx.fillStyle = this.themeColor('--text-muted');
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Nothing to show yet', this.width / 2, this.height / 2);
      return;
    }

    // Time the real frame paints (post-guard, so trivial empty frames don't
    // drag the average down) to latch low-effects mode on machines the WebGL
    // probe couldn't identify — see `render-perf.ts` (issue #2695).
    const frameStart = performance.now();
    // The low-effects animation blits switch smoothing off; landed frames are
    // always painted crisp, so restore it here.
    ctx.imageSmoothingEnabled = true;

    const level = this.activeLevel;
    const radius = meta.base_radius / Math.pow(2, level);
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
      this.densityMaxChanged.emit(this.maxCount);
    }

    // Resolve the colormap against the live theme once per frame, not per cell.
    const cmap = resolveColormap(this.colormap(), this.effectiveTheme());
    // Accent for selection rings + marquee, also resolved once per frame.
    this.selAccent = this.themeColor('--accent') || '#4f9dff';
    // Waveform tint colours (audio only), likewise resolved once per frame; the
    // tinted-mask cache is rebuilt against these on a theme flip.
    this.waveformTint = this.mediaType() === 'audio';
    if (this.waveformTint) {
      this.waveAccent = this.themeColor('--accent') || '#4f9dff';
      this.waveSurface = this.themeColor('--bg-surface') || '#1a1d27';
    }
    this.imageTiles = this.mediaType() === 'image';
    const selectionActive = this.selection.size > 0;

    // The enlarged cell is deferred and redrawn last (on top of its neighbours)
    // so the read-out is a size bump rather than a border — leaving the border
    // free to encode selection state. A pinned cell (its detail popup is open)
    // wins over the live hover, so the bin whose details are showing stays
    // enlarged even as the cursor moves off it. Matched by (level, q, r) via
    // `sameBin`: the pin outlives a wheel zoom, and axial coords alone would
    // match some unrelated finer cell once the zoom crosses a level.
    const enlargedCell = this.pinnedCell ?? this.hoveredCell;
    const enlargedLevel = this.pinnedCell ? this.pinnedLevel : this.hoveredLevel;
    let hovered: { cell: HexCellPayload; sx: number; sy: number } | null = null;
    for (const cell of allCells) {
      const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
      if (sx < -screenRadius * 2 || sx > this.width + screenRadius * 2) continue;
      if (sy < -screenRadius * 2 || sy > this.height + screenRadius * 2) continue;
      if (sameBin(enlargedLevel, enlargedCell, level, cell)) {
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

    // Region signposts letter the map above the bins (and the hover-enlarged
    // cell) but below the transient marquee, like place names on a map layer.
    if (this.signposts() && this.labels().length > 0) this.drawSigns(ctx);

    if (this.marquee) this.drawMarquee(ctx);

    // Publish the region now on screen so the minimap can draw its viewport box.
    this.viewport.setViewport(this.getVisibleBounds());

    this.updatePanDirection();
    this.prefetchNeighbors(visibleTiles);

    // Record what's now on screen so a zoom transition can grow/shrink this
    // exact frame the next time a zoom crosses a level boundary.
    this.displayedTransform = {
      centerX: this.transform.centerX,
      centerY: this.transform.centerY,
      zoom: this.transform.zoom,
    };
    this.displayedLevel = level;
    this.hasDrawn = true;

    this.recordFrameCost(frameStart);
    this.maybeReportFirstView();
  }

  /**
   * Emit {@link firstViewReady} once the opening view is fully painted with real
   * content. Called at the end of every {@link draw} until it fires. It holds
   * until three facts are true: the fit ran against the *real* canvas size (the
   * 800×600 fallback fit paints a framing {@link resize} immediately refits
   * away), every tile under the viewport has loaded, and — for thumbnail media —
   * every on-screen representative thumbnail has decoded or failed. Each pending
   * tile / thumbnail lands with its own redraw, so a later draw re-runs this
   * check until the view is whole; a one-shot timer, armed the first time we
   * start waiting on thumbnails, releases the cover if an image hangs. Pure
   * density media (audio/text) paint no thumbnails, so it fires as soon as the
   * visible tiles are in.
   */
  private maybeReportFirstView(): void {
    if (this.firstViewReported) return;
    // Hold until the fit used the real canvas size: the fallback fit paints
    // tiles/thumbnails for a framing the refit discards, so revealing then would
    // flash the wrong view.
    if (!this.fittedAgainstRealSize) return;
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;

    // Arm the backstop the first time we begin waiting: a tile or thumbnail that
    // hard-fails to load caches nothing and fires no redraw, so without this the
    // cover could strand. The timer releases it regardless (falling back to the
    // old show-then-fill behaviour), and reportFirstView clears it on success.
    if (this.firstViewTimer === null) {
      this.firstViewTimer = setTimeout(
        () => this.reportFirstView(),
        BrowseCanvasComponent.FIRST_VIEW_MAX_WAIT_MS,
      );
    }

    const thumbs = this.thumbnailMode;
    const level = this.activeLevel;
    const radius = meta.base_radius / Math.pow(2, level);
    const screenRadius = radius * this.effZoom;

    for (const { tx, ty } of this.getVisibleTiles()) {
      const tile = this.tileCache.getCached(level, tx, ty);
      // Geometry still loading: a tileLoaded$ redraw re-runs this check.
      if (!tile) return;
      if (!thumbs) continue;
      for (const cell of tile.cells) {
        // Only cells actually on screen gate the reveal — the visible-tile set
        // carries a one-tile margin, so mirror draw()'s per-cell cull exactly.
        const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
        if (sx < -screenRadius * 2 || sx > this.width + screenRadius * 2) continue;
        if (sy < -screenRadius * 2 || sy > this.height + screenRadius * 2) continue;
        if (this.thumbFailed.has(cell.rep_id)) continue;
        // draw() already kicked off the load for every drawn cell; here we only
        // read whether it has decoded. A missing / still-decoding image holds
        // the reveal (its onload fires a redraw that re-checks).
        const img = this.thumbCache.get(cell.rep_id);
        if (!img || !img.complete || img.naturalWidth === 0) return;
      }
    }
    this.reportFirstView();
  }

  /** Fire {@link firstViewReady} once and cancel the backstop timer. */
  private reportFirstView(): void {
    if (this.firstViewReported) return;
    this.firstViewReported = true;
    if (this.firstViewTimer !== null) {
      clearTimeout(this.firstViewTimer);
      this.firstViewTimer = null;
    }
    this.firstViewReady.emit();
  }

  /** Selection state of a cell: 0 = none, 1 = partial, 2 = full. Memoized on
   *  the cell against the selection version so a steady-state pan doesn't
   *  re-scan every bin's members each frame. */
  private selStateFor(cell: HexCellPayload): 0 | 1 | 2 {
    const memo = cell as HexCellPayload & { _selVer?: number; _selState?: 0 | 1 | 2 };
    if (memo._selVer === this.selection.version() && memo._selState !== undefined) {
      return memo._selState;
    }
    const members = this.cellMembers(cell);
    const sel = this.selection.selectedCountIn(members);
    const state: 0 | 1 | 2 = sel === 0 ? 0 : sel === members.length ? 2 : 1;
    memo._selVer = this.selection.version();
    memo._selState = state;
    return state;
  }

  /**
   * Paint the region signposts: theme-aware translucent pills lettered with the
   * region names whose `level` sits near the current zoom. All the geometry —
   * which signs are visible, how big, how faded, and the greedy de-clutter —
   * lives in the framework-free `sign-layout.ts`; this method only measures
   * text and paints what the layout kept. The view level is *continuous* (the
   * unrounded form of the LOD picker's level), so signs grow/fade smoothly
   * through a zoom rather than stepping at level boundaries.
   *
   * Signs the layout marked `approximate` — the coarse bands whose ground-truth
   * purity is measurably weak (issue #3346) — letter in italic with a `~`
   * prefix, at slightly lower opacity and with no drop shadow, so they read as
   * "roughly this way" rather than as a name.
   */
  private drawSigns(ctx: CanvasRenderingContext2D): void {
    const meta = this.meta();
    if (!meta) return;
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const placed = layoutSigns(
      this.labels(),
      {
        transform: this.transform,
        width: this.width,
        height: this.height,
        viewLevel: viewLevelForZoom(meta.base_radius, this.effZoom, this.targetRadius),
      },
      (text, fontPx, approximate) => {
        ctx.font = this.signFont(fontPx, approximate);
        return ctx.measureText(text).width;
      },
    );
    const pillBg = this.themeColor('--bg-body') || '#0f1117';
    const textColor = this.themeColor('--text-primary') || '#e0e0e0';
    for (const sign of placed) {
      // The pill sits behind the text at a lower opacity so the sign reads as
      // lettering over the map, not an opaque card punched through it. A drop
      // shadow under the pill floats the sign off the map toward the viewer as
      // the zoom grows past it: the smallest signs are flat (no shadow), and
      // larger (more zoomed-past) ones cast a progressively bigger, softer,
      // more-offset shadow — see `signShadow`. Shadow blurs are among the most
      // expensive canvas ops under software rasterization, and the signs repaint
      // on every pan frame — low-effects mode paints them flat (issue #2695).
      const shadow = this.lowFx ? null : signShadow(sign.scale, sign.fontPx, sign.approximate);
      ctx.globalAlpha = sign.alpha * 0.6;
      ctx.fillStyle = pillBg;
      if (shadow) {
        ctx.shadowColor = `rgba(0, 0, 0, ${shadow.alpha})`;
        ctx.shadowBlur = shadow.blur;
        ctx.shadowOffsetY = shadow.offsetY;
      }
      ctx.beginPath();
      ctx.roundRect(sign.sx - sign.w / 2, sign.sy - sign.h / 2, sign.w, sign.h, sign.h / 2);
      ctx.fill();
      // Clear the shadow before the text so the lettering isn't double-shadowed
      // and the next sign starts clean.
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      ctx.globalAlpha = sign.alpha;
      ctx.fillStyle = textColor;
      ctx.font = this.signFont(sign.fontPx, sign.approximate);
      ctx.fillText(sign.text, sign.sx, sign.sy);
    }
    ctx.restore();
  }

  /** Canvas font spec for a sign at `fontPx`. Semibold so the lettering holds
   *  up over busy density fills without needing an outline; italic when the sign
   *  names one of the weak coarse bands, which is the hedge's quietest half (the
   *  `~` prefix and the missing shadow carry it at small sizes). Both the
   *  measure pass and the paint pass go through here so a pill is never sized in
   *  a face it isn't drawn in. */
  private signFont(fontPx: number, approximate = false): string {
    return `${approximate ? 'italic ' : ''}600 ${fontPx}px ${SIGN_FONT_FAMILY}`;
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
    trim: { hw: number; hh: number } | null = null,
  ): void {
    // A multi-item ("pile") cell is drawn with a soft rounded shape — a disc in
    // hex mode, a rounded-corner rectangle in square mode — while a one-item cell
    // keeps its full sharp hexagon / square so a lone item reads as a crisp,
    // distinct tile. The hovered cell instead passes `trim`: a rectangle of the
    // thumbnail's native aspect ratio, so the enlarged tile breaks out of the bin
    // silhouette and shows the whole frame (no hex/square clip, no background
    // bars). A pile's break-out keeps rounded corners (see `traceTrimRect`) so it
    // still reads as a pile; a singleton's stays sharp.
    const single = cell.count === 1;
    const rounded = !single;
    if (trim) {
      this.traceTrimRect(ctx, cx, cy, trim, rounded, radius);
    } else {
      this.geom.traceCell(ctx, cx, cy, radius, rounded);
    }

    // Image / video: paint the central item's thumbnail clipped to the cell.
    // Until it loads, fall back to the density shading below so the cell is
    // never blank. Video/audio grid cells cover-fit (fill the bin, cropping the
    // edges); image cells instead use a balanced "half crop, half pad" fit
    // (`drawImageFit(..., 'balanced')`) that leaves a background-coloured gap
    // inside the bin's border. The hovered cell (either type) instead draws the
    // whole thumbnail to fill `trim`, whose rectangle already carries the
    // image's aspect ratio, so it shows undistorted and uncropped.
    const thumb = this.thumbnailMode ? this.getThumb(cell.rep_id) : null;
    if (thumb) {
      ctx.save();
      ctx.clip();
      if (this.waveformTint) {
        // Audio: the thumbnail is a theme-agnostic alpha mask (issue #2369).
        // Fill the cell with the themed surface, then paint the wave tinted to
        // the accent colour — so the tile matches dark / light / highviz and
        // recolours on a theme flip instead of showing baked-in pixels.
        ctx.fillStyle = this.waveSurface;
        ctx.fill();
        const tinted = this.getTintedThumb(cell.rep_id, thumb);
        if (trim) {
          ctx.drawImage(tinted, cx - trim.hw, cy - trim.hh, trim.hw * 2, trim.hh * 2);
        } else {
          this.drawImageFit(ctx, tinted, cx, cy, radius);
        }
      } else if (trim) {
        ctx.drawImage(thumb, cx - trim.hw, cy - trim.hh, trim.hw * 2, trim.hh * 2);
      } else if (this.imageTiles) {
        // Image grid cell: "half crop, half pad". Fill the bin with the body
        // background first so the padded gap reads as background between the
        // image content and the bin's border, then draw the image at the
        // balanced (geometric-mean) scale — cropped less than cover would be and
        // padded less than contain would be, with equal crop and pad fractions.
        ctx.fillStyle = this.themeColor('--bg-body');
        ctx.fill();
        this.drawImageFit(ctx, thumb, cx, cy, radius, 'balanced');
      } else {
        this.drawImageFit(ctx, thumb, cx, cy, radius);
      }
      ctx.restore();
    } else if (single) {
      // Singletons get the colormap's dedicated one-item colour, decoupled
      // from the density ramp so a lone item reads as "exactly one".
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
    } else if (thumb && !single && this.thumbnailBorder() > 0) {
      // Pile thumbnail: a band whose colormap colour encodes how many items are
      // stacked under this tile. Clipped to the cell so the full width sits just
      // inside the thumbnail edge rather than bleeding onto neighbours (a
      // centred stroke would spill half its width outward).
      const t = Math.log(cell.count) / Math.log(this.maxCount || 2);
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = densityColor(Math.max(0, Math.min(1, t)), cmap.ramp);
      ctx.lineWidth = this.thumbnailBorder() * 2;
      ctx.stroke();
      ctx.restore();
    } else if (thumb && single && this.thumbnailBorder() > 0) {
      // Singleton thumbnail: a border in the colormap's dedicated one-item
      // colour (`cmap.single`, the "1" end of the bin-size scale that the pile
      // ramp never reaches), so a lone item stands out against the background
      // and reads as "exactly one". Because the singleton's cell was traced
      // sharp-cornered (`rounded = false`), stroking that same path yields a
      // hard-edged rectangle — deliberately unlike the pile band's rounded
      // silhouette, so the sharp corners plus the single colour mark it apart.
      // Same clip + full-inset width as the pile band.
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = rgbString(cmap.single);
      ctx.lineWidth = this.thumbnailBorder() * 2;
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
    // A hovered thumbnail breaks out of its bin: it's shown whole at its native
    // aspect ratio as a rectangle (no silhouette), sized to grow until its edge
    // just reaches the nearest neighbour cell's centre (`hoverThumbRect`). A
    // pile's rectangle keeps rounded corners so it stays distinguishable from a
    // singleton (`traceTrimRect`). A non-thumbnail (flat density) cell has no
    // such rectangle, so it keeps its silhouette and simply lifts off with a
    // fixed size bump.
    const thumb = this.thumbnailMode ? this.getThumb(cell.rep_id) : null;
    const trim = thumb ? this.hoverThumbRect(thumb, radius) : null;
    const bumped = radius * HOVER_RADIUS_SCALE;

    // Cast a single clean drop shadow from an opaque base shape first, then
    // paint the real (shadow-free) cell on top so the fill/border don't each
    // stack their own shadow.
    ctx.save();
    // Shadow blurs rasterize dearly without a GPU; low-effects mode keeps the
    // enlarge (the actual hover signal) and skips the cosmetic lift shadow.
    if (!this.lowFx) {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = Math.max(4, radius * 0.3);
      ctx.shadowOffsetY = Math.max(1, radius * 0.1);
    }
    if (trim) {
      this.traceTrimRect(ctx, cx, cy, trim, cell.count > 1, radius);
    } else {
      this.geom.traceCell(ctx, cx, cy, bumped, cell.count > 1);
    }
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fill();
    ctx.restore();

    // `radius` is unused for the shape when `trim` is set (the rectangle drives
    // it), so the un-bumped radius is fine there; flat-density cells use the bump.
    this.drawHex(ctx, cx, cy, trim ? radius : bumped, cell, cmap, selectionActive, trim);
  }

  /**
   * Trace the hovered break-out thumbnail's rectangle as the current path,
   * centred on `(cx, cy)` with half-extents `trim`. A pile (`rounded`) rounds
   * its corners with the *same absolute* radius its grid disc / rounded square
   * curved with (`geom.roundedCornerRadius`, computed from the un-bumped bin
   * `radius`) — a fixed "total" round, not one proportional to the enlarged
   * rectangle, so the blown-up tile keeps the pile's corner curvature and only
   * nibbles the image corners. Singletons stay sharp-cornered; their one-item
   * colour is what marks them. The radius is clamped so it never exceeds half a
   * side of a narrow break-out rectangle.
   */
  private traceTrimRect(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    trim: { hw: number; hh: number },
    rounded: boolean,
    radius: number,
  ): void {
    ctx.beginPath();
    if (rounded) {
      const r = Math.min(this.geom.roundedCornerRadius(radius), trim.hw, trim.hh);
      ctx.roundRect(cx - trim.hw, cy - trim.hh, trim.hw * 2, trim.hh * 2, r);
    } else {
      ctx.rect(cx - trim.hw, cy - trim.hh, trim.hw * 2, trim.hh * 2);
    }
    ctx.closePath();
  }

  /** Draw an image centred over the hex's 2*radius square (the path must be
   *  clipped). ``fit`` picks the scale:
   *  - ``'cover'`` (default): fill the square, cropping whichever axis overflows
   *    — the historical behaviour, still used for video/audio tiles.
   *  - ``'balanced'``: the geometric mean of the cover and contain scales, so
   *    the fraction cropped off the long axis equals the fraction of background
   *    gap left on the short axis — the "half crop, half pad" image fit. The
   *    caller fills the cell with the body background first so that gap reads as
   *    background rather than the tile underneath.
   *
   *  Accepts a tinted waveform canvas (issue #2369) as well as a raw image; a
   *  canvas exposes its intrinsic size as ``width``/``height`` rather than
   *  ``naturalWidth``/``naturalHeight``. */
  private drawImageFit(
    ctx: CanvasRenderingContext2D,
    img: HTMLImageElement | HTMLCanvasElement,
    cx: number,
    cy: number,
    radius: number,
    fit: TileFit = 'cover',
  ): void {
    const iw = img instanceof HTMLCanvasElement ? img.width : img.naturalWidth;
    const ih = img instanceof HTMLCanvasElement ? img.height : img.naturalHeight;
    const { dw, dh } = imageTileFitDimensions(iw, ih, radius, fit);
    ctx.drawImage(img, cx - dw / 2, cy - dh / 2, dw, dh);
  }

  /**
   * Half-extents of a hovered thumbnail. On hover the thumbnail breaks out of
   * its bin and is shown whole at its native aspect ratio, grown as large as
   * possible under one rule: no neighbouring cell's centre may be covered. The
   * rectangle (aspect = image width / height) is centred on the cell, so a
   * neighbour at screen offset `(ox, oy)` stays uncovered while the half-height
   * `H` satisfies `H ≤ max(|ox| / aspect, |oy|)`; the binding neighbour is the
   * tightest of those (four for a square, six for a hex), and the rectangle's
   * edge then just touches that centre. Wide images grow until they reach the
   * side neighbours, tall ones until they reach the top/bottom — each axis
   * limited by whichever neighbour it would otherwise overrun.
   */
  private hoverThumbRect(img: HTMLImageElement, radius: number): { hw: number; hh: number } {
    const aspect = img.naturalWidth / img.naturalHeight;
    return hoverThumbHalfExtents(aspect, radius, this.geom.neighborOffsets());
  }

  /**
   * Return the loaded thumbnail for a representative media id, or null while it
   * loads / if it failed. Kicks off the fetch on first request and redraws when
   * the image arrives.
   */
  private getThumb(representativeId: number): HTMLImageElement | null {
    const cached = this.thumbCache.get(representativeId);
    if (cached) {
      // Bump recency: re-insert so a thumbnail painted this frame becomes the
      // newest entry. The cache is insertion-ordered (see {@link evictThumbs}),
      // so this keeps currently-visible thumbnails last in line for eviction —
      // off-view preloads (which never bump) are dropped first, so warming the
      // ring can never evict something that's on screen.
      this.thumbCache.delete(representativeId);
      this.thumbCache.set(representativeId, cached);
      return cached.complete && cached.naturalWidth > 0 ? cached : null;
    }
    this.startThumbLoad(representativeId, false);
    return null;
  }

  /**
   * Kick off the thumbnail fetch for a representative id and stash the pending
   * image in the cache. ``preload`` marks an off-view warm-up: it loads at low
   * network priority and does not repaint when it lands (the cell isn't on
   * screen), so it never competes with visible thumbnails. A no-op when the id
   * is already cached or known-failed.
   */
  private startThumbLoad(representativeId: number, preload: boolean): void {
    if (this.thumbCache.has(representativeId) || this.thumbFailed.has(representativeId)) return;

    if (this.thumbCache.size >= this.MAX_THUMBS) this.evictThumbs();

    const img = new Image();
    img.decoding = 'async';
    if (preload) {
      // Idle warm-up: let the browser schedule it behind visible-thumbnail and
      // tile requests, and skip the repaint — the cell is off-screen, so a later
      // draw picks it up from the cache once the user pans to it.
      img.setAttribute('fetchpriority', 'low');
    } else {
      img.onload = () => this.requestRedraw();
    }
    img.onerror = () => {
      this.thumbCache.delete(representativeId);
      this.thumbFailed.add(representativeId);
    };
    // Downscaled /thumbnail by default: a browse projection can hold thousands
    // of points, so painting full-size bitmaps onto every hex would exhaust
    // memory. The /thumbnail route serves the frame for video via the same
    // image_response hook, then downscales it. At the largest zoom levels,
    // though, a cell is drawn wider than the thumbnail's native resolution, so
    // we fetch the full-res /image instead (only a few such giant cells fit on
    // screen, so memory stays bounded). See {@link useFullResThumbs}.
    const endpoint = this.thumbsAreFullRes ? 'image' : 'thumbnail';
    img.src = this.activeContext.mediaUrl(`/api/medias/${representativeId}/${endpoint}`);
    this.thumbCache.set(representativeId, img);
  }

  /** Drop the oldest quarter of cached thumbnails (insertion-ordered LRU). */
  private evictThumbs(): void {
    const target = Math.floor(this.MAX_THUMBS * 0.75);
    const toRemove = this.thumbCache.size - target;
    let i = 0;
    for (const key of this.thumbCache.keys()) {
      if (i++ >= toRemove) break;
      this.thumbCache.delete(key);
      // Drop the matching tinted mask (issue #2369) so it can't outlive its
      // raw source and leak; it re-tints on demand if the clip is revisited.
      this.tintedThumbCache.delete(key);
    }
  }

  /**
   * Return the audio waveform thumbnail tinted to the live theme's accent
   * colour, built once per (clip, theme) and cached (issue #2369).
   *
   * The raw thumbnail is a theme-agnostic alpha mask — a transparent PNG whose
   * only opaque pixels are the wave. Painting it to an offscreen canvas and
   * compositing the accent through ``source-in`` recolours just the wave,
   * leaving the background transparent so the tile's themed surface (filled by
   * the caller) shows through. The tinted cache is cleared on a theme flip and
   * whenever the raw {@link thumbCache} is, so it never serves a stale colour.
   */
  private getTintedThumb(representativeId: number, src: HTMLImageElement): HTMLCanvasElement {
    const cached = this.tintedThumbCache.get(representativeId);
    if (cached) return cached;

    const w = src.naturalWidth || 1;
    const h = src.naturalHeight || 1;
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const octx = off.getContext('2d');
    if (octx) {
      octx.drawImage(src, 0, 0);
      octx.globalCompositeOperation = 'source-in';
      octx.fillStyle = this.waveAccent;
      octx.fillRect(0, 0, w, h);
    }
    this.tintedThumbCache.set(representativeId, off);
    return off;
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
    const meta = this.meta();
    if (!meta) return;
    const level = this.activeLevel;
    // Warm the geometry of the ring just beyond the drawn tiles (8-connected, so
    // diagonals are covered, plus an extra tile in the direction of travel) so a
    // pan into it never blanks while the tile fetch is in flight.
    for (const { tx, ty } of this.offViewRing(visibleTiles)) {
      this.tileCache.prefetch(level, tx, ty);
    }
    if (level > 0) {
      this.prefetchLevel(level - 1, visibleTiles);
    }
    if (meta.levels.length > level + 1) {
      this.prefetchLevel(level + 1, visibleTiles);
    }
    // Image/video datasets paint a thumbnail per cell; warm the off-view ring's
    // thumbnails (a pan) and the finer level's centre thumbnails (a zoom-in) on
    // idle so they're decoded before the cells appear. Pure-density media
    // (audio/text) draw no thumbnails, so there's nothing to warm. Warming follows
    // the same resolution tier (capped /thumbnail vs full-res /image) and the same
    // count-bounded LRU as on-demand loads, so it can't raise the memory ceiling —
    // it only reaches it sooner, and visible thumbnails (which bump recency) are
    // evicted last. See {@link runThumbPrefetch} for the pan/zoom budget split.
    if (this.thumbnailMode) this.scheduleThumbPrefetch();
  }

  /**
   * The ring of tiles just outside the drawn set: the bounding box of
   * ``visibleTiles`` grown by one tile on every side (8-connected, so diagonals
   * are included), minus the box itself. The growth is extended by a second tile
   * on whichever sides the view is panning toward ({@link panDirX} /
   * {@link panDirY}), so a sustained pan warms further ahead in the direction of
   * travel. ``visibleTiles`` already carries a one-tile margin (see
   * {@link getVisibleTiles}), so this ring sits one to two tiles beyond what's
   * actually painted.
   */
  private offViewRing(visibleTiles: { tx: number; ty: number }[]): { tx: number; ty: number }[] {
    if (visibleTiles.length === 0) return [];
    let txMin = Infinity;
    let txMax = -Infinity;
    let tyMin = Infinity;
    let tyMax = -Infinity;
    for (const { tx, ty } of visibleTiles) {
      if (tx < txMin) txMin = tx;
      if (tx > txMax) txMax = tx;
      if (ty < tyMin) tyMin = ty;
      if (ty > tyMax) tyMax = ty;
    }
    // One ring on every side, plus a directional extra tile on the leading edges.
    const DIR = 0.3;
    const outTxMin = txMin - (this.panDirX < -DIR ? 2 : 1);
    const outTxMax = txMax + (this.panDirX > DIR ? 2 : 1);
    const outTyMin = tyMin - (this.panDirY < -DIR ? 2 : 1);
    const outTyMax = tyMax + (this.panDirY > DIR ? 2 : 1);
    const ring: { tx: number; ty: number }[] = [];
    for (let tx = outTxMin; tx <= outTxMax; tx++) {
      for (let ty = outTyMin; ty <= outTyMax; ty++) {
        // Keep only the new border; the interior is the already-drawn box.
        if (tx >= txMin && tx <= txMax && ty >= tyMin && ty <= tyMax) continue;
        ring.push({ tx, ty });
      }
    }
    return ring;
  }

  /**
   * Queue a single low-priority pass that warms off-view thumbnails when the main
   * thread is idle. At most one pass is in flight; it recomputes the ring from the
   * live view when it runs, so a pan that continued after scheduling still warms
   * the right tiles. The ``timeout`` guarantees it runs even under a steady pan
   * (where the thread rarely goes fully idle), keeping the warm-up ahead of the
   * motion rather than only after it stops.
   */
  private scheduleThumbPrefetch(): void {
    if (this.destroyed) return;
    if (this.thumbPrefetchHandle !== null) return;
    const run = () => {
      this.thumbPrefetchHandle = null;
      this.runThumbPrefetch();
    };
    if (typeof requestIdleCallback === 'function') {
      this.thumbPrefetchIsTimeout = false;
      this.thumbPrefetchHandle = requestIdleCallback(run, { timeout: 500 });
    } else {
      this.thumbPrefetchIsTimeout = true;
      this.thumbPrefetchHandle = window.setTimeout(run, 100);
    }
  }

  /**
   * Warm off-view thumbnails on idle, bounded to {@link PRELOAD_MAX_PER_PASS}
   * new fetches so one pass can't flood the network or the cache. The budget is
   * split across the two ways the visible set changes:
   *
   * - **Pan** — the off-view ring at the current level (cells a pan scrolls in).
   * - **Zoom-in** — the finer cells now centred under the view (cells a zoom-in
   *   reveals). A zoom is no more surprising than a pan, but it swaps the whole
   *   screen for a denser level, so warming only the pan ring leaves a zoom-in
   *   facing a cold cache.
   *
   * Pan warms first up to its share; the zoom path takes the reserve plus any
   * budget pan left unused, and a final pan sweep mops up anything the zoom path
   * didn't need (e.g. no finer level, or its tiles already warm), so the split
   * only bites when both genuinely compete. Both paths share the count-bounded
   * LRU keyed by media id, so a thumbnail warmed for a zoom is reused if a pan
   * reaches it instead. Only tiles whose geometry is already cached are walked
   * (one still loading is picked up by a later pass), and warming stops short of
   * the cache cap so a preload never evicts a visible thumbnail.
   */
  private runThumbPrefetch(): void {
    const meta = this.meta();
    if (!this.thumbnailMode || !meta) return;
    const level = this.activeLevel;
    const total = BrowseCanvasComponent.PRELOAD_MAX_PER_PASS;
    const panRing = this.offViewRing(this.getVisibleTiles());
    // Only reserve for zoom when there's a finer level to zoom into; otherwise
    // pan gets the whole budget on the first sweep.
    const canZoomIn = meta.levels.length > level + 1;
    const zoomReserve = canZoomIn ? Math.floor(total * BrowseCanvasComponent.THUMB_PRELOAD_ZOOM_SHARE) : 0;

    // Pan: the off-view ring at the current level, capped so the reserve stays
    // for zoom.
    let remaining = total - this.warmThumbsForTiles(level, panRing, total - zoomReserve);
    // Zoom-in: the finer tiles centred under the view, newest-revealed first.
    if (canZoomIn && remaining > 0) {
      remaining -= this.warmThumbsForTiles(level + 1, this.finerTilesForZoom(), remaining);
    }
    // Hand any unspent budget back to the pan ring.
    if (remaining > 0) this.warmThumbsForTiles(level, panRing, remaining);
  }

  /**
   * Kick off idle thumbnail loads for the cells of ``tiles`` at ``level``, up to
   * ``budget`` new fetches, and return how many were started. Tiles whose
   * geometry isn't cached yet are skipped (a later pass catches them), already
   * loaded/failed cells are skipped, and warming stops at the cache cap so a
   * preload never evicts a visible thumbnail.
   */
  private warmThumbsForTiles(
    level: number,
    tiles: { tx: number; ty: number }[],
    budget: number,
  ): number {
    let spent = 0;
    for (const { tx, ty } of tiles) {
      if (spent >= budget) break;
      const tile = this.tileCache.getCached(level, tx, ty);
      if (!tile) continue;
      for (const cell of tile.cells) {
        if (spent >= budget) break;
        if (this.thumbCache.has(cell.rep_id) || this.thumbFailed.has(cell.rep_id)) continue;
        // Stop before the cache fills so a preload never forces an eviction; the
        // free slots come from visible getThumb()s bumping past stale warms.
        if (this.thumbCache.size >= this.MAX_THUMBS) return spent;
        this.startThumbLoad(cell.rep_id, true);
        spent++;
      }
    }
    return spent;
  }

  /**
   * The tiles of the next finer level ({@link activeLevel} + 1) that cover the
   * current viewport: what a zoom-in would render. A finer level halves the bin
   * radius, so each visible tile maps to a 2×2 block of finer tiles; the union is
   * returned sorted by distance from the view centre, since a zoom-in anchors
   * near the centre (or cursor) and so reveals the central tiles first. Mirrors
   * the geometry warmed by {@link prefetchLevel} for ``level + 1``, so the tiles
   * walked here are the ones already being fetched.
   */
  private finerTilesForZoom(): { tx: number; ty: number }[] {
    const meta = this.meta();
    if (!meta) return [];
    const finerLevel = this.activeLevel + 1;
    const radius = meta.base_radius / Math.pow(2, finerLevel);
    const tileW = meta.tile_span * this.geom.dx(radius);
    const tileH = meta.tile_span * this.geom.dy(radius);
    const [vxmin, vymin, vxmax, vymax] = this.getVisibleBounds();
    // View centre in finer-tile coordinates, to rank tiles centre-out.
    const ccx = (vxmin + vxmax) / 2 / tileW;
    const ccy = (vymin + vymax) / 2 / tileH;
    const seen = new Set<string>();
    const tiles: { tx: number; ty: number; d2: number }[] = [];
    for (const { tx, ty } of this.getVisibleTiles()) {
      // Each current-level tile spans a 2×2 block at the finer level.
      for (let dx = 0; dx <= 1; dx++) {
        for (let dy = 0; dy <= 1; dy++) {
          const ftx = tx * 2 + dx;
          const fty = ty * 2 + dy;
          const key = `${ftx}:${fty}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const ex = ftx + 0.5 - ccx;
          const ey = fty + 0.5 - ccy;
          tiles.push({ tx: ftx, ty: fty, d2: ex * ex + ey * ey });
        }
      }
    }
    tiles.sort((a, b) => a.d2 - b.d2);
    return tiles.map(({ tx, ty }) => ({ tx, ty }));
  }

  /**
   * Track a smoothed pan direction from the frame-to-frame change in the view
   * centre, used to bias which off-view tiles to warm first. The exponential
   * smoothing keeps a single jittery frame from flipping the bias and lets the
   * direction decay toward zero when the view holds still (so a stationary view
   * warms its ring symmetrically).
   */
  private updatePanDirection(): void {
    const cx = this.transform.centerX;
    const cy = this.transform.centerY;
    const SMOOTH = 0.4;
    if (!Number.isNaN(this.lastDrawCenterX)) {
      const dx = cx - this.lastDrawCenterX;
      const dy = cy - this.lastDrawCenterY;
      const mag = Math.hypot(dx, dy);
      const ux = mag > 1e-6 ? dx / mag : 0;
      const uy = mag > 1e-6 ? dy / mag : 0;
      this.panDirX += (ux - this.panDirX) * SMOOTH;
      this.panDirY += (uy - this.panDirY) * SMOOTH;
    }
    this.lastDrawCenterX = cx;
    this.lastDrawCenterY = cy;
  }

  private prefetchLevel(targetLevel: number, sourceTiles: { tx: number; ty: number }[]): void {
    const meta = this.meta();
    if (!meta) return;
    const sourceRadius = meta.base_radius / Math.pow(2, this.activeLevel);
    const targetRadius = meta.base_radius / Math.pow(2, targetLevel);
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
    // A new gesture takes over from any boundary snap-back or directional glide:
    // stop it where it is (don't snap) so the pan/marquee continues from the
    // current visual frame.
    this.cancelSettle();
    this.cancelPanAnim();
    // A fresh press settles any single-click toggle still waiting out the
    // double-click window (so quick clicks on different bins each register). The
    // second press of a double-click (detail >= 2) is exempt: flushing there
    // would commit the very toggle the double-click means to drop.
    if (event.detail < 2) this.flushPendingToggle();
    this.panStartX = event.clientX;
    this.panStartY = event.clientY;
    this.dragMoved = false;

    if (event.shiftKey || this.marqueeMode()) {
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
      !this.dragMoved &&
      (Math.abs(dx) > BrowseCanvasComponent.CLICK_MOVE_THRESHOLD ||
        Math.abs(dy) > BrowseCanvasComponent.CLICK_MOVE_THRESHOLD)
    ) {
      this.dragMoved = true;
      this.framedByUser = true;
      // The press turned into a pan: the map is about to slide out from under a
      // stationary cursor while hover stays frozen ({@link onCanvasMouseMove}
      // early-returns for the rest of the drag), so a hover held from before the
      // press would ride along — drawn enlarged at the bin's new position with its
      // preview still open. Drop it for the duration (this also cancels a hover
      // debounce that would otherwise fire mid-pan); {@link onMouseUp} re-resolves
      // it against the view the drag actually ends on.
      this.clearHover();
    }
    const z = this.effZoom;
    this.transform.centerX = this.panStartCenterX - dx / z;
    this.transform.centerY = this.panStartCenterY - dy / z;
    // Let the drag pull a little past the content edge with rubber-band
    // resistance instead of stopping dead, so hitting the wall reads as an
    // elastic edge; onMouseUp springs it back inside.
    this.softClampPan(z);
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
    if (wasPanning && this.dragMoved) {
      // The drag moved the map under the cursor with hover suppressed throughout,
      // so the pointer cache is as stale as the hover — re-sync it from the
      // release (which, being tracked on `document`, can land off-canvas).
      this.syncPointerFromEvent(event);
      // A drag that pulled past the content edge ends overshot; spring it back.
      this.settleToBounds();
      // Re-resolve the hover against where the view ended up: otherwise the
      // pre-pan bin stays lifted with its preview open, and a right-click still
      // targets it ({@link onContextMenu} prefers `hoveredCell` over a fresh
      // hit-test). A settle that actually animates does its own refresh when it
      // lands — refreshing here as well would resolve against the overshoot and
      // drag a lifted bin through the spring-back.
      if (!this.settleActive) this.refreshHoverAfterZoom();
    }
  }

  /**
   * Re-sync the cached pointer position (and inside-ness) from a drag event.
   * Drags are tracked on `document`, and {@link onCanvasMouseMove} — the usual
   * updater of that cache — is suppressed for their duration, so the cache is
   * stale by the time a drag ends and the hover has to be re-resolved.
   */
  private syncPointerFromEvent(event: MouseEvent): void {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    this.pointerInside = mx >= 0 && my >= 0 && mx < rect.width && my < rect.height;
    if (!this.pointerInside) return;
    this.lastMouseX = mx;
    this.lastMouseY = my;
    this.lastClientX = event.clientX;
    this.lastClientY = event.clientY;
  }

  /**
   * Schedule a single-click bin toggle, deferred by the double-click window so a
   * double-click (which zooms in) doesn't also flip the bin's selection. Any
   * pending toggle is committed first, so two quick clicks on *different* bins
   * each register rather than the second cancelling the first.
   */
  private scheduleToggle(mx: number, my: number): void {
    this.flushPendingToggle();
    // Resolve the bin against the transform the user clicked under — see
    // {@link pendingToggleCell}. Nothing to toggle over blank space.
    const cell = this.hitTest(mx, my);
    if (!cell) return;
    this.pendingToggleCell = cell;
    this.clickTimer = setTimeout(() => {
      this.clickTimer = null;
      this.commitPendingToggle();
    }, BrowseCanvasComponent.DBLCLICK_MS);
  }

  /** Run a pending single-click toggle now (if any) and clear the timer. */
  private flushPendingToggle(): void {
    if (this.clickTimer === null) return;
    clearTimeout(this.clickTimer);
    this.clickTimer = null;
    this.commitPendingToggle();
  }

  /** Flip the selection of the pending toggle's bin (bound at click time). */
  private commitPendingToggle(): void {
    const cell = this.pendingToggleCell;
    this.pendingToggleCell = null;
    if (!cell) return;
    // The selection store is signal-backed, so the write schedules change
    // detection on its own — no NgZone re-entry needed under zoneless.
    this.selection.toggleBin(this.cellMembers(cell));
    this.requestRedraw();
  }

  /** Drop a pending single-click toggle without running it (double-click path). */
  private cancelPendingToggle(): void {
    this.pendingToggleCell = null;
    if (this.clickTimer === null) return;
    clearTimeout(this.clickTimer);
    this.clickTimer = null;
  }

  /** Double-click zooms in about the cursor (map idiom). Shift/region-select are
   *  reserved for the marquee, so they don't zoom. */
  private onDblClick(event: MouseEvent): void {
    if (event.shiftKey || this.marqueeMode()) return;
    this.cancelPendingToggle();
    const [mx, my] = this.canvasXY(event);
    this.zoomBy(BrowseCanvasComponent.DOUBLE_CLICK_ZOOM, mx, my);
  }

  /** Right-click: suppress the native menu, pin the bin under the cursor so it
   *  stays enlarged, and ask the view to open its detail popup. A second
   *  right-click on empty canvas (no bin under the cursor) within the
   *  double-click window zooms out about the cursor instead, mirroring
   *  double-click-to-zoom-in. Landing on a bin at either click breaks the pair,
   *  so it never fires while browsing bin popups. */
  private onContextMenu(event: MouseEvent): void {
    event.preventDefault();
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    // Target the bin the cursor is already hovering first, falling back to a
    // hit-test only when nothing is hovered (the pointer only just arrived, or a
    // different bin is currently pinned). This is what makes the *first*
    // right-click open the detail view: a hovered thumbnail breaks out well past
    // its true hex, so a raw hit-test at the cursor can land outside the true
    // silhouette and find nothing — which used to shrink the bin and take a
    // second click. The hovered cell is unambiguously what the user is aiming at.
    const hovered = this.hoveredCell;
    const cell = hovered ?? this.hitTest(mx, my);
    // The level the pinned cell belongs to: the hover's own level when we reuse
    // it, otherwise the level the fresh hit-test just resolved against.
    const cellLevel = hovered ? this.hoveredLevel : this.activeLevel;
    // Stop the transient hover audition and drop the hover highlight; from here
    // the pinned cell drives the enlarge and the popup drives the audio.
    this.clearHover();

    if (!cell) {
      // Empty canvas: a second right-click within the double-click window zooms
      // out about the cursor (mirrors double-click-to-zoom-in). A bin under
      // either click breaks the pair.
      const now = event.timeStamp;
      const isDoubleRightClick = now - this.lastEmptyContextMenuAt < BrowseCanvasComponent.DBLCLICK_MS;
      this.lastEmptyContextMenuAt = isDoubleRightClick ? 0 : now;
      if (isDoubleRightClick) {
        this.zoomBy(1 / BrowseCanvasComponent.DOUBLE_CLICK_ZOOM, mx, my);
        return;
      }
    } else {
      this.lastEmptyContextMenuAt = 0;
    }

    // Pin the bin so it stays enlarged while its detail popup is open (right-
    // clicking another bin re-pins to it; dismissing the popup unpins via the
    // view calling {@link unpinCell}). Blank space clears any pin.
    this.pinnedCell = cell;
    this.pinnedLevel = cell ? cellLevel : -1;
    this.requestRedraw();
    this.contextMenu.emit({
      clientX: event.clientX,
      clientY: event.clientY,
      members: cell ? this.cellMembers(cell) : [],
      repId: cell ? cell.rep_id : null,
      bounds: rect,
    });
  }

  /**
   * Release the pinned bin — called by the browse view when the detail popup is
   * dismissed — and resume live hover at the current cursor position, so the bin
   * now under the cursor lifts (and, for audio, plays) again. No-op when nothing
   * is pinned.
   */
  unpinCell(): void {
    if (!this.pinnedCell) return;
    this.pinnedCell = null;
    this.pinnedLevel = -1;
    this.requestRedraw();
    if (this.pointerInside) {
      this.emitHoverHit(this.lastMouseX, this.lastMouseY, this.lastClientX, this.lastClientY);
    }
  }

  /** Canvas-relative ``[x, y]`` for a mouse event. */
  private canvasXY(event: MouseEvent): [number, number] {
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  /** Member media ids of a cell, falling back to its representative. */
  private cellMembers(cell: HexCellPayload): number[] {
    return cell.member_ids && cell.member_ids.length > 0 ? cell.member_ids : [cell.rep_id];
  }

  /** Add every bin whose centre falls inside the marquee rectangle. */
  private commitMarquee(): void {
    if (!this.marquee || !this.meta()) return;
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
      this.selection.addAll(ids);
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
    this.cancelSettle();
    this.cancelPanAnim();
    this.fitToData();
    this.commitZoomChange();
    this.refreshHoverAfterZoom();
  }

  /** Fraction of the viewport one arrow-key press pans. */
  private static readonly KEY_PAN_FRACTION = 0.2;

  /**
   * Pan the view one step per arrow-key press, ``(dirX, dirY)`` each in
   * ``{-1, 0, 1}``: +x pans right (reveals content to the right), +y pans down.
   * The move is eased (not snapped) so a N/E/S/W push glides like a zoom does;
   * see {@link startPanAnim}. Unlike a drag it hard-clamps to the content bounds
   * (no rubber-band / settle) since a keypress is a discrete programmatic move,
   * like the minimap recenter. A burst of presses retargets the glide on to the
   * further-out destination rather than restarting from the moving centre. Marks
   * the view user-framed so a later size change won't re-fit over the pan.
   */
  panByKey(dirX: number, dirY: number): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    if (this.width <= 0 || this.height <= 0) return;
    // A keyboard pan takes over from any in-flight zoom transition / snap-back.
    if (this.animActive) this.endZoomAnim();
    this.cancelSettle();
    this.framedByUser = true;
    const z = this.effZoom;
    const stepX = (dirX * this.width * BrowseCanvasComponent.KEY_PAN_FRACTION) / z;
    const stepY = (dirY * this.height * BrowseCanvasComponent.KEY_PAN_FRACTION) / z;
    // Accumulate onto the in-flight glide's destination so a second press while
    // the first is still easing keeps gliding further out, instead of snapping
    // the base back to the (still-moving) live centre.
    const base = this.panAnimActive ? this.panAnimTo : this.transform;
    const target = this.clampedTransform({
      centerX: base.centerX + stepX,
      centerY: base.centerY + stepY,
      zoom: this.transform.zoom,
    });
    this.startPanAnim(target);
  }

  /**
   * Glide to clamped `target` over {@link PAN_ANIM_MS} by freezing the current
   * frame and blitting it translated each step (picture-in-picture, the zoom=1
   * case of {@link startZoomAnim}) — so the cost is one snapshot plus cheap copies
   * rather than a full {@link draw} per frame. A pan keeps the same zoom and bins,
   * so `target.zoom` matches the current zoom and only the centre moves. The
   * snapshot overscans toward the move so the revealed edge shows cached content
   * sliding in. Honours reduced-motion by jumping straight to the target.
   */
  private startPanAnim(target: ViewTransform): void {
    const settled =
      Math.abs(target.centerX - this.transform.centerX) < 1e-3 &&
      Math.abs(target.centerY - this.transform.centerY) < 1e-3;
    // Already hard against the edge in this direction (and nothing in flight) —
    // there's nowhere to glide, so leave the view untouched.
    if (settled && !this.panAnimActive) return;
    // No prior frame to freeze (or motion disabled): jump straight to the target.
    if (prefersReducedMotion() || !this.hasDrawn || this.width <= 0) {
      this.cancelPanAnim();
      this.transform.centerX = target.centerX;
      this.transform.centerY = target.centerY;
      this.transform.zoom = target.zoom;
      this.updateActiveLevel();
      this.requestRedraw();
      this.refreshHoverAfterZoom();
      return;
    }

    const canvasEl = this.canvasRef().nativeElement;
    // Freeze from the frame on screen now (mid-glide that's the interpolated
    // centre, so a burst of presses chains seamlessly) and ease to `target`.
    this.panAnimFrom = { ...this.transform };
    this.panAnimTo = { ...target };
    const z = this.panAnimFrom.zoom;
    // Overscan toward the move so the revealed edge is real content, capped at
    // SNAP_OVERSCAN_MAX× the viewport (shared with the zoom-out buffer). Symmetric
    // margins keep the maths simple; the trailing side just slides off unused.
    // Low-effects mode glides with a plain viewport snapshot: the overscan
    // ring costs a bigger buffer to fill (renderSnapshotBorder walks every ring
    // bin) and to blit each frame — the exact work software rasterization can't
    // afford (issue #2695). The revealed edge shows bare background until the
    // glide lands, which the closing full draw immediately fills.
    const maxMarginX = this.lowFx ? 0 : (this.width * (BrowseCanvasComponent.SNAP_OVERSCAN_MAX - 1)) / 2;
    const maxMarginY = this.lowFx ? 0 : (this.height * (BrowseCanvasComponent.SNAP_OVERSCAN_MAX - 1)) / 2;
    const marginX = Math.min(maxMarginX, Math.abs(target.centerX - this.panAnimFrom.centerX) * z);
    const marginY = Math.min(maxMarginY, Math.abs(target.centerY - this.panAnimFrom.centerY) * z);
    // Render the overscan ring for every media type: without it the revealed
    // edge is bare background (a black gap) until the glide lands. The ring comes
    // straight from the cached tiles — thumbnails in thumbnail mode, colour-mapped
    // hexes in flat-density mode (audio/text) — so both slide in real content.
    const doBorder = marginX > 1 || marginY > 1;
    this.panSnapW = this.width + 2 * Math.ceil(marginX);
    this.panSnapH = this.height + 2 * Math.ceil(marginY);

    let snap = this.panSnapshot;
    if (!snap) snap = document.createElement('canvas');
    const wantW = Math.round(this.panSnapW * this.dpr);
    const wantH = Math.round(this.panSnapH * this.dpr);
    if (snap.width !== wantW || snap.height !== wantH) {
      snap.width = wantW;
      snap.height = wantH;
    }
    const sctx = snap.getContext('2d')!;
    sctx.setTransform(1, 0, 0, 1, 0, 0);
    sctx.clearRect(0, 0, snap.width, snap.height);
    if (doBorder) {
      // Ring first (cached bins around the frozen viewport), then drop the live
      // canvas copy into the centre so it overwrites any seam bins exactly.
      sctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.renderSnapshotBorder(
        sctx, this.panAnimFrom, this.panSnapW, this.panSnapH, this.activeLevel,
        this.width / 2, this.height / 2,
      );
      sctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    const ox = Math.round((this.panSnapW - this.width) * this.dpr * 0.5);
    const oy = Math.round((this.panSnapH - this.height) * this.dpr * 0.5);
    sctx.drawImage(canvasEl, ox, oy);
    this.panSnapshot = snap;

    this.panAnimBg = this.themeColor('--bg-body');
    this.panAnimStartTs = performance.now();
    // The glide owns the canvas now; drop any pending plain redraw.
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.needsRedraw = false;
    if (this.panAnimRafId) cancelAnimationFrame(this.panAnimRafId);
    this.panAnimActive = true;
    this.ngZone.runOutsideAngular(() => {
      this.panAnimRafId = requestAnimationFrame(this.stepPanAnim);
    });
  }

  /** One frame of the directional-pan glide: blit the frozen snapshot translated
   *  so its centre lands where the eased centre puts it, then paint the real
   *  rebinned frame once the glide lands. Zoom is constant, so the blit is a pure
   *  translation (no scale). */
  private readonly stepPanAnim = (now: number): void => {
    const ctx = this.ctx;
    const snap = this.panSnapshot;
    if (!this.panAnimActive || !snap || !ctx) return;
    // Clamp the elapsed fraction to [0, 1]. The lower clamp matters: `panAnimStartTs`
    // is stamped with `performance.now()` synchronously inside the keydown handler,
    // but the rAF callback is handed the *frame-start* timestamp. A keydown is
    // dispatched mid-frame, so the rAF it schedules can run in that same frame with
    // a `now` that predates the mid-frame stamp — `now - panAnimStartTs` then goes
    // negative, and easeOutCubic turns a negative `t` into a negative `e`, kicking
    // the view *backwards* (opposite the pan) for one frame. Mirror the same guard
    // the zoom transition uses.
    const frameStart = performance.now();
    const t = Math.min(1, Math.max(0, (now - this.panAnimStartTs) / BrowseCanvasComponent.PAN_ANIM_MS));
    const e = 1 - Math.pow(1 - t, 3);
    const from = this.panAnimFrom;
    const to = this.panAnimTo;
    const z = from.zoom;
    const curX = from.centerX + (to.centerX - from.centerX) * e;
    const curY = from.centerY + (to.centerY - from.centerY) * e;
    // Keep the live transform tracking the visible frame so a gesture that takes
    // over mid-glide (drag / zoom / minimap recenter) continues from here rather
    // than jumping to the destination.
    this.transform.centerX = curX;
    this.transform.centerY = curY;
    this.displayedTransform = { centerX: curX, centerY: curY, zoom: z };

    // Screen position the snapshot centre (frozen at `from.center`) maps to once
    // the interpolated transform is applied.
    const cxScreen = this.width / 2 + (from.centerX - curX) * z;
    const cyScreen = this.height / 2 + (from.centerY - curY) * z;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.panAnimBg;
    ctx.fillRect(0, 0, this.width, this.height);
    // The blit is a pure translation but its offsets are fractional, so
    // smoothing still resamples every pixel; low-effects mode snaps to
    // nearest-neighbour, which mid-glide is indistinguishable (issue #2695).
    ctx.imageSmoothingEnabled = !this.lowFx;
    ctx.drawImage(
      snap, 0, 0, snap.width, snap.height,
      cxScreen - this.panSnapW / 2, cyScreen - this.panSnapH / 2, this.panSnapW, this.panSnapH,
    );
    this.recordFrameCost(frameStart);

    if (t < 1) {
      this.panAnimRafId = requestAnimationFrame(this.stepPanAnim);
    } else {
      this.panAnimActive = false;
      this.panAnimRafId = 0;
      this.transform.centerX = to.centerX;
      this.transform.centerY = to.centerY;
      // Paint the real, rebinned frame at the destination. This is the first
      // draw() since the glide began, so it also republishes the viewport — the
      // minimap rectangle snaps to the new spot once, instead of animating along.
      this.draw();
      this.refreshHoverAfterZoom();
    }
  };

  /** Stop a directional-pan glide in flight (without snapping to its target) —
   *  used when a new gesture takes over, so it continues from the current frame. */
  private cancelPanAnim(): void {
    if (!this.panAnimActive) return;
    this.panAnimActive = false;
    if (this.panAnimRafId) {
      cancelAnimationFrame(this.panAnimRafId);
      this.panAnimRafId = 0;
    }
  }

  /**
   * Select every bin that lies *fully* within the current viewport — its whole
   * silhouette on screen, not clipped at an edge — adding all their members to
   * the selection. The ctrl-A affordance for the canvas; mirrors
   * {@link commitMarquee} but bounds by the viewport rather than a drawn
   * rectangle, and only walks tiles already cached (i.e. what's actually drawn).
   */
  selectAllInView(): void {
    const meta = this.meta();
    if (!meta || meta.point_count === 0) return;
    const level = this.activeLevel;
    const radius = meta.base_radius / Math.pow(2, level);
    const screenRadius = radius * this.effZoom;
    const ids: number[] = [];
    for (const { tx, ty } of this.getVisibleTiles()) {
      const tile = this.tileCache.getCached(level, tx, ty);
      if (!tile) continue;
      for (const cell of tile.cells) {
        const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
        // Fully on-screen: the cell's whole extent (its circumradius) clears
        // every edge, so a bin clipped by the viewport border is left out.
        if (sx - screenRadius < 0 || sx + screenRadius > this.width) continue;
        if (sy - screenRadius < 0 || sy + screenRadius > this.height) continue;
        for (const id of this.cellMembers(cell)) ids.push(id);
      }
    }
    // Latches the selection panel's tri-state checkbox to [x]; an empty view is
    // a no-op inside the service.
    this.selection.selectAllInView(ids);
  }

  zoomBy(factor: number, anchorX = this.width / 2, anchorY = this.height / 2): void {
    this.framedByUser = true;
    // A zoom takes over from any in-flight snap-back or directional glide;
    // continue from where the view visually is and reschedule the settle below.
    this.cancelSettle();
    this.cancelPanAnim();
    const [projX, projY] = this.screenToProj(anchorX, anchorY);
    const rawZoom = Math.max(0.01, Math.min(100000, this.transform.zoom * factor));
    // Below the whole-projection fit the zoom-out is resisted, not blocked, and
    // above the finest-level ceiling the zoom-in is likewise resisted, so the
    // wheel keeps responding at either edge; the overshoot is sprung back by the
    // debounced settle once the wheel goes quiet. (The two limits never overlap —
    // computeMaxZoom floors at computeFitZoom — so the order of the two soft
    // clamps doesn't matter.)
    const newZoom = this.softCeilZoom(this.softFloorZoom(rawZoom));
    // Keep the point under the cursor fixed while zooming.
    this.transform.centerX = projX - (anchorX - this.width / 2) / newZoom;
    this.transform.centerY = projY - (anchorY - this.height / 2) / newZoom;
    this.transform.zoom = newZoom;
    // Let the pan drift past the content edge with the same rubber-band give a
    // zoom-out off the anchor would otherwise slam into.
    this.softClampPan(newZoom);

    // Zoom holds the thumbnail size (targetRadius) and re-selects the level, so
    // a smaller region is re-binned more finely while bins stay ~the same size.
    // The picture-in-picture transition eases the current frame into place on
    // every notch — growing/shrinking it before any rebin lands when the
    // re-selection crosses a level, and simply scaling it within a level.
    this.updateActiveLevel();
    this.commitZoomChange();
    this.refreshHoverAfterZoom();
    this.scheduleSettle();
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
    this.cancelSettle();
    this.cancelPanAnim();
    const meta = this.meta();
    if (reframe && meta && this.fittedAgainstRealSize) {
      this.framedByUser = true;
      this.transform.zoom *= radius / this.targetRadius;
    }
    this.targetRadius = radius;
    // A large enough size crosses into full-res /image territory; drop any
    // capped thumbnails so cells reload sharp (and vice versa on shrink).
    this.syncThumbResolutionTier();
    // On initial load (the user hasn't framed yet) a saved cell size changes the
    // bin radius, so re-fit to keep the whole projection in view rather than
    // letting the now-larger edge bins spill past the default-radius framing.
    // Once the user has panned/zoomed, only re-select the level so their view is
    // preserved.
    if (!reframe && !this.framedByUser && meta && this.fittedAgainstRealSize) {
      this.fitToData();
    } else {
      // A reframe scales the zoom in lock-step with the size, which can push it
      // below the whole-projection fit (shrinking thumbnails) — clamp it back.
      this.clampView();
      this.updateActiveLevel();
    }
    this.requestRedraw();
    if (reframe) this.refreshHoverAfterZoom();
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    // Gate the wheel while a left-drag pan or marquee is in progress. A zoom
    // mid-drag re-anchors the transform about the cursor, but the next
    // {@link onMouseMove} recomputes the centre from the drag's start values
    // (`panStartCenter` minus `dx / z`), discarding that re-anchoring — and for
    // a marquee it would shift the screen-space rectangle out from under the
    // pointer. The zoom transition it kicks off also owns the canvas for its
    // ~220 ms, freezing the pan/marquee paint. Ignoring the notch (the button is
    // still down) keeps the in-progress gesture coherent; the wheel resumes the
    // instant the drag ends.
    if (this.isPanning || this.isMarquee) return;
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const wheelFactor = this.wheelZoomFactor();
    const factor = event.deltaY < 0 ? wheelFactor : 1 / wheelFactor;
    this.zoomBy(factor, mx, my);
  }

  private onCanvasMouseMove(event: MouseEvent): void {
    // No hover while panning or marqueeing (mid-drag), and none at all in
    // region-select mode: the cursor is a crosshair for drawing a box, so a
    // hover preview/highlight popping up under it would just be noise.
    if (this.isPanning || this.isMarquee || this.marqueeMode()) return;
    const rect = this.canvasRef().nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    this.lastMouseX = mx;
    this.lastMouseY = my;
    this.lastClientX = event.clientX;
    this.lastClientY = event.clientY;
    this.pointerInside = true;

    // A pinned bin (its detail popup is open) has precedence: suppress hover on
    // other bins so the popup keeps its enlarge and audition. Keep tracking the
    // cursor above, so hover resolves correctly the moment the popup is closed.
    if (this.pinnedCell) return;

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
    // Hover is suppressed while a bin's detail popup is pinned (also guards the
    // post-zoom refresh path, which calls this directly).
    if (this.pinnedCell) return;
    const hit = this.hitTest(mx, my);
    const level = this.activeLevel;
    const prev = this.hoveredCell;
    const prevLevel = this.hoveredLevel;
    this.hoveredCell = hit;
    this.hoveredLevel = hit ? level : -1;
    if (hit) {
      // Compared with the level included: a post-zoom refresh that lands on a
      // finer cell sharing the old (q, r) is a *different* bin, so it must
      // re-emit — otherwise the preview keeps describing the coarser bin.
      if (!sameBin(level, hit, prevLevel, prev)) {
        this.hexHover.emit({ cell: hit, screenX: clientX, screenY: clientY });
        this.requestRedraw();
      }
    } else if (prev) {
      this.hexHover.emit(null);
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
      this.hoveredLevel = -1;
      this.hexHover.emit(null);
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
      this.hoveredLevel = -1;
      this.hexHover.emit(null);
      this.requestRedraw();
    }
  }

  private hitTest(sx: number, sy: number): HexCellPayload | null {
    const meta = this.meta();
    if (!meta) return null;
    const [px, py] = this.screenToProj(sx, sy);
    const level = this.activeLevel;
    const radius = meta.base_radius / Math.pow(2, level);
    const geom = this.geom;
    const tileW = meta.tile_span * geom.dx(radius);
    const tileH = meta.tile_span * geom.dy(radius);

    const txEst = Math.floor(px / tileW);
    const tyEst = Math.floor(py / tileH);

    // Gather every cell from the 3×3 block of tiles around the cursor, then let
    // the pure `pickCell` rule find the nearest-and-containing one (or null over
    // blank space). The tile lookup stays here; the geometry is testable on its own.
    const candidates: HexCellPayload[] = [];
    for (let dtx = -1; dtx <= 1; dtx++) {
      for (let dty = -1; dty <= 1; dty++) {
        const tile = this.tileCache.getCached(level, txEst + dtx, tyEst + dty);
        if (tile) candidates.push(...tile.cells);
      }
    }

    return pickCell(candidates, px, py, geom, radius);
  }
}
