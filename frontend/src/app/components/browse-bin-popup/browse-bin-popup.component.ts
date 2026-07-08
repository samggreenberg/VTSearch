import { AfterViewInit, ChangeDetectionStrategy, ChangeDetectorRef, Component, effect, ElementRef, HostListener, inject, input, OnChanges, OnDestroy, output, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { IconComponent } from '../icon/icon.component';
import { CopyDetailButtonComponent } from '../copy-detail-button/copy-detail-button.component';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { applyClipWindow, clearClipWindow } from '../../utils/clip-window';
import { usesThumbnails } from '../browse-canvas/hex-render.util';
import { shortcutsBlocked } from '../../utils/keyboard-shortcuts';
import type { NowPlaying } from '../browse-hover-preview/browse-hover-preview.component';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';
import type { MediaBatchResponse } from '../../generated/api-client/models/media-batch-response';

/** Height (px) of the name's line box under a grid thumbnail. Mirrors the pinned
 *  ``line-height`` on ``.bin-popup-name`` (``--font-xs`` = 12px × 1.25 = 15px);
 *  kept in sync so {@link rowSize} reserves exactly the space the name renders in.
 *  The name's line-height is pinned (rather than left at the font's ``normal``,
 *  which varies ~1.15–1.35 by platform font) precisely so this stays exact — an
 *  unpinned name renders a couple px taller than its slot, which is enough to
 *  force a stray scrollbar on even a single row. */
const GRID_LABEL_HEIGHT = 15;
/** Gap (px) between the thumbnail and its name inside a cell (``.bin-popup-entry``
 *  flex ``gap``); present only when the name shows. */
const GRID_THUMB_NAME_GAP = 2;
/** Vertical padding (px) inside every grid cell (``.bin-popup-entry`` 2px top +
 *  2px bottom), always present regardless of icon size. Reserved in {@link
 *  rowSize} so the cell's real rendered height never exceeds its virtual slot. */
const GRID_CELL_PADDING = 4;
/** Goal width (px) at/above which grid thumbnails still show their name. Below
 *  this (the XS/S icon sizes) the name truncates to a useless "a…", so the SCSS
 *  hides it (``@container … (max-width: 60px)``) and {@link rowSize} drops the
 *  reserved label height to match, keeping the grid gap-free. */
const GRID_NAME_MIN_WIDTH = 65;
/** Gap (px) between grid cells (and grid rows); matches ``--space-2xs``-ish. */
const GRID_GAP = 4;
/** Width (px) available to lay out cells inside the popup's scroll column (≈ its
 *  width minus padding and the scrollbar). Columns are derived from this. */
const GRID_CONTENT_WIDTH = 256;
/** Width (px) of the scrolling grid column; mirrors the historic popup width. */
const GRID_COLUMN_WIDTH = 280;
/** Width (px) of the optional metadata column shown left of the preview pane. */
const METADATA_COLUMN_WIDTH = 200;
/** Tallest the scrolling body grows before it caps and scrolls internally. */
const MAX_BODY_PX = 400;
/** Shortest the scrolling body is ever squeezed to when the visible region is
 *  too short to fit the full popup; below this it just scrolls internally. */
const MIN_BODY_PX = 80;
/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/** Gap (px) kept between the popup and the visible edge when clamping. */
const EDGE_MARGIN = 8;
/** Gap (px) between the body's columns (metadata / preview / grid). Mirrors the
 *  body's flex ``gap: var(--space-sm)`` (6px); used by {@link clampInto} to model
 *  the popup's real width so the computed clamp matches what renders. */
const COLUMN_GAP = 6;
/** Horizontal/vertical padding (px) inside ``.bin-popup-body`` on each side
 *  (``padding: var(--space-sm)`` = 6px). The body is ``box-sizing: border-box``,
 *  so its bound ``height`` already includes the vertical padding, but its
 *  fit-content *width* grows by this on each side; {@link clampInto} folds it in
 *  so the modelled width equals the rendered box. */
const BODY_PADDING = 6;
/** Popup border (px) per side (``.bin-popup`` ``border: 1px``). Added to the
 *  modelled width/height in {@link clampInto} so the computed clamp accounts for
 *  the full rendered footprint rather than depending on {@link nudgeOnScreen} to
 *  mop up the residual — the audio (no-preview) layout has no large preview pane
 *  to anchor its size, so its clamp must be exact on its own. */
const POPUP_BORDER = 1;
/** Smallest the preview pane is squeezed to in a tight region. */
const MIN_PREVIEW_PX = 96;
/** Largest the preview pane grows, regardless of icon size, so an extreme
 *  thumbnail-size setting can't make the popup absurdly large. */
const MAX_PREVIEW_PX = 720;
/**
 * How much larger the preview pane is than the item's on-canvas mouse-over size.
 * Sized well past the bare hover break-out so the zoom-in pane reads as a proper
 * detail view rather than a slightly-enlarged thumbnail. 2.0 == +100%.
 */
const PREVIEW_OVERSIZE = 2.0;
/**
 * Full on-screen extent (px) of a square thumbnail's hover break-out on the main
 * canvas, as a multiple of the bin radius. A hovered thumbnail grows until its
 * edge just reaches the nearest neighbour centre; for a square image on a hex
 * lattice the binding neighbour sits at ``1.5 * radius`` (see
 * ``hoverThumbRect`` in browse-canvas), so the full height is ``2 * 1.5 = 3``
 * radii. The preview tracks that reference size, oversized by
 * {@link PREVIEW_OVERSIZE}.
 */
const HOVER_EXTENT_PER_RADIUS = 3;
/** Vertical room (px) the member-count label takes above the scrolling grid. */
const COUNT_LABEL_HEIGHT = 22;
/** Vertical padding (px) inside ``.bin-popup-body`` ({@link BODY_PADDING} on the
 *  top + bottom). The body's bound ``height`` is a *border-box* height (the app
 *  sets ``box-sizing: border-box`` globally), so this padding must be added on
 *  top of the content the flex columns need ({@link bodyHeight}); otherwise the
 *  padding eats into the content area and each column's ``height: 100%`` resolves
 *  to 12px less than {@link bodyHeight}. For audio (where the member grid is the
 *  exact-fit element) that shortfall forces a stray scrollbar on even a single
 *  row; for the square preview pane it renders the box 12px non-square. */
const BODY_PADDING_Y = 2 * BODY_PADDING;
/**
 * Discrete ladder (px) of detail-canvas sizes the top-left size buttons step
 * through. Each click moves to the next rung past the current size in the click
 * direction, so the popup grows/shrinks in clean increments. Bounded by
 * {@link MIN_PREVIEW_PX}..{@link MAX_PREVIEW_PX}.
 */
const PREVIEW_SIZE_STEPS = [120, 160, 208, 272, 352, 448, 560, 640, 720] as const;

/**
 * The bin popup: a floating panel showing the media items in the bin the user
 * right-clicked on the VTSBrowse canvas, plus — for thumbnail media (image /
 * video) — a large preview pane on the left.
 *
 * The members render as a virtualized thumbnail grid (always grid; there is no
 * list mode). Hovering a grid thumbnail paints that item's *full-resolution*
 * original (not its grid thumbnail) into the preview pane, so the user can pull
 * detail out of any pile member in turn. The pane opens showing the bin's
 * representative, so even a singleton bin lands on a large high-res view without
 * any hover. The pane is sized to {@link PREVIEW_OVERSIZE} (50%) larger than the
 * item's on-canvas mouse-over break-out at the current main-canvas thumbnail
 * size ({@link hoverThumbRadius}).
 *
 * The thumbnail size of the grid is remembered per media type under the
 * ``grid_icon_size_popup`` setting (independent of the left/right panels), so
 * tuning the popup while browsing one bin becomes the default for every future
 * popup of that media type. The top-right {@link ViewControlsComponent} writes
 * that setting and this component re-reads it from {@link SettingsStateService},
 * keyed by the active dataset's media type.
 *
 * A second, top-left pair of size buttons controls the *detail canvas* (the
 * large preview pane) rather than the grid thumbnails. Because the popup's
 * height is the preview pane's height, growing/shrinking the detail canvas
 * resizes the whole window. That size is likewise remembered per media type,
 * under ``popup_preview_size`` (px); unset, the pane falls back to a size scaled
 * from the main-canvas thumbnail radius ({@link hoverThumbRadius}).
 *
 * It shares the {@link BrowseSelectionService} instance provided by the browse
 * view, so toggling an item here is the same selection the canvas rings and the
 * selection panel reflect; selected members render highlighted and update live.
 * Names/thumbnails resolve lazily through {@link MediaMetadataCacheService}
 * (the browse view never loads the full media list), prefetched around the
 * visible window so large bins stay responsive.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-bin-popup',
  standalone: true,
  imports: [CommonModule, ScrollingModule, ViewControlsComponent, IconComponent, CopyDetailButtonComponent],
  templateUrl: './browse-bin-popup.component.html',
  styleUrl: './browse-bin-popup.component.scss',
})
export class BrowseBinPopupComponent implements AfterViewInit, OnChanges, OnDestroy {
  private host = inject<ElementRef<HTMLElement>>(ElementRef);
  private selection = inject(BrowseSelectionService);
  private metadataCache = inject(MediaMetadataCacheService);
  private activeContext = inject(ActiveContextService);
  private settingsState = inject(SettingsStateService);
  private cdr = inject(ChangeDetectorRef);

  /** Member media ids of the bin the popup was summoned over. */
  readonly memberIds = input<number[]>([]);
  /** The bin's representative (centroid) id — the clip whose thumbnail is drawn
   *  for the pile on the canvas. The popup opens its preview on this item and
   *  scrolls the member grid to it, so the detail view starts on the same image
   *  the user right-clicked rather than the 1-D list's first item. Null falls
   *  back to the first member. */
  readonly repId = input<number | null>(null);
  /** Active dataset media type, used for the view prefs, hover-to-hear, and placeholders. */
  readonly mediaType = input('');
  /** Viewport anchor (clientX/clientY) the popup opens at, then clamps inward. */
  readonly x = input(0);
  readonly y = input(0);
  /** The canvas's bounding rect (viewport coords); the popup is clamped to the
   *  on-screen part of it so it stays fully visible. Null falls back to the
   *  full viewport. */
  readonly bounds = input<DOMRect | null>(null);
  /** Current on-screen bin radius (CSS px) of the main canvas, i.e. the radius
   *  the hovered thumbnail breaks out from. The preview pane is sized relative
   *  to this so it tracks the main-canvas thumbnail-size setting. */
  readonly hoverThumbRadius = input(28);
  /** Preview-audio volume (0–1), driven by the Browse toolbar's volume control. */
  readonly volume = input(1);

  /** Emitted when the popup should close (outside click, Escape, or the X). */
  readonly dismissed = output<void>();
  /** The clip now auditioning from a grid-row hover (for the top-left
   *  now-playing indicator, shared with the canvas hover), or ``null`` once
   *  the hover clears. */
  readonly nowPlaying = output<NowPlaying | null>();

  @ViewChild('panel') private panelRef?: ElementRef<HTMLElement>;
  @ViewChild('header') private headerRef?: ElementRef<HTMLElement>;
  @ViewChild(CdkVirtualScrollViewport) private viewport?: CdkVirtualScrollViewport;
  @ViewChild('audioEl') private audioRef?: ElementRef<HTMLAudioElement>;

  /** Clamped on-screen position; starts at the anchor and is nudged inward. */
  left = 0;
  top = 0;
  /** False until the popup has been clamped *and* measured at its on-screen spot;
   *  the panel is kept ``visibility: hidden`` until then so the user never sees it
   *  flash at the raw summon point (which may be half off-screen) or at a computed
   *  clamp that the post-render measurement then corrects. Flipped true only in
   *  {@link nudgeOnScreen} (the rAF after {@link place}), once the rendered panel
   *  has been measured and any residual overflow removed, and only once settings
   *  have loaded so the size is final. Reset on every genuine re-summon so the
   *  move to a new bin also stays hidden until re-placed. */
  placed = false;
  /** Max width (px) the popup may take; shrunk to fit a narrow visible region. */
  maxWidthPx = GRID_COLUMN_WIDTH;
  /** True once the user has dragged the popup by its header. While set, the
   *  popup keeps the user's chosen spot (re-clamped to stay on-screen) instead
   *  of re-anchoring to the summon point on content changes. Reset per bin. */
  dragged = false;
  /** True only mid-drag (pointer down on the header, not yet released). */
  dragging = false;
  private dragStart = { x: 0, y: 0, left: 0, top: 0 };

  /** Body height cap (px) for the current visible region; the body never grows
   *  past this, so a short region can't push the popup off the bottom edge. */
  bodyCapPx = MAX_BODY_PX;

  /** Preview-pane height cap (px) for the current visible region. The preview is
   *  allowed to grow taller than {@link bodyCapPx} (which only governs the
   *  scrolling grid), so a large zoom-in view isn't truncated to the grid's cap;
   *  it is still kept within the visible region so the popup stays on-screen. */
  previewCapPx = MAX_PREVIEW_PX;

  /** The bin's member ids, in bin order — a locality-preserving 1-D (Hilbert)
   *  traversal of the layout, so spatially/semantically similar items sit
   *  together in the list (the server orders them; see ``tile_member_ids``). */
  ids: number[] = [];

  /** Grid thumbnail target width (px), mirrored from settings per media type. */
  gridGoalWidth = 80;

  /** Number of columns the grid lays out. */
  columns = 1;
  /** Member ids chunked into rows of {@link columns}; the virtual list's data. */
  rows: number[][] = [];

  /** Id whose full-res original is painted into the preview pane. Defaults to
   *  the bin's representative (first member) so the pane is never blank, and
   *  follows the grid thumbnail under the cursor while hovering. */
  previewId: number | null = null;

  /** Currently-playing hover audio source, so re-entering the same row is a no-op. */
  audioSrc = '';

  private readonly failedThumbs = new Set<string>();
  /** Ids whose full-res ``/image`` failed; the preview falls back to the
   *  thumbnail for these so it still shows something. */
  private readonly failedPreviews = new Set<number>();
  private readonly subs: Subscription[] = [];
  private scrollSub: Subscription | null = null;

  constructor() {
    // Keep the hover-to-hear preview element in step with the Browse toolbar's
    // volume slider mid-playback; ``onEntryEnter`` also seeds it when a clip
    // starts.
    effect(() => {
      const el = this.audioRef?.nativeElement;
      if (el) el.volume = this.volume();
    });
    // Re-read the popup's thumbnail size whenever settings change (this is how
    // the in-header size buttons take effect, and how a change on one popup
    // becomes the default for every future popup of this media type).
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      this.gridSizeDict = (settings.grid_icon_size_popup as Record<string, string>) ?? {};
      this.previewSizeDict = (settings.popup_preview_size as Record<string, number>) ?? {};
      this.metadataShownDict = (settings.popup_metadata_shown as Record<string, boolean>) ?? {};
      this.applyViewPrefs();
      // A detail-canvas size change (the top-left buttons) resizes the preview
      // pane, hence the whole popup, so re-clamp it back fully on-screen. Showing
      // or hiding the metadata column likewise changes the popup's width, so it
      // re-clamps too.
      const override = this.previewOverride;
      const metaShown = this.showMetadataColumn;
      if (override !== this.lastPreviewOverride || metaShown !== this.lastMetadataShown) {
        this.lastPreviewOverride = override;
        this.lastMetadataShown = metaShown;
        setTimeout(() => this.place());
      }
    });
    // A selection change anywhere (here, the canvas, the panel) re-highlights.
    // An effect on the signal (rather than a `changed$` subscription) schedules
    // the repaint under zoneless from any mutation context.
    effect(() => {
      this.selection.version();
      this.cdr.markForCheck();
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['memberIds'] || changes['repId']) {
      this.ids = this.memberIds() ?? [];
      this.stopAudio();
      // Open on the bin's representative (the centroid whose thumbnail the user
      // right-clicked) so the pane is never blank and the detail view starts on
      // the same image — not the 1-D list's first item, which differs.
      this.previewId = this.representativeId();
      // Auto-play the representative's clip on open (audio) so the detail window
      // is an "intense hover": opening it hears the rep, just like resting on the
      // bin on the canvas. This is the only way to hear a singleton audio bin,
      // whose member grid (and its hover-to-hear) is dropped (see previewOnly).
      if (this.previewId != null) this.playAudio(this.previewId);
      // A fresh bin is a fresh popup: forget any drag from the previous one so
      // it re-anchors to the new summon point.
      this.dragged = false;
      this.rebuildRows();
      // A fresh bin: scroll the list to the representative (centred) and prefetch
      // the window around it. Deferred so the virtual viewport has the new rows
      // (and, on first open, exists at all — it's created after ngOnChanges).
      setTimeout(() => {
        this.scrollToRep();
        this.prefetchVisible();
      });
    }
    if (changes['mediaType']) {
      // A new media type may carry a different remembered thumbnail size.
      this.applyViewPrefs();
    }
    // A genuine (re)summon — a fresh bin or a new anchor/region — re-seeds the
    // position at the summon point (unless the user has dragged the popup). A
    // pure size change (the main-canvas thumbnail radius) must NOT snap back to
    // the cursor: it keeps the popup where it sits and only re-clamps it
    // on-screen. Without this, the first resize after opening lurches the window
    // from the cursor across to the clamped edge, because ``place`` was
    // re-deriving the position from the raw summon point every time rather than
    // from where the window had settled.
    const resummoned = changes['x'] || changes['y'] || changes['bounds'] || changes['memberIds'];
    if (resummoned || changes['hoverThumbRadius']) {
      if (resummoned && !this.dragged) {
        this.left = this.x();
        this.top = this.y();
        // Hide the popup until it's re-clamped at the new anchor, so it doesn't
        // flash at the raw summon point (which may be half off-screen) first.
        this.placed = false;
      }
      // Measure + clamp after the new content lays out.
      setTimeout(() => this.place());
    }
  }

  ngAfterViewInit(): void {
    this.subs.push(
      // Names/thumbnails arrive asynchronously; repaint the visible rows.
      this.metadataCache.version$.subscribe(() => this.cdr.markForCheck()),
    );
    this.settingsState.load();
    this.scrollSub =
      this.viewport?.scrolledIndexChange.subscribe(() => this.prefetchVisible()) ?? null;
    this.prefetchVisible();
    setTimeout(() => this.place());
  }

  ngOnDestroy(): void {
    for (const sub of this.subs) sub.unsubscribe();
    this.scrollSub?.unsubscribe();
    this.stopAudio();
  }

  private gridSizeDict: Record<string, string> = {};
  /** Per-media-type detail-canvas size (px) the user has chosen via the popup's
   *  top-left buttons; absent entries fall back to the radius-derived default. */
  private previewSizeDict: Record<string, number> = {};
  /** Last applied preview override, so the settings effect only re-clamps the
   *  popup when the detail-canvas size actually changed. */
  private lastPreviewOverride: number | null = null;
  /** Per-media-type memory of whether the metadata column is shown, mirrored
   *  from the ``popup_metadata_shown`` setting. Absent entries default to shown
   *  (mirroring the Train/Find center panel's metadata default). */
  private metadataShownDict: Record<string, boolean> = {};
  /** Last applied column visibility, so the settings effect only re-clamps the
   *  popup when it actually toggled. */
  private lastMetadataShown: boolean | null = null;

  /** True for media types that carry real visual thumbnails (image / video):
   *  the ones that magnify on the main canvas and are worth a large preview. */
  get showPreview(): boolean {
    return usesThumbnails(this.mediaType());
  }

  // --- Metadata column ------------------------------------------------------
  // A column carrying the same fields the Train/Find center panel shows for the
  // focused item (name, media type, custom metadata, MD5). For thumbnail media
  // (image / video) it sits left of the detail-canvas preview; for non-preview
  // media (audio / text / document) it sits left of the member grid and tracks
  // the hovered/arrowed item, so the user can still read metadata for the item
  // under the cursor even without a preview pane. It is offered for every media
  // type, since the fields are media-agnostic.

  /** Whether the metadata toggle (the Info button) is offered. The panel is
   *  media-agnostic, so it's available for every media type — audio and the
   *  other non-preview types included — as long as there's an active type. */
  get canToggleMetadata(): boolean {
    return !!this.mediaType();
  }

  /** Whether the user has the metadata column shown for the active media type.
   *  Absent entries default to shown, mirroring the center panel's metadata
   *  tray, which is expanded by default. */
  get metadataShown(): boolean {
    const mediaType = this.mediaType();
    const value = mediaType ? this.metadataShownDict[mediaType] : undefined;
    return value !== false;
  }

  /** Whether the metadata column actually renders: the metadata feature is
   *  offered for this media type and the user hasn't hidden it. */
  get showMetadataColumn(): boolean {
    return this.canToggleMetadata && this.metadataShown;
  }

  /** Fixed width (px) of the metadata column when shown. */
  get metadataColWidth(): number {
    return METADATA_COLUMN_WIDTH;
  }

  /** Toggle the metadata column for the active media type and persist it (per
   *  media type, under ``popup_metadata_shown``), so it becomes the default for
   *  future popups of this type — mirroring how the size buttons persist. The
   *  settings effect re-clamps the popup, since the width changed. */
  toggleMetadata(): void {
    const mediaType = this.mediaType();
    if (!mediaType) return;
    const dict = { ...this.metadataShownDict, [mediaType]: !this.metadataShown };
    this.settingsState.update({ popup_metadata_shown: dict } as SettingsUpdate).subscribe();
  }

  /** Cached metadata for the focused (previewed) item, or ``undefined`` before
   *  it has loaded. Everything the column shows reads through this. */
  private focusedMedia(): MediaBatchResponse | undefined {
    return this.previewId == null ? undefined : this.metadataCache.get(this.previewId);
  }

  /** Display name of the focused item, matching the center panel's Name field. */
  get metadataName(): string {
    const media = this.focusedMedia();
    if (!media) return this.previewId == null ? '' : `Media #${this.previewId}`;
    return media.filename || `Media #${media.id}`;
  }

  /** Media type of the focused item, matching the center panel's Media Type
   *  field. Falls back to the active dataset's media type before load. */
  get metadataMediaType(): string {
    return this.focusedMedia()?.media_type || this.mediaType();
  }

  /** MD5 of the focused item, matching the center panel's MD5 field. */
  get metadataMd5(): string {
    return this.focusedMedia()?.md5 ?? '';
  }

  /** Custom metadata of the focused item — the category name/value pairs the
   *  center panel renders between Media Type and MD5. */
  get metadataCustom(): Record<string, unknown> {
    return (this.focusedMedia()?.custom_metadata as Record<string, unknown>) ?? {};
  }

  /** Format a custom-metadata value for display. Mirrors the center panel's
   *  ``formatMetadataValue`` so the same categories read identically here. */
  formatMetadataValue(label: string, value: unknown): string {
    if (label === 'File Size' && typeof value === 'number') {
      return (value / 1024).toFixed(1) + ' KB';
    }
    if (label === 'Duration' && typeof value === 'number') {
      return value.toFixed(1) + 's';
    }
    if (label === 'Frequency' && typeof value === 'number') {
      return value + ' Hz';
    }
    return String(value);
  }

  /** Pull the remembered thumbnail size for the active media type, rechunk the
   *  rows, and re-clamp (the body height may have changed). */
  private applyViewPrefs(): void {
    const prevGoal = this.gridGoalWidth;
    const mediaType = this.mediaType();
    this.gridGoalWidth = iconSizeToGoalWidth(
      (mediaType && this.gridSizeDict[mediaType]) || 'M',
    );
    if (this.gridGoalWidth !== prevGoal) {
      this.rebuildRows();
      // The row stride changed, so the old scroll offset now points at a
      // different row. Let the virtual viewport remeasure, re-centre on the item
      // being viewed (the representative until the user hovers/arrows elsewhere)
      // so it stays in view across the size change, then clamp.
      setTimeout(() => {
        this.viewport?.checkViewportSize();
        this.centreRowFor(this.focusIndex());
        this.place();
      });
    }
    this.cdr.markForCheck();
  }

  /** The id the preview opens on and the grid scrolls to: the bin's
   *  representative (centroid) when it's a member, else the first member so the
   *  pane is never blank. */
  private representativeId(): number | null {
    const rep = this.repId();
    if (rep != null && this.ids.includes(rep)) return rep;
    return this.ids.length > 0 ? this.ids[0] : null;
  }

  /** Index of the representative within {@link ids} (the bin's 1-D order), or 0
   *  when it isn't resolvable so we fall back to the top of the list. */
  private repIndex(): number {
    const rep = this.representativeId();
    const idx = rep == null ? -1 : this.ids.indexOf(rep);
    return idx >= 0 ? idx : 0;
  }

  /** Scroll the member grid so the representative's row sits roughly centred, so
   *  the popup opens looking at the same item whose pile thumbnail was clicked
   *  rather than the 1-D list's first item. */
  private scrollToRep(): void {
    this.centreRowFor(this.repIndex());
  }

  /** Scroll the member grid so the row holding ``index`` sits roughly centred.
   *  No-op for a singleton bin (no grid) or before the viewport exists.
   *
   *  The virtual viewport may not have applied its scrollable content size yet
   *  (on open, or right after a thumbnail-size change re-chunks the rows), so the
   *  browser clamps this scroll back to 0 and a large bin is left sitting at the
   *  top with the target off-screen. When we meant to scroll down but the offset
   *  didn't take, retry on the next frame (bounded) until the viewport is
   *  scrollable and the target sticks. */
  private centreRowFor(index: number, attempt = 0): void {
    const vp = this.viewport;
    if (!vp) return;
    const row = Math.floor(index / Math.max(1, this.columns));
    const viewportH = vp.elementRef.nativeElement.clientHeight || this.gridHeight;
    // The most we can scroll: content height (all rows) minus the visible window.
    const maxOffset = Math.max(0, this.rows.length * this.rowSize - viewportH);
    // Centre the row in the visible window, clamped to [0, maxOffset] so we never
    // scroll past either end.
    const target = Math.min(
      maxOffset,
      Math.max(0, row * this.rowSize - Math.max(0, viewportH - this.rowSize) / 2),
    );
    vp.scrollToOffset(target);
    if (target > 0 && vp.measureScrollOffset('top') < target - 1 && attempt < 5) {
      requestAnimationFrame(() => this.centreRowFor(index, attempt + 1));
    }
  }

  /** Recompute the column count + row chunking for the current thumbnail size. */
  private rebuildRows(): void {
    this.columns = Math.max(
      1,
      Math.floor((GRID_CONTENT_WIDTH + GRID_GAP) / (this.gridGoalWidth + GRID_GAP)),
    );
    const cols = this.columns;
    const rows: number[][] = [];
    for (let i = 0; i < this.ids.length; i += cols) {
      rows.push(this.ids.slice(i, i + cols));
    }
    this.rows = rows;
  }

  /** Pixel stride of one virtual grid row: the cell's real rendered height (the
   *  thumbnail, its always-present vertical padding, and — when shown — the
   *  thumb→name gap and name line box) plus an inter-row gap. Accounting for the
   *  cell padding and pinned name line box keeps the row's rendered content from
   *  overflowing its virtual slot by a sub-pixel, which would otherwise force a
   *  stray scrollbar on even a single row. At the smallest icon sizes the name is
   *  hidden (see {@link GRID_NAME_MIN_WIDTH}), so its gap + label height are
   *  dropped to keep rows flush. */
  get rowSize(): number {
    const nameShown = this.gridGoalWidth >= GRID_NAME_MIN_WIDTH;
    const nameHeight = nameShown ? GRID_THUMB_NAME_GAP + GRID_LABEL_HEIGHT : 0;
    return this.gridGoalWidth + GRID_CELL_PADDING + nameHeight + GRID_GAP;
  }

  /** Height (px) the grid takes: just enough for its rows, capped (to the room
   *  the visible region leaves, see {@link bodyCapPx}) then scrolled. */
  get gridHeight(): number {
    return Math.min(Math.max(this.rows.length, 1) * this.rowSize, this.bodyCapPx);
  }

  /** The user's chosen detail-canvas size (px) for the active media type, or
   *  ``null`` when they haven't set one (and the radius-derived default is used). */
  get previewOverride(): number | null {
    const mediaType = this.mediaType();
    const value = mediaType ? this.previewSizeDict[mediaType] : undefined;
    return typeof value === 'number' ? value : null;
  }

  /** Detail-canvas side (px) the popup opens at before any user override:
   *  scaled up from the item's on-canvas mouse-over break-out at the current
   *  main-canvas thumbnail size. */
  private previewDefault(): number {
    return this.hoverThumbRadius() * HOVER_EXTENT_PER_RADIUS * PREVIEW_OVERSIZE;
  }

  /** Target/minimum side (px) of the square preview pane: the user's chosen size
   *  (the top-left buttons) or, unset, a size scaled up from the item's
   *  on-canvas mouse-over break-out at the current main-canvas thumbnail size.
   *  Clamped to the room the visible region leaves. The rendered pane grows to
   *  {@link previewPaneSize} (the full body height) when the member grid is
   *  taller. Zero when there is no preview. */
  get previewSize(): number {
    if (!this.showPreview) return 0;
    const desired = this.previewOverride ?? this.previewDefault();
    // Keep it within the vertical room the region leaves (which already folds in
    // the absolute MAX_PREVIEW_PX cap via ``previewCapPx``).
    return Math.round(Math.max(MIN_PREVIEW_PX, Math.min(desired, this.previewCapPx)));
  }

  /** Step the detail-canvas size to the next ladder rung past the current size in
   *  the given direction and persist it (per media type, under
   *  ``popup_preview_size``), so it becomes the default for future popups of this
   *  type — mirroring how the grid thumbnail-size buttons persist. The settings
   *  effect re-clamps the popup so growing the canvas can't push it off-screen. */
  bumpPreview(delta: 1 | -1): void {
    const mediaType = this.mediaType();
    if (!this.showPreview || !mediaType) return;
    const current = this.previewOverride ?? Math.round(this.previewDefault());
    const next =
      delta > 0
        ? (PREVIEW_SIZE_STEPS.find((s) => s > current) ?? PREVIEW_SIZE_STEPS[PREVIEW_SIZE_STEPS.length - 1])
        : ([...PREVIEW_SIZE_STEPS].reverse().find((s) => s < current) ?? PREVIEW_SIZE_STEPS[0]);
    if (next === this.previewOverride) return;
    const dict = { ...this.previewSizeDict, [mediaType]: next };
    this.settingsState.update({ popup_preview_size: dict } as SettingsUpdate).subscribe();
  }

  /** True when the detail canvas is already at the smallest ladder rung. */
  get atMinPreview(): boolean {
    if (!this.showPreview) return true;
    const current = this.previewOverride ?? Math.round(this.previewDefault());
    return current <= PREVIEW_SIZE_STEPS[0];
  }

  /** True when the detail canvas is already at the largest ladder rung. */
  get atMaxPreview(): boolean {
    if (!this.showPreview) return true;
    const current = this.previewOverride ?? Math.round(this.previewDefault());
    return current >= PREVIEW_SIZE_STEPS[PREVIEW_SIZE_STEPS.length - 1];
  }

  /** True for a one-member bin that has a preview pane: the grid would just
   *  repeat the single item already shown large in the pane, so we drop it and
   *  show only the zoom-in canvas. */
  get previewOnly(): boolean {
    return this.showPreview && this.ids.length === 1;
  }

  /** Height (px) of the grid column: the scrolling grid (capped to the region)
   *  plus the member-count label stacked above it. Zero for a singleton bin
   *  (no grid column — only the preview). */
  get gridColHeight(): number {
    return this.previewOnly ? 0 : this.gridHeight + COUNT_LABEL_HEIGHT;
  }

  /** Height (px) of the body row: tall enough for the grid column (grid capped to
   *  the region, plus its count label), but at least the preview pane's height so
   *  the pane is shown in full. The preview may exceed the grid's cap; both are
   *  region-bounded. */
  get bodyHeight(): number {
    return Math.max(this.gridColHeight, this.previewSize);
  }

  /** Border-box height (px) bound to ``.bin-popup-body``: the content the flex
   *  columns need ({@link bodyHeight}) plus the body's own vertical padding. Since
   *  the body is ``box-sizing: border-box``, the bound ``height`` has to include
   *  that padding for the content area to actually equal {@link bodyHeight};
   *  without it every column's ``height: 100%`` comes up {@link BODY_PADDING_Y}px
   *  short, which shows as a stray scrollbar on the audio member grid and a
   *  slightly non-square preview pane. */
  get bodyOuterHeight(): number {
    return this.bodyHeight + BODY_PADDING_Y;
  }

  /** Side (px) of the *rendered* square preview pane. {@link previewSize} is the
   *  pane's target/minimum side; here we grow it to the full body height so the
   *  pane is always square and the image scales to the largest size that fits.
   *  Without this the pane stays {@link previewSize} wide while the body is
   *  stretched taller by the member grid beside it, leaving the detail image
   *  pinned to a narrow column and floating small in a tall empty space. Zero
   *  when there is no preview. */
  get previewPaneSize(): number {
    return this.showPreview ? this.bodyHeight : 0;
  }

  // --- Dismissal -----------------------------------------------------------

  close(): void {
    this.dismissed.emit();
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.dismissed.emit();
    }
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.dismissed.emit();
  }

  /**
   * Keyboard shortcuts while the popup is open: arrow keys walk the viewed item
   * through the grid, ``+``/``-`` resize the detail image (mirroring the
   * top-left buttons), and Ctrl/Cmd-A selects every item in the bin. The popup
   * owns the keyboard whenever it's open, so this takes precedence over the
   * canvas shortcuts (the browse view suppresses its own while the popup is up).
   * Suppressed while typing or behind a modal ({@link shortcutsBlocked}).
   */
  @HostListener('document:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (shortcutsBlocked()) return;

    // Ctrl/Cmd-A: select every item in this bin (always select, never toggle).
    if ((event.ctrlKey || event.metaKey) && !event.altKey && (event.key === 'a' || event.key === 'A')) {
      event.preventDefault();
      this.selection.addAll(this.ids);
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        this.moveFocus(0, -1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        this.moveFocus(0, 1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        this.moveFocus(-1, 0);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.moveFocus(1, 0);
        break;
      case ' ':
        event.preventDefault();
        this.onPreviewClick();
        break;
      case '+':
      case '=':
        event.preventDefault();
        this.bumpPreview(1);
        break;
      case '-':
      case '_':
        event.preventDefault();
        this.bumpPreview(-1);
        break;
    }
  }

  /**
   * Move the viewed item one grid step: ``dCol`` along a row, ``dRow`` across
   * rows (a row holds {@link columns} items). Updates the preview pane (image /
   * video) and the grid's focus ring, and scrolls the target row into view.
   * No-op for a singleton/preview-only bin, where there's no grid to walk.
   */
  private moveFocus(dCol: number, dRow: number): void {
    if (this.previewOnly || this.ids.length <= 1) return;
    const cur = this.focusIndex();
    const next = Math.max(0, Math.min(this.ids.length - 1, cur + dCol + dRow * this.columns));
    if (next === cur) return;
    this.previewId = this.ids[next];
    this.scrollRowIntoView(next);
    this.cdr.markForCheck();
  }

  /** Index of the viewed item within {@link ids}, falling back to the
   *  representative when nothing is currently viewed. */
  private focusIndex(): number {
    const idx = this.previewId == null ? -1 : this.ids.indexOf(this.previewId);
    return idx >= 0 ? idx : this.repIndex();
  }

  /** True for the viewed item — the one shown in the preview pane and ringed in
   *  the grid as arrow keys walk through it. */
  isFocused(id: number): boolean {
    return this.previewId != null && id === this.previewId;
  }

  /** Scroll the grid the minimum amount so the row holding ``index`` is fully
   *  visible (no-op when it already is). */
  private scrollRowIntoView(index: number): void {
    const vp = this.viewport;
    if (!vp) return;
    const row = Math.floor(index / Math.max(1, this.columns));
    const rowTop = row * this.rowSize;
    const rowBottom = rowTop + this.rowSize;
    const top = vp.measureScrollOffset('top');
    const viewportH = vp.elementRef.nativeElement.clientHeight || this.gridHeight;
    if (rowTop < top) {
      vp.scrollToOffset(rowTop);
    } else if (rowBottom > top + viewportH) {
      vp.scrollToOffset(rowBottom - viewportH);
    }
  }

  // --- Dragging (move the popup by its header) ------------------------------

  /** Begin a drag when the user presses the header (but not the close button or
   *  the view controls, which keep their own click behavior). */
  onHeaderPointerDown(event: PointerEvent): void {
    if (event.button !== 0) return;
    if ((event.target as HTMLElement).closest('button, vt-view-controls')) return;
    event.preventDefault();
    this.dragging = true;
    this.dragged = true;
    this.dragStart = { x: event.clientX, y: event.clientY, left: this.left, top: this.top };
    // Keep receiving moves even if the pointer outruns the header.
    (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
  }

  @HostListener('document:pointermove', ['$event'])
  onPointerMove(event: PointerEvent): void {
    if (!this.dragging) return;
    const dx = event.clientX - this.dragStart.x;
    const dy = event.clientY - this.dragStart.y;
    // Clamp as we go so the popup can't be dragged off the visible region.
    this.clampInto(this.dragStart.left + dx, this.dragStart.top + dy);
  }

  @HostListener('document:pointerup')
  onPointerUp(): void {
    this.dragging = false;
  }

  // --- Selection (click stays open so the user can scroll + multi-select) ---

  isSelected(id: number): boolean {
    return this.selection.has(id);
  }

  /** Tri-state of the select-all control: how many members are selected, as
   *  none / some / all — mirroring the dashboard's master-checkbox states. */
  get selectionState(): 'none' | 'some' | 'all' {
    const total = this.ids.length;
    if (total === 0) return 'none';
    const sel = this.selection.selectedCountIn(this.ids);
    if (sel === 0) return 'none';
    if (sel >= total) return 'all';
    return 'some';
  }

  /** Select every member, or — when all are already selected — clear them.
   *  Matches the dashboard's toggle-all semantics. */
  toggleAll(): void {
    if (this.selectionState === 'all') {
      this.selection.removeAll(this.ids);
    } else {
      this.selection.addAll(this.ids);
    }
  }

  onEntryClick(id: number): void {
    if (this.selection.has(id)) {
      this.selection.remove(id);
    } else {
      this.selection.addAll([id]);
    }
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onEntryClick(id);
    }
  }

  /** True when the item currently shown in the preview pane is selected, so the
   *  large detail image can render the same highlight ring as a grid entry. */
  isPreviewSelected(): boolean {
    return this.previewId != null && this.selection.has(this.previewId);
  }

  /** Toggle selection of the item shown in the preview pane (the hovered grid
   *  item, or the lone member of a singleton bin). This is the only way to select
   *  in a one-member popup, where the grid — and so every other select target —
   *  is dropped; it also lets the user select by clicking the big detail image in
   *  a multi-member popup. */
  onPreviewClick(): void {
    const id = this.previewId;
    if (id != null) this.onEntryClick(id);
  }

  onPreviewKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onPreviewClick();
    }
  }

  // --- Hover: preview the full-res original (image/video) + hear (audio) ----

  onEntryEnter(id: number): void {
    // Follow the hover: paint the hovered item's full-res original into the pane
    // (thumbnail media) and drive the metadata column + focus ring to it (every
    // media type, so audio/text/document read metadata for the item under the
    // cursor even without a preview pane).
    this.previewId = id;
    this.playAudio(id);
  }

  /**
   * Audition ``id`` in the popup's audio element (audio media only), updating
   * the shared now-playing indicator. Shared by the open-time autoplay and the
   * grid-row hover; a no-op when the same clip is already playing so re-entering
   * a row (or a redundant re-summon) doesn't restart it.
   */
  private playAudio(id: number): void {
    if (this.mediaType() !== 'audio') return;
    const src = this.activeContext.mediaUrl(`/api/medias/${id}/audio`);
    if (this.audioSrc === src) return;
    this.audioSrc = src;
    // The autoplay-on-open path may fire before prefetchVisible has hydrated the
    // representative's metadata, so make sure its clip extents are loading (the
    // clip-window handlers read them lazily as they land).
    this.metadataCache.ensureLoaded([id]);
    this.nowPlaying.emit({ mediaId: id, waveUrl: this.thumbnailUrl(id) });
    setTimeout(() => {
      const el = this.audioRef?.nativeElement;
      if (!el) return;
      el.volume = this.volume();
      // Windowed archive-member clips serve the whole file: seek to clip_start
      // and loop within [clip_start, clip_end]. Metadata is read lazily inside
      // the handlers regardless.
      applyClipWindow(el, () => this.metadataCache.get(id));
      el.load();
      el.play().catch(() => {});
    });
  }

  /** Cursor left the grid: stop any hover audio and fall the preview back to the
   *  bin's representative so the pane stays populated. */
  onGridLeave(): void {
    this.stopAudio();
    // Fall the focus back to the bin's representative so the pane and metadata
    // column stay populated (every media type, matching the hover-follow above).
    this.previewId = this.representativeId();
    this.cdr.markForCheck();
  }

  private stopAudio(): void {
    const wasPlaying = this.audioSrc !== '';
    const el = this.audioRef?.nativeElement;
    if (el) {
      clearClipWindow(el);
      el.pause();
      el.currentTime = 0;
    }
    this.audioSrc = '';
    if (wasPlaying) this.nowPlaying.emit(null);
  }

  // --- Names + thumbnails (mirrors the selection panel's treatment) ---------

  name(id: number): string {
    return this.metadataCache.get(id)?.filename || `Clip #${id}`;
  }

  hasThumbnailUrl(id: number): boolean {
    const url = this.thumbnailUrl(id);
    if (this.failedThumbs.has(url)) return false;
    const media = this.metadataCache.get(id);
    return (
      !!media &&
      (media.media_type === 'image' ||
        media.media_type === 'video' ||
        media.media_type === 'document' ||
        media.media_type === 'audio')
    );
  }

  thumbnailUrl(id: number): string {
    return this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
  }

  onThumbnailError(url: string): void {
    if (url) this.failedThumbs.add(url);
  }

  /** Full-res source for the preview pane: the original ``/image`` unless it has
   *  failed for this id, in which case fall back to the thumbnail. Empty when
   *  there is nothing to preview. */
  previewUrl(): string {
    const id = this.previewId;
    if (id == null) return '';
    if (this.failedPreviews.has(id)) return this.thumbnailUrl(id);
    return this.activeContext.mediaUrl(`/api/medias/${id}/image`);
  }

  onPreviewError(): void {
    const id = this.previewId;
    if (id != null && !this.failedPreviews.has(id)) {
      this.failedPreviews.add(id);
      this.cdr.markForCheck();
    }
  }

  placeholderIcon(id: number): string {
    if (this.hasThumbnailUrl(id)) return '';
    const media = this.metadataCache.get(id);
    if (!media) return '□';
    if (media.media_type === 'audio') return '♫';
    if (media.media_type === 'text') return '¶';
    return '□';
  }

  trackById(_index: number, id: number): number {
    return id;
  }

  trackByRow(index: number): number {
    return index;
  }

  // --- Positioning ---------------------------------------------------------

  /** Re-clamp the popup to stay fully on-screen, anchored at its *current*
   *  position. The summon point seeds {@link left}/{@link top} once, when the
   *  popup opens (and on a genuine re-summon — see {@link ngOnChanges}); from
   *  then on every re-clamp (size changes, the settings-driven detail-image
   *  resize, region changes) keeps the popup where it sits rather than snapping
   *  back to the cursor. The computed clamp derives the popup size from known
   *  widths/heights; once it has laid out we additionally measure the real panel
   *  and correct any residual overflow, so the window ends up entirely on-screen
   *  even if the computed height drifts from what actually rendered. */
  private place(): void {
    // Before the panel exists there's nothing to measure/clamp; a later place()
    // call (e.g. from ngAfterViewInit) will run once it's in the DOM. Staying
    // unplaced keeps the popup hidden until then rather than showing it unclamped.
    if (!this.panelRef?.nativeElement) return;
    this.clampInto(this.left, this.top);
    // Flush the freshly-clamped position and the size caps clampInto just set
    // (bodyCapPx/previewCapPx/maxWidthPx) to the DOM. This runs in a bare
    // setTimeout, so under zoneless change detection nothing else schedules a
    // render; without this markForCheck the clamp would never reach the DOM
    // until some unrelated event happened to tick CD. The popup stays hidden
    // (``placed`` is still false) — the actual reveal happens in nudgeOnScreen,
    // on the next frame, once the *rendered* panel has been measured and any
    // residual overflow corrected. Revealing here, before that measurement, was
    // what let the popup flash at the computed clamp and then jump to the
    // corrected spot.
    this.cdr.markForCheck();
    requestAnimationFrame(() => this.nudgeOnScreen());
  }

  /** Measure the rendered panel and slide it so its real rect sits inside the
   *  visible region (canvas ∩ viewport), keeping the bottom (and the detail
   *  image's bottom with it) on-screen. Purely corrective: a no-op when the
   *  computed clamp already fits. */
  private nudgeOnScreen(): void {
    const panel = this.panelRef?.nativeElement;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const b = this.bounds();
    const regionLeft = Math.max(b ? b.left : 0, 0);
    const regionTop = Math.max(b ? b.top : 0, 0);
    const regionRight = Math.min(b ? b.right : window.innerWidth, window.innerWidth);
    const regionBottom = Math.min(b ? b.bottom : window.innerHeight, window.innerHeight);
    // Correct off the panel's measured *size* applied to its intended position
    // (this.left/this.top), not the panel's currently-rendered absolute rect: the
    // clamp from place() may not have painted yet, so rect.right/rect.bottom can
    // still describe the previous spot. Using this.left + width keeps the
    // correction right regardless of paint timing.
    const width = rect.width;
    const height = rect.height;
    let l = this.left;
    let t = this.top;
    // Pull in from the far edges first, then guarantee the near edges, so a popup
    // larger than the region pins to top-left (losing the far edge, not the near).
    if (this.left + width > regionRight - EDGE_MARGIN) {
      l -= this.left + width - (regionRight - EDGE_MARGIN);
    }
    if (this.top + height > regionBottom - EDGE_MARGIN) {
      t -= this.top + height - (regionBottom - EDGE_MARGIN);
    }
    l = Math.max(regionLeft + EDGE_MARGIN, l);
    t = Math.max(regionTop + EDGE_MARGIN, t);
    const moved = l !== this.left || t !== this.top;
    this.left = l;
    this.top = t;
    // Reveal now that the panel is measured and corrected, so the first painted
    // frame is already at the final on-screen spot. Gate the first reveal on the
    // settings being loaded: the popup's size (grid thumbnail size, whether the
    // metadata column shows, the preview-pane size) all come from settings, so
    // revealing before they arrive would show the popup at default sizes and then
    // re-clamp/move it when they land. The settings effect calls place() again
    // once settings resolve, which reveals then. Browse only ever mounts with
    // settings present, so this never strands the popup permanently hidden.
    const settingsReady = this.settingsState.settingsSignal() != null;
    if (!this.placed) {
      if (settingsReady) {
        this.placed = true;
        this.cdr.markForCheck();
      }
    } else if (moved) {
      this.cdr.markForCheck();
    }
  }

  /** Clamp ``(desiredLeft, desiredTop)`` so the *whole* popup sits inside the
   *  visible part of the canvas — the canvas rect intersected with the viewport
   *  — so it never spills off an edge, onto the side panel, or below the bottom
   *  of the window. Sizing is derived from known quantities (the column widths
   *  and the header + body heights) rather than a possibly-mid-layout
   *  measurement, so the clamp is correct even before the virtualized body has
   *  settled. When the region is too short to hold the full popup, the body is
   *  capped (and scrolls internally) so the popup still fits top-to-bottom. */
  private clampInto(desiredLeft: number, desiredTop: number): void {
    const panel = this.panelRef?.nativeElement;
    if (!panel) return;
    const b = this.bounds();
    // Visible region = canvas rect clipped to the viewport. Clipping to the
    // window is what keeps the popup fully on-screen when the canvas extends
    // past the bottom/right edges of the window.
    const regionLeft = Math.max(b ? b.left : 0, 0);
    const regionTop = Math.max(b ? b.top : 0, 0);
    const regionRight = Math.min(b ? b.right : window.innerWidth, window.innerWidth);
    const regionBottom = Math.min(b ? b.bottom : window.innerHeight, window.innerHeight);
    // The header is always rendered (not virtualized), so its height is reliable
    // immediately; fall back to a sane default before the view exists.
    const headerH = this.headerRef?.nativeElement.getBoundingClientRect().height ?? 37;
    // Squeeze the scrolling body to whatever vertical room the region leaves, so
    // a short canvas can't make the popup taller than what's visible. The caps
    // below bound the body *content* height ({@link bodyHeight}), so the room they
    // work against must first give up everything the popup spends around that
    // content — the body's own vertical padding and the popup's top/bottom border
    // — or a cap-filling body (a large audio bin's grid, which pins right at the
    // cap) renders BODY_PADDING_Y + 2*POPUP_BORDER taller than the region and its
    // bottom edge slips off-screen. Image previews are radius-sized and usually
    // sit below the cap, so they had slack that hid this; audio has none.
    const chromeY = BODY_PADDING_Y + 2 * POPUP_BORDER;
    const regionRoom = regionBottom - regionTop - 2 * EDGE_MARGIN - headerH - chromeY;
    // The grid column also carries the member-count label above the scroll, so
    // its cap leaves that label room; the scroll then fills what's left.
    const gridRoom = regionRoom - COUNT_LABEL_HEIGHT;
    this.bodyCapPx = Math.max(MIN_BODY_PX, Math.min(MAX_BODY_PX, gridRoom));
    // The preview gets its own, larger cap: it may grow past the grid's body cap,
    // bounded only by the visible region and the absolute MAX_PREVIEW_PX.
    this.previewCapPx = Math.max(MIN_PREVIEW_PX, Math.min(MAX_PREVIEW_PX, regionRoom));
    // Likewise shrink the overall width to fit a narrow region.
    this.maxWidthPx = Math.max(0, regionRight - regionLeft - 2 * EDGE_MARGIN);
    // A singleton bin drops the grid column and shows only the preview pane.
    const gridW = this.previewOnly ? 0 : GRID_COLUMN_WIDTH;
    const paneW = this.previewPaneSize;
    const previewW = paneW ? paneW + (gridW ? COLUMN_GAP : 0) : 0;
    // The optional metadata column sits left of the preview, adding its width
    // plus a gap when shown.
    const metaW = this.showMetadataColumn ? METADATA_COLUMN_WIDTH + COLUMN_GAP : 0;
    // The columns span the body's content box; the body adds its own horizontal
    // padding (BODY_PADDING each side) and the popup a 1px border, so the real
    // rendered width is that much wider than the columns alone. Folding it in
    // keeps the computed clamp exact instead of leaning on nudgeOnScreen to catch
    // the residual — the no-preview (audio) layout has no preview pane sized to
    // absorb the slop.
    const width = Math.min(
      metaW + previewW + gridW + 2 * BODY_PADDING + 2 * POPUP_BORDER,
      this.maxWidthPx,
    );
    // ``bodyOuterHeight`` already folds the body's vertical padding into the
    // content height; add the popup's own top/bottom border for the full extent.
    const height = headerH + this.bodyOuterHeight + 2 * POPUP_BORDER;
    let l = desiredLeft;
    let t = desiredTop;
    if (l + width + EDGE_MARGIN > regionRight) {
      l = regionRight - width - EDGE_MARGIN;
    }
    if (t + height + EDGE_MARGIN > regionBottom) {
      t = regionBottom - height - EDGE_MARGIN;
    }
    // Never push the top-left off the opposite edge (popup larger than region).
    this.left = Math.max(regionLeft + EDGE_MARGIN, l);
    this.top = Math.max(regionTop + EDGE_MARGIN, t);
    this.cdr.markForCheck();
  }

  /** Prefetch metadata for the items around the visible window of the grid. */
  private prefetchVisible(): void {
    if (this.ids.length === 0) return;
    const cols = this.columns;
    const vp = this.viewport;
    if (!vp) {
      const window = Math.ceil(MAX_BODY_PX / this.rowSize) * cols + PREFETCH_BUFFER;
      this.metadataCache.ensureLoaded(this.ids.slice(0, window));
      return;
    }
    const startRow = Math.floor(vp.measureScrollOffset('top') / this.rowSize);
    const visibleRows = Math.ceil(
      (vp.elementRef.nativeElement.clientHeight || this.gridHeight) / this.rowSize,
    );
    const from = Math.max(0, (startRow - Math.ceil(PREFETCH_BUFFER / cols)) * cols);
    const to = Math.min(this.ids.length, (startRow + visibleRows) * cols + PREFETCH_BUFFER);
    this.metadataCache.ensureLoaded(this.ids.slice(from, to));
  }
}
