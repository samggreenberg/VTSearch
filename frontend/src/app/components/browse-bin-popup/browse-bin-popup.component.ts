import {
  AfterViewInit,
  ChangeDetectorRef,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { usesThumbnails } from '../browse-canvas/hex-render.util';

/** Vertical room (px) reserved under a grid thumbnail for its truncated name. */
const GRID_LABEL_HEIGHT = 18;
/** Gap (px) between grid cells (and grid rows); matches ``--space-2xs``-ish. */
const GRID_GAP = 4;
/** Width (px) available to lay out cells inside the popup's scroll column (≈ its
 *  width minus padding and the scrollbar). Columns are derived from this. */
const GRID_CONTENT_WIDTH = 256;
/** Width (px) of the scrolling grid column; mirrors the historic popup width. */
const GRID_COLUMN_WIDTH = 280;
/** Tallest the scrolling body grows before it caps and scrolls internally. */
const MAX_BODY_PX = 400;
/** Shortest the scrolling body is ever squeezed to when the visible region is
 *  too short to fit the full popup; below this it just scrolls internally. */
const MIN_BODY_PX = 80;
/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/** Gap (px) kept between the popup and the visible edge when clamping. */
const EDGE_MARGIN = 8;
/** Gap (px) between the preview pane and the scroll column. */
const PREVIEW_GAP = 8;
/** Smallest the preview pane is squeezed to in a tight region. */
const MIN_PREVIEW_PX = 96;
/** Largest the preview pane grows, regardless of icon size, so an extreme
 *  thumbnail-size setting can't make the popup absurdly large. */
const MAX_PREVIEW_PX = 520;
/**
 * How much larger the preview pane is than the item's on-canvas mouse-over size.
 * The brief: "50% larger than the mouse-over size when we hover in the main
 * canvas." 1.5 == +50%.
 */
const PREVIEW_OVERSIZE = 1.5;
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
 * popup of that media type. The in-header {@link ViewControlsComponent} writes
 * that setting (its Grid/List toggle is hidden here) and this component re-reads
 * it from {@link SettingsStateService}, keyed by the active dataset's media type.
 *
 * It shares the {@link BrowseSelectionService} instance provided by the browse
 * view, so toggling an item here is the same selection the canvas rings and the
 * selection panel reflect; selected members render highlighted and update live.
 * Names/thumbnails resolve lazily through {@link MediaMetadataCacheService}
 * (the browse view never loads the full media list), prefetched around the
 * visible window so large bins stay responsive.
 */
@Component({
  selector: 'vt-browse-bin-popup',
  standalone: true,
  imports: [CommonModule, ScrollingModule, ViewControlsComponent],
  templateUrl: './browse-bin-popup.component.html',
  styleUrl: './browse-bin-popup.component.scss',
})
export class BrowseBinPopupComponent implements AfterViewInit, OnChanges, OnDestroy {
  /** Member media ids of the bin the popup was summoned over. */
  @Input() memberIds: number[] = [];
  /** Active dataset media type, used for the view prefs, hover-to-hear, and placeholders. */
  @Input() mediaType = '';
  /** Viewport anchor (clientX/clientY) the popup opens at, then clamps inward. */
  @Input() x = 0;
  @Input() y = 0;
  /** The canvas's bounding rect (viewport coords); the popup is clamped to the
   *  on-screen part of it so it stays fully visible. Null falls back to the
   *  full viewport. */
  @Input() bounds: DOMRect | null = null;
  /** Current on-screen bin radius (CSS px) of the main canvas, i.e. the radius
   *  the hovered thumbnail breaks out from. The preview pane is sized relative
   *  to this so it tracks the main-canvas thumbnail-size setting. */
  @Input() hoverThumbRadius = 28;

  /** Emitted when the popup should close (outside click, Escape, or the X). */
  @Output() dismissed = new EventEmitter<void>();

  @ViewChild('panel') private panelRef?: ElementRef<HTMLElement>;
  @ViewChild('header') private headerRef?: ElementRef<HTMLElement>;
  @ViewChild(CdkVirtualScrollViewport) private viewport?: CdkVirtualScrollViewport;
  @ViewChild('audioEl') private audioRef?: ElementRef<HTMLAudioElement>;

  /** Clamped on-screen position; starts at the anchor and is nudged inward. */
  left = 0;
  top = 0;
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

  constructor(
    private host: ElementRef<HTMLElement>,
    private selection: BrowseSelectionService,
    private metadataCache: MediaMetadataCacheService,
    private activeContext: ActiveContextService,
    private settingsState: SettingsStateService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['memberIds']) {
      this.ids = this.memberIds ?? [];
      this.stopAudio();
      // Open on the bin's representative so the pane is never blank (a singleton
      // therefore lands straight on a large high-res view).
      this.previewId = this.ids.length > 0 ? this.ids[0] : null;
      // A fresh bin is a fresh popup: forget any drag from the previous one so
      // it re-anchors to the new summon point.
      this.dragged = false;
      this.rebuildRows();
      // A fresh bin: jump the list back to the top and prefetch its first window.
      this.viewport?.scrollToIndex(0);
      this.prefetchVisible();
    }
    if (changes['mediaType']) {
      // A new media type may carry a different remembered thumbnail size.
      this.applyViewPrefs();
    }
    if (
      changes['x'] ||
      changes['y'] ||
      changes['bounds'] ||
      changes['memberIds'] ||
      changes['hoverThumbRadius']
    ) {
      // Re-anchor to the summon point, unless the user has dragged the popup —
      // then keep their spot (``place`` re-clamps it back on-screen if needed).
      if (!this.dragged) {
        this.left = this.x;
        this.top = this.y;
      }
      // Measure + clamp after the new content lays out.
      setTimeout(() => this.place());
    }
  }

  ngAfterViewInit(): void {
    this.subs.push(
      // Names/thumbnails arrive asynchronously; repaint the visible rows.
      this.metadataCache.version$.subscribe(() => this.cdr.markForCheck()),
      // A selection change anywhere (here, the canvas, the panel) re-highlights.
      this.selection.changed$.subscribe(() => this.cdr.markForCheck()),
      // Re-read the popup's thumbnail size whenever settings change (this is how
      // the in-header size buttons take effect, and how a change on one popup
      // becomes the default for every future popup of this media type).
      this.settingsState.settings$.subscribe((settings) => {
        if (!settings) return;
        this.gridSizeDict = (settings.grid_icon_size_popup as Record<string, string>) ?? {};
        this.applyViewPrefs();
      }),
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

  /** True for media types that carry real visual thumbnails (image / video):
   *  the ones that magnify on the main canvas and are worth a large preview. */
  get showPreview(): boolean {
    return usesThumbnails(this.mediaType);
  }

  /** Pull the remembered thumbnail size for the active media type, rechunk the
   *  rows, and re-clamp (the body height may have changed). */
  private applyViewPrefs(): void {
    const prevGoal = this.gridGoalWidth;
    this.gridGoalWidth = iconSizeToGoalWidth(
      (this.mediaType && this.gridSizeDict[this.mediaType]) || 'M',
    );
    if (this.gridGoalWidth !== prevGoal) {
      this.rebuildRows();
      // The row stride changed; let the virtual viewport remeasure, then clamp.
      setTimeout(() => {
        this.viewport?.checkViewportSize();
        this.place();
      });
    }
    this.cdr.markForCheck();
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

  /** Pixel stride of one virtual grid row (a row of cells plus its labels). */
  get rowSize(): number {
    return this.gridGoalWidth + GRID_LABEL_HEIGHT + GRID_GAP;
  }

  /** Height (px) the grid takes: just enough for its rows, capped (to the room
   *  the visible region leaves, see {@link bodyCapPx}) then scrolled. */
  get gridHeight(): number {
    return Math.min(Math.max(this.rows.length, 1) * this.rowSize, this.bodyCapPx);
  }

  /** Side (px) of the square preview pane: 50% larger than the item's on-canvas
   *  mouse-over break-out at the current main-canvas thumbnail size, clamped to
   *  the room the visible region leaves. Zero when there is no preview. */
  get previewSize(): number {
    if (!this.showPreview) return 0;
    const desired = this.hoverThumbRadius * HOVER_EXTENT_PER_RADIUS * PREVIEW_OVERSIZE;
    // Keep it within the vertical room the region leaves and a sane absolute cap.
    return Math.round(
      Math.max(MIN_PREVIEW_PX, Math.min(desired, MAX_PREVIEW_PX, this.bodyCapPx)),
    );
  }

  /** Height (px) of the body row: tall enough for the grid, but at least the
   *  preview pane's height so the pane is shown in full. Capped to the region. */
  get bodyHeight(): number {
    return Math.min(this.bodyCapPx, Math.max(this.gridHeight, this.previewSize));
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

  // --- Hover: preview the full-res original (image/video) + hear (audio) ----

  onEntryEnter(id: number): void {
    // Thumbnail media: paint the hovered item's full-res original into the pane.
    if (this.showPreview) this.previewId = id;
    if (this.mediaType !== 'audio') return;
    const src = this.activeContext.mediaUrl(`/api/medias/${id}/audio`);
    if (this.audioSrc === src) return;
    this.audioSrc = src;
    setTimeout(() => {
      const el = this.audioRef?.nativeElement;
      if (!el) return;
      el.loop = true;
      el.load();
      el.play().catch(() => {});
    });
  }

  /** Cursor left the grid: stop any hover audio and fall the preview back to the
   *  bin's representative so the pane stays populated. */
  onGridLeave(): void {
    this.stopAudio();
    if (this.showPreview) this.previewId = this.ids.length > 0 ? this.ids[0] : null;
  }

  private stopAudio(): void {
    const el = this.audioRef?.nativeElement;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    this.audioSrc = '';
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

  /** Re-clamp the popup, anchoring at the summon point unless the user has
   *  dragged it (then keep their spot, just nudged back on-screen if needed). */
  private place(): void {
    this.clampInto(this.dragged ? this.left : this.x, this.dragged ? this.top : this.y);
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
    const b = this.bounds;
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
    // a short canvas can't make the popup taller than what's visible.
    this.bodyCapPx = Math.max(
      MIN_BODY_PX,
      Math.min(MAX_BODY_PX, regionBottom - regionTop - 2 * EDGE_MARGIN - headerH),
    );
    // Likewise shrink the overall width to fit a narrow region.
    this.maxWidthPx = Math.max(0, regionRight - regionLeft - 2 * EDGE_MARGIN);
    const previewW = this.previewSize ? this.previewSize + PREVIEW_GAP : 0;
    const width = Math.min(previewW + GRID_COLUMN_WIDTH, this.maxWidthPx);
    const height = headerH + this.bodyHeight;
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
