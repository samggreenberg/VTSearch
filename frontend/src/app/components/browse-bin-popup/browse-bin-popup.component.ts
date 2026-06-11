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

/** Row stride (px) for the virtualized list mode; matches ``.bin-popup-entry``. */
const LIST_ROW_HEIGHT = 36;
/** Vertical room (px) reserved under a grid thumbnail for its truncated name. */
const GRID_LABEL_HEIGHT = 18;
/** Gap (px) between grid cells (and grid rows); matches ``--space-2xs``-ish. */
const GRID_GAP = 4;
/** Width (px) available to lay out cells inside the popup body (≈ popup width
 *  minus padding and the scrollbar). Columns are derived from this. */
const GRID_CONTENT_WIDTH = 256;
/** Tallest the scrolling body grows before it caps and scrolls internally. */
const MAX_BODY_PX = 400;
/** Shortest the scrolling body is ever squeezed to when the visible region is
 *  too short to fit the full popup; below this it just scrolls internally. */
const MIN_BODY_PX = 80;
/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/** Gap (px) kept between the popup and the visible edge when clamping. */
const EDGE_MARGIN = 8;
/** Default popup width (px); mirrors ``width`` in the component SCSS. */
const POPUP_WIDTH = 280;

/**
 * The bin popup: a small floating panel showing the media items in the bin the
 * user right-clicked on the VTSBrowse canvas. It is a miniature knock-off of
 * the Find right panel — the same List/Grid + thumbnail-size controls (via
 * {@link ViewControlsComponent}) sit in its header, and the body renders the
 * bin's members in whichever mode is chosen. This is how you reach the
 * individual items folded into a dense bin without zooming all the way in.
 *
 * The view-mode + size choice is remembered per media type under the
 * ``view_mode_popup`` / ``grid_icon_size_popup`` settings (independent of the
 * left/right panels), so tuning the popup while browsing one bin becomes the
 * default for every future popup of that media type. The controls write those
 * settings and this component re-reads them from {@link SettingsStateService},
 * keyed by the active dataset's media type.
 *
 * It shares the {@link BrowseSelectionService} instance provided by the browse
 * view, so toggling an item here is the same selection the canvas rings and the
 * selection panel reflect; selected members render highlighted and update live.
 * Names/thumbnails resolve lazily through {@link MediaMetadataCacheService}
 * (the browse view never loads the full media list), prefetched around the
 * visible window so large bins stay responsive in either mode.
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
  maxWidthPx = POPUP_WIDTH;
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

  /** Per-media-type view prefs, mirrored from settings. */
  viewMode: 'grid' | 'list' = 'grid';
  gridGoalWidth = 80;

  /** Number of columns the grid lays out; always 1 in list mode. */
  columns = 1;
  /** Member ids chunked into rows of {@link columns}; the virtual list's data. */
  rows: number[][] = [];

  /** Currently-playing hover audio source, so re-entering the same row is a no-op. */
  audioSrc = '';

  private readonly failedThumbs = new Set<string>();
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
      // A fresh bin is a fresh popup: forget any drag from the previous one so
      // it re-anchors to the new summon point.
      this.dragged = false;
      this.rebuildRows();
      // A fresh bin: jump the list back to the top and prefetch its first window.
      this.viewport?.scrollToIndex(0);
      this.prefetchVisible();
    }
    if (changes['mediaType']) {
      // A new media type may carry a different remembered view mode/size.
      this.applyViewPrefs();
    }
    if (changes['x'] || changes['y'] || changes['bounds'] || changes['memberIds']) {
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
      // Re-read the popup's view mode + size whenever settings change (this is
      // how the in-header controls take effect, and how a change on one popup
      // becomes the default for every future popup of this media type).
      this.settingsState.settings$.subscribe((settings) => {
        if (!settings) return;
        this.viewModeDict = (settings.view_mode_popup as Record<string, 'grid' | 'list'>) ?? {};
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

  private viewModeDict: Record<string, 'grid' | 'list'> = {};
  private gridSizeDict: Record<string, string> = {};

  /** Pull the remembered view mode + thumbnail size for the active media type,
   *  rechunk the rows, and re-clamp (the body height may have changed). */
  private applyViewPrefs(): void {
    const prevMode = this.viewMode;
    const prevGoal = this.gridGoalWidth;
    this.viewMode = this.mediaType ? (this.viewModeDict[this.mediaType] ?? 'grid') : 'grid';
    this.gridGoalWidth = iconSizeToGoalWidth(
      (this.mediaType && this.gridSizeDict[this.mediaType]) || 'M',
    );
    if (this.viewMode !== prevMode || this.gridGoalWidth !== prevGoal) {
      this.rebuildRows();
      // The row stride changed; let the virtual viewport remeasure, then clamp.
      setTimeout(() => {
        this.viewport?.checkViewportSize();
        this.place();
      });
    }
    this.cdr.markForCheck();
  }

  /** Recompute the column count + row chunking for the current mode/size. */
  private rebuildRows(): void {
    this.columns =
      this.viewMode === 'grid'
        ? Math.max(1, Math.floor((GRID_CONTENT_WIDTH + GRID_GAP) / (this.gridGoalWidth + GRID_GAP)))
        : 1;
    const cols = this.columns;
    const rows: number[][] = [];
    for (let i = 0; i < this.ids.length; i += cols) {
      rows.push(this.ids.slice(i, i + cols));
    }
    this.rows = rows;
  }

  /** Pixel stride of one virtual row (a list entry, or a grid row of cells). */
  get rowSize(): number {
    return this.viewMode === 'grid'
      ? this.gridGoalWidth + GRID_LABEL_HEIGHT + GRID_GAP
      : LIST_ROW_HEIGHT;
  }

  /** Height (px) the body takes: just enough for its rows, capped (to the room
   *  the visible region leaves, see {@link bodyCapPx}) then scrolled. */
  get bodyHeight(): number {
    return Math.min(Math.max(this.rows.length, 1) * this.rowSize, this.bodyCapPx);
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
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

  // --- Hover-to-hear (audio only; other types just highlight, via CSS) ------

  onEntryEnter(id: number): void {
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

  onListLeave(): void {
    this.stopAudio();
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
   *  of the window. Sizing is derived from known quantities (the fixed width and
   *  the header + body heights) rather than a possibly-mid-layout measurement,
   *  so the clamp is correct even before the virtualized body has settled. When
   *  the region is too short to hold the full popup, the body is capped (and
   *  scrolls internally) so the popup still fits top-to-bottom. */
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
    // Likewise shrink the width to fit a narrow region.
    this.maxWidthPx = Math.max(0, regionRight - regionLeft - 2 * EDGE_MARGIN);
    const width = Math.min(POPUP_WIDTH, this.maxWidthPx);
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

  /** Prefetch metadata for the items around the visible window of the body. */
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
      (vp.elementRef.nativeElement.clientHeight || this.bodyHeight) / this.rowSize,
    );
    const from = Math.max(0, (startRow - Math.ceil(PREFETCH_BUFFER / cols)) * cols);
    const to = Math.min(this.ids.length, (startRow + visibleRows) * cols + PREFETCH_BUFFER);
    this.metadataCache.ensureLoaded(this.ids.slice(from, to));
  }
}
