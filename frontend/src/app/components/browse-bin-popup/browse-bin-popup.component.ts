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
/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/** Gap (px) kept between the popup and the visible edge when clamping. */
const EDGE_MARGIN = 8;

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
  @ViewChild(CdkVirtualScrollViewport) private viewport?: CdkVirtualScrollViewport;
  @ViewChild('audioEl') private audioRef?: ElementRef<HTMLAudioElement>;

  /** Clamped on-screen position; starts at the anchor and is nudged inward. */
  left = 0;
  top = 0;

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
      this.left = this.x;
      this.top = this.y;
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

  /** Height (px) the body takes: just enough for its rows, capped then scrolled. */
  get bodyHeight(): number {
    return Math.min(Math.max(this.rows.length, 1) * this.rowSize, MAX_BODY_PX);
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

  /** Clamp the popup inside the *visible* part of the canvas — the canvas rect
   *  intersected with the viewport — so it never spills off an edge, onto the
   *  side panel, or below the bottom of the window. */
  private place(): void {
    const panel = this.panelRef?.nativeElement;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const b = this.bounds;
    // Visible region = canvas rect clipped to the viewport. Clipping to the
    // window is what keeps the popup fully on-screen when the canvas extends
    // past the bottom/right edges of the window.
    const regionLeft = Math.max(b ? b.left : 0, 0);
    const regionTop = Math.max(b ? b.top : 0, 0);
    const regionRight = Math.min(b ? b.right : window.innerWidth, window.innerWidth);
    const regionBottom = Math.min(b ? b.bottom : window.innerHeight, window.innerHeight);
    let l = this.x;
    let t = this.y;
    if (l + rect.width + EDGE_MARGIN > regionRight) {
      l = regionRight - rect.width - EDGE_MARGIN;
    }
    if (t + rect.height + EDGE_MARGIN > regionBottom) {
      t = regionBottom - rect.height - EDGE_MARGIN;
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
