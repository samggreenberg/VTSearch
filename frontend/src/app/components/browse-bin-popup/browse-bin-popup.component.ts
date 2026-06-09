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

/** Row stride (px) for the virtualized list; must match ``.bin-popup-entry``. */
const ITEM_HEIGHT = 36;
/** Most rows shown before the popup caps its height and scrolls. */
const MAX_VISIBLE_ROWS = 10;
/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/** Gap (px) kept between the popup and the canvas edge when clamping. */
const EDGE_MARGIN = 8;

/**
 * The bin popup: a small floating, virtualized list of the media items in the
 * bin the user right-clicked on the VTSBrowse canvas. It replaces the old
 * right-click action menu — instead of menu commands, it shows the bin's
 * members rendered like the right-panel selection list, so the user can scroll
 * the bin, hear each item on hover (audio), and click to add/remove it from the
 * canvas selection. This is how you reach the individual items folded into a
 * dense bin without zooming all the way in.
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
  imports: [CommonModule, ScrollingModule],
  templateUrl: './browse-bin-popup.component.html',
  styleUrl: './browse-bin-popup.component.scss',
})
export class BrowseBinPopupComponent implements AfterViewInit, OnChanges, OnDestroy {
  /** Member media ids of the bin the popup was summoned over. */
  @Input() memberIds: number[] = [];
  /** Active dataset media type, used to decide hover-to-hear and placeholders. */
  @Input() mediaType = '';
  /** Viewport anchor (clientX/clientY) the popup opens at, then clamps inward. */
  @Input() x = 0;
  @Input() y = 0;
  /** The canvas's bounding rect (viewport coords); the popup is clamped inside
   *  it so it stays on the canvas. Null falls back to the full viewport. */
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

  readonly itemHeight = ITEM_HEIGHT;

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
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['memberIds']) {
      this.ids = this.memberIds ?? [];
      this.stopAudio();
      // A fresh bin: jump the list back to the top and prefetch its first window.
      this.viewport?.scrollToIndex(0);
      this.prefetchVisible();
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
    );
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

  /** Height (px) the list takes: just enough for its rows, capped then scrolled. */
  get listHeight(): number {
    return Math.min(Math.max(this.ids.length, 1), MAX_VISIBLE_ROWS) * this.itemHeight;
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

  // --- Positioning ---------------------------------------------------------

  /** Clamp the popup inside the canvas (or the viewport, when no bounds are
   *  given) so it never spills off an edge or onto the side panel. */
  private place(): void {
    const panel = this.panelRef?.nativeElement;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const b = this.bounds;
    const minLeft = b ? b.left + EDGE_MARGIN : EDGE_MARGIN;
    const minTop = b ? b.top + EDGE_MARGIN : EDGE_MARGIN;
    const maxRight = b ? b.right : window.innerWidth;
    const maxBottom = b ? b.bottom : window.innerHeight;
    let l = this.x;
    let t = this.y;
    if (l + rect.width + EDGE_MARGIN > maxRight) {
      l = maxRight - rect.width - EDGE_MARGIN;
    }
    if (t + rect.height + EDGE_MARGIN > maxBottom) {
      t = maxBottom - rect.height - EDGE_MARGIN;
    }
    // Never push the top-left off the opposite edge (popup larger than canvas).
    this.left = Math.max(minLeft, l);
    this.top = Math.max(minTop, t);
    this.cdr.markForCheck();
  }

  /** Prefetch metadata for the rows around the visible window of the list. */
  private prefetchVisible(): void {
    if (this.ids.length === 0) return;
    const vp = this.viewport;
    if (!vp) {
      this.metadataCache.ensureLoaded(this.ids.slice(0, MAX_VISIBLE_ROWS + PREFETCH_BUFFER));
      return;
    }
    const start = Math.floor(vp.measureScrollOffset('top') / this.itemHeight);
    const visible = Math.ceil((vp.elementRef.nativeElement.clientHeight || this.listHeight) / this.itemHeight);
    const from = Math.max(0, start - PREFETCH_BUFFER);
    const to = Math.min(this.ids.length, start + visible + PREFETCH_BUFFER);
    this.metadataCache.ensureLoaded(this.ids.slice(from, to));
  }
}
