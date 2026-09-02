import { ChangeDetectionStrategy, Component, ElementRef, OnDestroy, computed, effect, inject, input, output, untracked, viewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { IconComponent } from '../icon/icon.component';
import { binGridRowSize } from '../../utils/bin-grid-metrics';

/** Extra rows of metadata prefetched beyond the visible window. */
const PREFETCH_BUFFER = 50;
/**
 * Viewport height (px) assumed while the real one is unmeasurable — before the
 * virtual viewport exists, or under jsdom, where ``clientHeight`` is always 0.
 * Only ever used to size a prefetch window or centre a scroll, so an
 * approximation is harmless; it mirrors the popup's own body cap so the guessed
 * window matches what the floating presentation actually shows.
 */
const FALLBACK_VIEWPORT_PX = 400;

/**
 * The bin's member items as a virtualized thumbnail grid, with the select-all
 * control, member count and thumbnail-size buttons on a header row above it.
 *
 * Split out of ``BrowseBinPopupComponent``, which is now purely the *shell*
 * around this: the floating window's drag / summon / clamp machinery and the
 * detail pane + metadata column. Everything to do with laying the members out
 * and driving the virtual viewport — column chunking, scroll-driven metadata
 * prefetch, centring a row, moving DOM focus onto a virtualized entry — lives
 * here, and is reached from the shell through the three imperative methods at
 * the bottom of this class ({@link prefetchVisible}, {@link centreOn}, {@link
 * revealAndFocus}).
 *
 * Both of the shell's presentations render this same grid; ``docked`` only
 * changes its cell tracks (fixed-width columns packed left, so widening the
 * panel reveals a whole new column rather than stretching the existing ones)
 * and enables the empty hint. The class names keep the ``bin-popup-`` prefix
 * they had inside the shell: the DOM contract (notably the
 * ``[data-entry-id]`` entry lookup) and the stylesheet moved across unchanged.
 *
 * **Reactivity.** Two of the values this template renders come from services
 * with no signal of their own per item — selection membership
 * ({@link BrowseSelectionService.has}) and cached names/thumbnails
 * ({@link MediaMetadataCacheService.get}). Every method that reads one also
 * reads that service's *version* signal first, so the read registers as a
 * dependency of this view and a mutation anywhere repaints the grid without a
 * ``markForCheck()``. See ``docs/FRONTEND.md`` §5.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-bin-member-grid',
  standalone: true,
  imports: [CommonModule, ScrollingModule, ViewControlsComponent, IconComponent],
  templateUrl: './browse-bin-member-grid.component.html',
  styleUrl: './browse-bin-member-grid.component.scss',
  host: { '[class.docked]': 'docked()' },
})
export class BrowseBinMemberGridComponent implements OnDestroy {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly selection = inject(BrowseSelectionService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly mediaTypeCaps = inject(MediaTypeCapabilityService);

  /** Every member of the bin, in bin order. Drives the count, the select-all
   *  tri-state and the prefetch window; {@link rows} is what actually renders. */
  readonly ids = input<number[]>([]);
  /** {@link ids} chunked into rows of {@link columns} — the virtual list's data.
   *  The shell chunks rather than this component, because the same chunking
   *  feeds its clamp math; a docked singleton is passed an empty list so the
   *  grid area stays allocated but blank. */
  readonly rows = input<number[][]>([]);
  /** Cells per row, mirroring how {@link rows} was chunked. */
  readonly columns = input(1);
  /** Target width (px) of one thumbnail cell. */
  readonly cellWidth = input(80);
  /** Active media type, for the thumbnail-size controls and waveform rendering. */
  readonly mediaType = input('');
  /** Docked presentation: fixed-width cell tracks and an empty-state hint. */
  readonly docked = input(false);
  /** The viewed item — shown in the shell's detail pane and ringed here. */
  readonly focusedId = input<number | null>(null);

  /** An entry was activated (click, or Enter/Space on the focused entry). The
   *  shell owns the selection toggle, since its detail pane shares it. */
  readonly entryClick = output<number>();
  /** DOM focus landed on an entry (arrow-walk, Tab, or click). */
  readonly entryFocus = output<number>();
  /** The cursor entered an entry (drives the shell's preview + hover audio). */
  readonly entryEnter = output<number>();
  /** The cursor left the grid entirely. */
  readonly gridLeave = output<void>();

  private readonly viewport = viewChild(CdkVirtualScrollViewport);
  private scrollSub: Subscription | null = null;
  /** The viewport instance whose ``scrolledIndexChange`` is currently subscribed. */
  private scrollSubscribedViewport: CdkVirtualScrollViewport | null = null;

  private readonly failedThumbs = new Set<string>();

  /** Pixel stride of one virtual row. */
  readonly rowSize = computed(() => binGridRowSize(this.cellWidth()));

  /** True when these thumbnails are audio waveforms: theme-agnostic alpha-mask
   *  PNGs (issue #2369) rendered as a CSS mask over the accent colour rather
   *  than a plain <img>, so they recolour with the live theme. */
  readonly isAudioWaveform = computed(() => this.mediaType() === 'audio');

  constructor() {
    // (Re-)subscribe the scroll-driven prefetch whenever the virtual viewport
    // instance changes — it lives behind the template's ``@if``, and the shell
    // is reused across summons while open (right-clicking another bin only
    // swaps inputs), so an empty→populated transition creates a brand-new
    // viewport. Tracking the view query re-runs this the moment that instance
    // appears; a one-shot ``ngAfterViewInit`` wiring would strand it (mirrors
    // media-list's viewport re-wire pattern).
    effect(() => {
      this.viewport();
      untracked(() => this.ensureScrollSubscription());
    });
  }

  ngOnDestroy(): void {
    this.scrollSub?.unsubscribe();
  }

  private ensureScrollSubscription(): void {
    const vp = this.viewport() ?? null;
    if (!vp || this.scrollSubscribedViewport === vp) return;
    this.scrollSubscribedViewport = vp;
    this.scrollSub?.unsubscribe();
    this.scrollSub = vp.scrolledIndexChange.subscribe(() => this.prefetchVisible());
    this.prefetchVisible();
  }

  // --- Selection -----------------------------------------------------------

  /** Whether ``id`` is selected. Reads the selection version signal so this
   *  view repaints when the selection changes anywhere (here, the canvas, the
   *  selection panel) — see the reactivity note on the class. */
  isSelected(id: number): boolean {
    this.selection.version();
    return this.selection.has(id);
  }

  /** True for the viewed item — the one in the shell's detail pane, ringed here
   *  as the arrow keys walk through the grid. */
  isFocused(id: number): boolean {
    const focused = this.focusedId();
    return focused != null && id === focused;
  }

  /** Tri-state of the select-all control: how many members are selected, as
   *  none / some / all — mirroring the dashboard's master-checkbox states. */
  get selectionState(): 'none' | 'some' | 'all' {
    this.selection.version();
    const total = this.ids().length;
    if (total === 0) return 'none';
    const sel = this.selection.selectedCountIn(this.ids());
    if (sel === 0) return 'none';
    if (sel >= total) return 'all';
    return 'some';
  }

  /** Select every member, or — when all are already selected — clear them.
   *  Matches the dashboard's toggle-all semantics. */
  toggleAll(): void {
    if (this.selectionState === 'all') {
      this.selection.removeAll(this.ids());
    } else {
      this.selection.addAll(this.ids());
    }
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      // Stop the bubble so the shell's document-level Space fallback (which acts
      // on the viewed item) doesn't also fire and double-toggle. The focused
      // entry owns its own activation; the fallback is only for when nothing in
      // the grid holds focus.
      event.stopPropagation();
      this.entryClick.emit(id);
    }
  }

  // --- Names + thumbnails --------------------------------------------------
  // Each of these reads the metadata cache's version signal before consulting
  // the cache, so a batch arrival repaints the visible rows on its own.

  name(id: number): string {
    this.metadataCache.version();
    return this.metadataCache.get(id)?.filename || `Clip #${id}`;
  }

  hasThumbnailUrl(id: number): boolean {
    this.metadataCache.version();
    const url = this.thumbnailUrl(id);
    if (this.failedThumbs.has(url)) return false;
    const media = this.metadataCache.get(id);
    return !!media && this.mediaTypeCaps.usesThumbnails(media.media_type);
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

  // --- Imperative surface used by the shell ---------------------------------

  /** Re-read the viewport's own size after the panel or the row stride changed,
   *  so its virtual window is computed against the new geometry. */
  remeasure(): void {
    this.viewport()?.checkViewportSize();
  }

  /** Prefetch metadata for the items around the visible window. */
  prefetchVisible(): void {
    const ids = this.ids();
    if (ids.length === 0) return;
    const cols = Math.max(1, this.columns());
    const rowSize = this.rowSize();
    const vp = this.viewport();
    if (!vp) {
      const window = Math.ceil(FALLBACK_VIEWPORT_PX / rowSize) * cols + PREFETCH_BUFFER;
      this.metadataCache.ensureLoaded(ids.slice(0, window));
      return;
    }
    const startRow = Math.floor(vp.measureScrollOffset('top') / rowSize);
    const visibleRows = Math.ceil(this.viewportHeight(vp) / rowSize);
    const from = Math.max(0, (startRow - Math.ceil(PREFETCH_BUFFER / cols)) * cols);
    const to = Math.min(ids.length, (startRow + visibleRows) * cols + PREFETCH_BUFFER);
    this.metadataCache.ensureLoaded(ids.slice(from, to));
  }

  /**
   * Remeasure the viewport (the row stride or the panel width changed) and
   * scroll so the row holding ``index`` sits roughly centred, so a resize keeps
   * looking at the same item instead of jumping to the top.
   *
   * The virtual viewport may not have applied its scrollable content size yet
   * (on open, or right after a thumbnail-size change re-chunks the rows), so the
   * browser clamps this scroll back to 0 and a large bin is left sitting at the
   * top with the target off-screen. When we meant to scroll down but the offset
   * didn't take, retry on the next frame (bounded) until the viewport is
   * scrollable and the target sticks.
   */
  centreOn(index: number, attempt = 0): void {
    const vp = this.viewport();
    if (!vp) return;
    if (attempt === 0) this.remeasure();
    const rowSize = this.rowSize();
    const row = Math.floor(index / Math.max(1, this.columns()));
    const viewportH = this.viewportHeight(vp);
    // The most we can scroll: content height (all rows) minus the visible window.
    const maxOffset = Math.max(0, this.rows().length * rowSize - viewportH);
    // Centre the row in the visible window, clamped to [0, maxOffset] so we never
    // scroll past either end.
    const target = Math.min(
      maxOffset,
      Math.max(0, row * rowSize - Math.max(0, viewportH - rowSize) / 2),
    );
    vp.scrollToOffset(target);
    if (target > 0 && vp.measureScrollOffset('top') < target - 1 && attempt < 5) {
      requestAnimationFrame(() => this.centreOn(index, attempt + 1));
    }
  }

  /**
   * Scroll the entry at ``index`` into view (the minimum amount; a no-op when it
   * already is) and move DOM focus onto it, so keyboard activation (Enter /
   * Space) targets the arrow-walked item rather than whatever entry last held
   * focus.
   *
   * The entry may not be in the DOM yet — the row was just scrolled into view
   * and the virtual viewport renders it on a subsequent frame — so query for it
   * after a frame and retry (bounded) until it exists. ``preventScroll`` keeps
   * the native focus scroll from fighting the scroll we just performed.
   */
  revealAndFocus(index: number): void {
    this.scrollRowIntoView(index);
    this.focusEntry(index);
  }

  private scrollRowIntoView(index: number): void {
    const vp = this.viewport();
    if (!vp) return;
    const rowSize = this.rowSize();
    const row = Math.floor(index / Math.max(1, this.columns()));
    const rowTop = row * rowSize;
    const rowBottom = rowTop + rowSize;
    const top = vp.measureScrollOffset('top');
    const viewportH = this.viewportHeight(vp);
    if (rowTop < top) {
      vp.scrollToOffset(rowTop);
    } else if (rowBottom > top + viewportH) {
      vp.scrollToOffset(rowBottom - viewportH);
    }
  }

  private focusEntry(index: number, attempt = 0): void {
    const root = this.host.nativeElement;
    const id = this.ids()[index];
    if (!root || id == null) return;
    requestAnimationFrame(() => {
      const el = root.querySelector<HTMLElement>(`.bin-popup-entry[data-entry-id="${id}"]`);
      if (el) {
        el.focus({ preventScroll: true });
      } else if (attempt < 5) {
        this.focusEntry(index, attempt + 1);
      }
    });
  }

  /** Rendered height (px) of the virtual viewport, falling back to the content
   *  height (capped) when it is unmeasurable — before layout, or under jsdom. */
  private viewportHeight(vp: CdkVirtualScrollViewport): number {
    const measured = vp.elementRef.nativeElement.clientHeight;
    if (measured) return measured;
    return Math.min(this.rows().length * this.rowSize(), FALLBACK_VIEWPORT_PX);
  }
}
