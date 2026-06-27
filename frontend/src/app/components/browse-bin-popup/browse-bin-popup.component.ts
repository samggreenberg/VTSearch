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
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { usesThumbnails } from '../browse-canvas/hex-render.util';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';

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
  imports: [CommonModule, ScrollingModule, ViewControlsComponent, IconComponent],
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

  /** Emitted when the popup should close (outside click, Escape, or the X). */
  readonly dismissed = output<void>();

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
    // Re-read the popup's thumbnail size whenever settings change (this is how
    // the in-header size buttons take effect, and how a change on one popup
    // becomes the default for every future popup of this media type).
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      this.gridSizeDict = (settings.grid_icon_size_popup as Record<string, string>) ?? {};
      this.previewSizeDict = (settings.popup_preview_size as Record<string, number>) ?? {};
      this.applyViewPrefs();
      // A detail-canvas size change (the top-left buttons) resizes the preview
      // pane, hence the whole popup, so re-clamp it back fully on-screen.
      const override = this.previewOverride;
      if (override !== this.lastPreviewOverride) {
        this.lastPreviewOverride = override;
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

  /** True for media types that carry real visual thumbnails (image / video):
   *  the ones that magnify on the main canvas and are worth a large preview. */
  get showPreview(): boolean {
    return usesThumbnails(this.mediaType());
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
      // The row stride changed; let the virtual viewport remeasure, then clamp.
      setTimeout(() => {
        this.viewport?.checkViewportSize();
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

  /** True for the representative entry, so the grid can ring the item whose
   *  thumbnail the user right-clicked (the one shown large in the preview). */
  isRepresentative(id: number): boolean {
    return id === this.representativeId();
  }

  /** Scroll the member grid so the representative's row sits roughly centred, so
   *  the popup opens looking at the same item whose pile thumbnail was clicked
   *  rather than the 1-D list's first item. No-op for a singleton bin (no grid)
   *  or before the viewport exists. */
  private scrollToRep(): void {
    const vp = this.viewport;
    if (!vp) return;
    const row = Math.floor(this.repIndex() / Math.max(1, this.columns));
    const viewportH = vp.elementRef.nativeElement.clientHeight || this.gridHeight;
    // Centre the row in the visible window, clamped so we never scroll past 0.
    const offset = row * this.rowSize - Math.max(0, viewportH - this.rowSize) / 2;
    vp.scrollToOffset(Math.max(0, offset));
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

  // --- Hover: preview the full-res original (image/video) + hear (audio) ----

  onEntryEnter(id: number): void {
    // Thumbnail media: paint the hovered item's full-res original into the pane.
    if (this.showPreview) this.previewId = id;
    if (this.mediaType() !== 'audio') return;
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
    if (this.showPreview) this.previewId = this.representativeId();
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
    this.clampInto(this.left, this.top);
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
    let l = this.left;
    let t = this.top;
    // Pull in from the far edges first, then guarantee the near edges, so a popup
    // larger than the region pins to top-left (losing the far edge, not the near).
    if (rect.right > regionRight - EDGE_MARGIN) l -= rect.right - (regionRight - EDGE_MARGIN);
    if (rect.bottom > regionBottom - EDGE_MARGIN) t -= rect.bottom - (regionBottom - EDGE_MARGIN);
    l = Math.max(regionLeft + EDGE_MARGIN, l);
    t = Math.max(regionTop + EDGE_MARGIN, t);
    if (l !== this.left || t !== this.top) {
      this.left = l;
      this.top = t;
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
    // a short canvas can't make the popup taller than what's visible.
    const regionRoom = regionBottom - regionTop - 2 * EDGE_MARGIN - headerH;
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
    const previewW = paneW ? paneW + (gridW ? PREVIEW_GAP : 0) : 0;
    const width = Math.min(previewW + gridW, this.maxWidthPx);
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
