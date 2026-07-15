import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  inject,
  input,
  NgZone,
  OnChanges,
  OnDestroy,
  output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { ScrollingModule, CdkVirtualScrollViewport } from '@angular/cdk/scrolling';

/**
 * One display-ready cell in a Good/Bad vote pile. Every field is precomputed
 * by the parent when its inputs change (inside the existing sort pass), so
 * change detection over a large pile binds stored values instead of
 * re-deriving thumbnail URLs and icons per row per cycle.
 */
export interface VoteGridEntry {
  /** Stable identity for `track` (stringified media id / labelset element id). */
  key: string;
  name: string;
  /** Downscaled tile URL, or '' when the media type has no thumbnail. */
  thumbnailUrl: string;
  /** Icon shown when there is no thumbnail (or it failed to load); null renders the name only. */
  fallbackIcon: string | null;
  /** True when the item isn't present in the current dataset. */
  missing: boolean;
  /** True when {@link thumbnailUrl} is an audio waveform: a theme-agnostic
   *  alpha-mask PNG (issue #2369) that must be tinted via a CSS mask rather
   *  than shown as a plain <img>, so it recolours with the live theme. */
  isAudio?: boolean;
}

/**
 * Entry count above which the pile switches to CDK virtual scrolling.
 * Mirrors the left grid's ``GRID_VIRTUAL_THRESHOLD`` and its rationale: a few
 * hundred thumbnail rows re-rendered on every vote (and every labelset poll)
 * froze the panel, so small piles stay plain DOM (natural height, exact
 * layout) and larger ones mount only the visible rows.
 */
const PILE_VIRTUAL_THRESHOLD = 80;
/** Horizontal/vertical gap between pile cells (px); mirrors ``--space-xs``. */
const GRID_GAP_PX = 4;
/** Fallback row stride before the first real cell is measured (px). */
const ROW_HEIGHT_FALLBACK = 100;
/**
 * Smallest plausible measured row stride (px). A cell mid-relayout (its
 * ``<img>`` still loading) can momentarily report a near-zero height;
 * accepting that as the CDK ``itemSize`` would mount nearly every row at
 * once and defeat virtualization. Heights below this floor are treated as
 * "not yet laid out" and ignored until a real cell measures.
 */
const MIN_ROW_HEIGHT = 24;

/**
 * The grid of vote cells shared by the Good/Bad piles (`vt-label-list`) and
 * the saved-labelset piles (`vt-labelset-list`). Renders precomputed
 * ``VoteGridEntry`` rows and owns the plain-vs-virtualized switch, so both
 * parents stay a thin "sort + enrich" layer.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-vote-grid',
  standalone: true,
  imports: [NgTemplateOutlet, ScrollingModule],
  templateUrl: './vote-grid.component.html',
  styleUrl: './vote-grid.component.scss',
})
export class VoteGridComponent implements OnChanges, AfterViewChecked, OnDestroy {
  private cdr = inject(ChangeDetectorRef);
  private zone = inject(NgZone);

  readonly entries = input<VoteGridEntry[]>([]);
  readonly label = input<'good' | 'bad'>('good');
  readonly gridGoalWidth = input<number>(80);
  readonly focusMode = input<'click' | 'hover'>('click');

  readonly entrySelected = output<VoteGridEntry>();
  readonly entryVote = output<{
    entry: VoteGridEntry;
    vote: 'good' | 'bad';
}>();

  @ViewChild(CdkVirtualScrollViewport) virtualViewport?: CdkVirtualScrollViewport;

  /** ``entries`` chunked into rows of ``columns`` cells for virtual scrolling. */
  rows: VoteGridEntry[][] = [];
  /** Number of cells per virtualized row, derived from the viewport width. */
  columns = 1;
  /** Measured stride of one virtualized row (px); fed to CDK as ``itemSize``. */
  rowHeight = ROW_HEIGHT_FALLBACK;

  /** Thumbnail URLs that 404'd; their cells fall back to the type icon. */
  readonly thumbnailFailedUrls = new Set<string>();

  private resizeObserver?: ResizeObserver;
  private observedViewportEl?: HTMLElement;
  /** True once ``rowHeight`` has been measured from a real rendered cell. */
  private rowHeightMeasured = false;

  /** Whether the pile is large enough to render through the CDK viewport. */
  get useVirtual(): boolean {
    return this.entries().length > PILE_VIRTUAL_THRESHOLD;
  }

  ngOnChanges(changes: SimpleChanges): void {
    // A wider/narrower goal width changes how many cells fit per row and the
    // height of each cell, so recompute columns and re-measure the stride.
    if (changes['gridGoalWidth'] && !changes['gridGoalWidth'].firstChange) {
      this.rowHeightMeasured = false;
      this.recomputeColumns();
    }
    if (changes['entries'] || changes['gridGoalWidth']) {
      this.rebuildRows();
    }
  }

  ngAfterViewChecked(): void {
    // Keep the viewport measured while virtualizing: a ResizeObserver tracks
    // width (→ column count) and the first rendered cell's height (→ CDK row
    // stride). Tear it down when the pile shrinks back to plain DOM.
    if (this.useVirtual && this.virtualViewport) {
      this.setupViewport();
    } else if (this.observedViewportEl) {
      this.resizeObserver?.disconnect();
      this.resizeObserver = undefined;
      this.observedViewportEl = undefined;
    }
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
  }

  private rebuildRows(): void {
    const cols = Math.max(1, this.columns);
    const entries = this.entries();
    const rows: VoteGridEntry[][] = [];
    for (let i = 0; i < entries.length; i += cols) {
      rows.push(entries.slice(i, i + cols));
    }
    this.rows = rows;
  }

  /**
   * Recompute how many cells fit across the viewport. Returns ``true`` when
   * the column count changed (so callers know to rebuild the rows).
   */
  private recomputeColumns(): boolean {
    const el = this.virtualViewport?.elementRef.nativeElement;
    if (!el) return false;
    const inner = el.clientWidth;
    if (inner <= 0) return false;
    const cols = Math.max(1, Math.floor((inner + GRID_GAP_PX) / (this.gridGoalWidth() + GRID_GAP_PX)));
    if (cols === this.columns) return false;
    this.columns = cols;
    return true;
  }

  /** Observe the viewport for width/height changes (runs outside Angular). */
  private setupViewport(): void {
    const vp = this.virtualViewport?.elementRef.nativeElement;
    if (!vp || this.observedViewportEl === vp) {
      // Already observing the right element. Re-measure only until the row
      // height has been captured from a real cell, so the steady state doesn't
      // force a reflow on every change-detection cycle.
      if (!this.rowHeightMeasured) this.measureLayout();
      return;
    }
    this.resizeObserver?.disconnect();
    this.observedViewportEl = vp;
    this.rowHeightMeasured = false;
    this.zone.runOutsideAngular(() => {
      this.resizeObserver = new ResizeObserver(() => this.measureLayout());
      this.resizeObserver.observe(vp);
    });
    this.measureLayout();
  }

  /**
   * Recompute the column count (from viewport width) and row stride (from a
   * rendered cell), applying changes in a fresh CD tick. Guarded so it
   * converges instead of looping: it only re-renders when something moved.
   */
  private measureLayout(): void {
    let changed = this.recomputeColumns();
    if (changed) this.rebuildRows();

    const vp = this.virtualViewport?.elementRef.nativeElement;
    const cell = vp?.querySelector('.vote-entry') as HTMLElement | null;
    if (cell) {
      const measured = Math.round(cell.getBoundingClientRect().height + GRID_GAP_PX);
      // Only trust a measurement once the cell has actually laid out (see
      // MIN_ROW_HEIGHT); leave ``rowHeightMeasured`` false so the next pass
      // re-measures against a real cell.
      if (measured >= MIN_ROW_HEIGHT) {
        this.rowHeightMeasured = true;
        if (Math.abs(measured - this.rowHeight) > 1) {
          this.rowHeight = measured;
          changed = true;
        }
      }
    }

    if (changed) {
      // detectChanges() runs CD on this view synchronously and is
      // zone-independent (this fires from the ResizeObserver callback and
      // from ngAfterViewChecked).
      this.cdr.detectChanges();
    }
  }

  onThumbnailError(url: string): void {
    if (url) {
      this.thumbnailFailedUrls.add(url);
      this.cdr.markForCheck();
    }
  }

  onEntryClick(entry: VoteGridEntry): void {
    if (this.focusMode() === 'hover') {
      this.entryVote.emit({ entry, vote: 'bad' });
    } else {
      this.entrySelected.emit(entry);
    }
  }

  onEntryContextMenu(event: MouseEvent, entry: VoteGridEntry): void {
    if (this.focusMode() === 'hover') {
      event.preventDefault();
      this.entryVote.emit({ entry, vote: 'good' });
    }
  }

  onEntryMouseEnter(entry: VoteGridEntry): void {
    if (this.focusMode() === 'hover') {
      this.entrySelected.emit(entry);
    }
  }

  onEntryKeydown(event: KeyboardEvent, entry: VoteGridEntry): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.entrySelected.emit(entry);
    }
  }

  trackByRow(index: number, _row: VoteGridEntry[]): number {
    return index;
  }
}
