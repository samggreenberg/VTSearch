import { HttpContext } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  effect,
  inject,
  input,
  OnInit,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { CdkVirtualScrollViewport, ScrollingModule } from '@angular/cdk/scrolling';
import { Observable, of, Subscription } from 'rxjs';
import { catchError, finalize, take, takeWhile } from 'rxjs/operators';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { ToastyHintComponent } from '../../toasty-hint/toasty-hint.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { ColumnChoice, ColumnChooserComponent } from '../../column-chooser/column-chooser.component';
import { ViewControlsComponent } from '../../view-controls/view-controls.component';
import { SKIP_ERROR_TOAST } from '../../../interceptors/error.interceptor';
import { ActiveContextService } from '../../../services/active-context.service';
import { adaptivePoll } from '../../../services/adaptive-poll';
import { BrowseSubsetPrepService } from '../../../services/browse-subset-prep.service';
import { runBrowseIds } from '../../../services/browse-subset.service';
import { ContextSwitchService } from '../../../services/context-switch.service';
import { DetectorsFindApiService } from '../../../services/detectors-find-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { ToastService } from '../../../services/toast.service';
import { PluginTemplateVarsService } from '../../../services/plugin-template-vars.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import {
  AutoDetectDetectorResult,
  AutoDetectHit,
  AutoDetectResultsData,
  ImporterField,
} from '../../../models/api.models';
import type { AutoFindBrowsePrepResponse } from '../../../generated/api-client/models/auto-find-browse-prep-response';
import type { AutoFindExportStatus } from '../../../generated/api-client/models/auto-find-export-status';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import type { MediaSimilarityOrder } from '../../../generated/api-client/models/media-similarity-order';
import { IconComponent } from '../../icon/icon.component';
import { apiErrorMessage } from '../../../utils/api-error';
import { autoExportFailure } from '../../../utils/auto-export-status';
import { exporterIconKind } from '../../../utils/exporter-icon';
import { openBlankTab, openExternalUrl, safeExternalUrl } from '../../../utils/external-url';
import { iconSizeToGoalWidth } from '../../../utils/grid-icon-size';
import { gridDom, ManagedColumns } from '../../../utils/managed-columns';
import { visibleFields } from '../../../utils/plugin-fields';
import { notchMoves, runNotch, splitAtNotch } from '../../../utils/results-notch';
import { labelsOwed } from '../../../utils/label-quota';
import { coerceNonEmptyString } from '../../../utils/settings-coerce';
import { isSoloMediaType } from '../../../utils/solo-media-type';
import { DomainShiftWarning, domainShiftWarning } from '../../../utils/training-domain';
import { BalanceComponent } from '../../left-panel/balance/balance.component';
import { PluginCheckboxComponent } from '../../plugin-checkbox/plugin-checkbox.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

/** Which side of each detector's threshold the list shows. */
export type ResultsSides = 'good' | 'bad' | 'both';

/** One listed row: a hit, the side and detector it came from. */
export interface ResultRow {
  hit: AutoDetectHit;
  /** Unique in the list: one media can be a row under each detector. */
  key: string;
  /** The detector's key in the run's ``results``. */
  resultKey: string;
  /** The detector's name, for the Detector column. */
  detector: string;
  label: 'good' | 'bad';
}

/** A column the list can show; the thumbnail is shown but not copied or exported. */
export interface ResultsColumn extends ColumnChoice {
  /** The column's track in the list's CSS grid, until the user resizes one. */
  track: string;
  /** Whether a click on its heading orders the list by it. */
  sortable: boolean;
  displayOnly?: boolean;
}

/** The Similarity column's state: the run's ids ranked once the order lands. */
type SimilarityState =
  | { status: 'idle' }
  | { status: 'building'; progress: MediaSimilarityOrder | null }
  | { status: 'ready'; rank: ReadonlyMap<number, number> }
  | { status: 'error'; message: string };

/** How long the scroll must rest before the thumbnails on screen load (ms). */
export const THUMB_SETTLE_MS = 120;
/** How many detectors the summary names before giving their count instead. */
export const MAX_NAMED_DETECTORS = 3;
/** The destination tab that sends the run as AutoFind does (#5584). */
export const AUTOFIND_TAB = ':autofind';
/** A row with no thumbnail column (px): one line of text. */
const TEXT_ROW_PX = 30;
/** Space above and below a thumbnail in its row (px). */
const THUMB_PAD_PX = 2;
/** The thumbnail column's track: its tile plus the cell's side padding (px). */
const THUMB_CELL_PAD_PX = 12;
/** The share of the list a column ticked on after a resize starts with (%). */
const LATE_COL_PCT = 10;
/** The columns in the order they are first offered; a run's metadata keys join after. */
const COLUMN_KEYS = ['thumbnail', 'origin_name', 'score', 'similarity', 'label', 'detector', 'filename', 'origin', 'md5'];
const COLUMN_ORDER_KEY = 'vtsearch.findResults.columnOrder';

/** The Find Results dialog: one Find or AutoFind run's hits on one dataset.
 *
 * Laid out by what each control acts on (#5464): above the list, what the list
 * shows, in the order each acts on it (#5540): the Threshold splits the run
 * into Good and Bad, Good / Bad / Both picks the side to list; then the
 * columns, with the thumbnail size at the end of their row; the list, whose
 * headings sort it, reorder it by drag and resize it, as the Dashboard's grids
 * do (#5584); below it, what acts on the listed rows (Browse, which opens them
 * as a map of their own, #4615; and where they go, a tab per destination with
 * Clipboard first and, last, AutoFind, which sends the run the way a finished
 * run does); and the footer's Close / Export / Export and Close. While the
 * dialog is open, the server lays out the Good results for Browse whenever it
 * has nothing else to do (#4683). Mounted once in `AppComponent`, fed by
 * `AutoFindService`.
 *
 * The list is virtual (a run's Both can be the whole dataset), its rows a CSS
 * grid of the chosen columns; thumbnails load once the scroll rests and stay
 * loaded, as New Detector's picker grid does. The Similarity column is each
 * item's place in the dataset's 1-D UMAP, fit with its Browse layout: one the
 * dataset has answers at once, one it lacks is fit while the dialog waits on
 * it; sorting by the column puts look-alikes side by side. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autodetect-results-modal',
  standalone: true,
  imports: [
    FormsModule,
    ScrollingModule,
    ModalComponent,
    ToastyHintComponent,
    ClipboardCopyComponent,
    ColumnChooserComponent,
    ViewControlsComponent,
    BalanceComponent,
    IconComponent,
    PluginCheckboxComponent,
    ProgressBarComponent,
  ],
  templateUrl: './autodetect-results-modal.component.html',
  styleUrl: './autodetect-results-modal.component.scss',
})
export class AutoDetectResultsModalComponent implements OnInit {
  private exportersApi = inject(ExportersApiService);
  private templateVars = inject(PluginTemplateVarsService);
  private toast = inject(ToastService);
  private activeContext = inject(ActiveContextService);
  private contextSwitch = inject(ContextSwitchService);
  private findApi = inject(DetectorsFindApiService);
  private mediasApi = inject(MediasApiService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);
  private router = inject(Router);
  private settingsState = inject(SettingsStateService);
  /** Builds the Browse map while the dialog shows its progress. */
  readonly browsePrep = inject(BrowseSubsetPrepService);
  /** Browse was clicked and the run's dataset is being made active (and
   *  loaded, if it was unloaded) before the map build starts. */
  readonly browseStarting = signal(false);

  readonly data = input<AutoDetectResultsData>({ results: {} });
  readonly closed = output<void>();

  readonly AUTOFIND_TAB = AUTOFIND_TAB;

  // --- The summary -----------------------------------------------------------

  /** The detectors that ran, by name, in the run's order. */
  readonly detectorNames = computed(() =>
    Object.entries(this.data().results || {}).map(([key, result]) => result.detector_name || key),
  );

  /** The summary names the detectors that ran; past {@link MAX_NAMED_DETECTORS}
   *  it gives their count, with the names on hover ({@link detectorsTitle}). */
  get detectorsSummary(): string {
    const names = this.detectorNames();
    if (names.length === 0) return String(this.data().detectors_run ?? 0);
    return names.length <= MAX_NAMED_DETECTORS ? names.join(', ') : String(names.length);
  }

  get detectorsLabel(): string {
    return this.detectorNames().length === 1 ? 'Detector:' : 'Detectors:';
  }

  /** Every detector that ran, one per line, for the summary's tooltip. */
  get detectorsTitle(): string {
    return this.detectorNames().join('\n');
  }

  // --- What the list shows --------------------------------------------------

  /**
   * The notch of the Threshold the run is split at (#5540): a preset's beta,
   * or `null` for each detector's own, the notch the run was cut at. Another
   * notch lists what it would have returned: no retrain, the run's scores
   * split at that balance's cut (`splitAtNotch`). No detector's own balance
   * moves.
   */
  readonly notch = signal<number | null>(null);
  /** The notch the Threshold shows: the one picked, else the run's own (none when its detectors differ). */
  readonly shownNotch = computed(() => this.notch() ?? runNotch(this.data()));
  /** Whether another notch would list anything else; when not, the Threshold is frozen. */
  readonly notchMoves = computed(() => notchMoves(this.data()));
  /** Why the Threshold is frozen, on each of its radios. */
  readonly notchFrozenHint =
    "Every notch lists the same items here: these results' line doesn't move with the Threshold.";

  /** Each detector's Good and Bad at the {@link notch}, by its key in the run's ``results``. */
  readonly splits = computed(() => {
    const notch = this.notch();
    return Object.entries(this.data().results || {}).map(([resultKey, result]) => ({
      resultKey,
      result,
      split: splitAtNotch(result, notch),
    }));
  });
  /** Whether the list is split at another notch than the run's own, for the note under the row. */
  readonly notchMoved = computed(() => this.splits().some(({ split }) => split.moved));

  /** Split the run at *beta*'s notch; the run's own notch goes back to the run as it came. */
  setNotch(beta: number): void {
    const moves = Object.values(this.data().results || {}).some((result) => splitAtNotch(result, beta).moved);
    this.notch.set(moves ? beta : null);
  }

  /** Which side of the threshold is listed (and exported, and browsed). */
  readonly sides = signal<ResultsSides>('good');

  readonly mediaType = computed(() => this.data().media_type ?? '');

  /** The media type's icon and name, as the Dashboard's Type column shows them. */
  readonly mediaTypeIcon = computed(() => this.mediaTypeCaps.iconFor(this.mediaType()));
  readonly mediaTypeName = computed(() => {
    const type = this.mediaType();
    return type ? type.charAt(0).toUpperCase() + type.slice(1) : '';
  });

  /** Whether the run's items have thumbnails to show: a type with them, in a
   *  dataset the run names (they are served out of it). */
  readonly hasThumbnails = computed(
    () => !!this.data().dataset_id && this.mediaTypeCaps.usesThumbnails(this.mediaType()),
  );
  /** Audio thumbnails are waveform alpha masks tinted to the accent (#2369). */
  readonly isAudioWaveform = computed(() => this.mediaType() === 'audio');

  /** Every listed row, side and detector stamped, in the run's own order. */
  readonly rows = computed<ResultRow[]>(() => {
    const sides = this.sides();
    const rows: ResultRow[] = [];
    for (const { resultKey, result, split } of this.splits()) {
      const detector = result.detector_name || resultKey;
      const add = (hit: AutoDetectHit, label: 'good' | 'bad') =>
        rows.push({ hit, label, resultKey, detector, key: `${resultKey}|${label}|${hit.id ?? hit.md5}` });
      if (sides !== 'bad') for (const hit of split.good) add(hit, 'good');
      if (sides !== 'good') for (const hit of split.bad) add(hit, 'bad');
    }
    return rows;
  });

  /** {@link rows} in the order the headings set ({@link sort}). A row with
   *  nothing in the sort column goes last whichever way the column runs;
   *  ties, by score, highest first. The Similarity order, until it has
   *  landed, is the score order. */
  readonly sortedRows = computed<ResultRow[]>(() => {
    const { column, asc } = this.sort();
    const similarity = this.similarity();
    const rank = similarity.status === 'ready' ? similarity.rank : null;
    const dir = asc ? 1 : -1;
    const byScore = (a: ResultRow, b: ResultRow) => (b.hit.score ?? -Infinity) - (a.hit.score ?? -Infinity);
    return [...this.rows()].sort((a, b) => {
      const va = this.sortValue(a, column, rank);
      const vb = this.sortValue(b, column, rank);
      if (va === null || vb === null) return (va === null ? 1 : 0) - (vb === null ? 1 : 0) || byScore(a, b);
      const cmp =
        typeof va === 'number' && typeof vb === 'number'
          ? va - vb
          : String(va).localeCompare(String(vb), undefined, { numeric: true, sensitivity: 'base' });
      return cmp * dir || byScore(a, b);
    });
  });

  /** What *row* sorts by in column *column*: a number where the column is one
   *  (Score, Similarity, Label as Good before Bad, a numeric metadata key),
   *  its text otherwise, and `null` for nothing to sort by. */
  private sortValue(row: ResultRow, column: string, rank: ReadonlyMap<number, number> | null): number | string | null {
    switch (column) {
      case 'score':
        return row.hit.score ?? null;
      case 'similarity':
        return (row.hit.id != null ? rank?.get(row.hit.id) : undefined) ?? null;
      case 'label':
        return row.label === 'good' ? 0 : 1;
      default: {
        const raw = row.hit.custom_metadata?.[column];
        if (typeof raw === 'number') return raw;
        const text = this.cellValue(row, column);
        return text === '' ? null : text;
      }
    }
  }

  /** The listed hits in order, each stamped with its side when both are listed. */
  readonly displayHits = computed<AutoDetectHit[]>(() =>
    this.sortedRows().map((row) => (this.sides() === 'both' ? { ...row.hit, label: row.label } : row.hit)),
  );

  // --- Columns ----------------------------------------------------------------

  /**
   * The list's headings, as the Dashboard's grids have them (#5584): a click
   * sorts by the column, a drag reorders it, the line between two resizes
   * them. The order is kept across sessions; the sort starts on Score,
   * highest first. The list is a CSS grid of divs under a virtual scroller,
   * so the controller reads its geometry through `gridDom`.
   */
  readonly cols = new ManagedColumns<string>(COLUMN_KEYS, {}, {
    initialSort: 'score',
    initialSortAsc: false,
    storageKey: COLUMN_ORDER_KEY,
    dom: gridDom('.results-table', '.results-head [data-col]', '.results-row'),
  });
  /** The sort, as a signal, so the order recomputes on a heading click. */
  readonly sort = toSignal(this.cols.sortState$, { requireSync: true });
  /** The columns' order, as a signal, so the list re-lays out on a drop. */
  private readonly order = signal<string[]>([...this.cols.columnOrder]);
  /** The columns' widths once one has been resized, as a signal, so the grid follows the drag. */
  private readonly layout = signal<{ fixed: boolean; widths: Readonly<Record<string, number>> }>({
    fixed: false,
    widths: {},
  });

  /** The user's ticks, by column key; a column not here is at its default. */
  private readonly columnTicks = signal<Readonly<Record<string, boolean>>>({});

  /** Every key the run's hits carry in ``custom_metadata``, first seen first. */
  private readonly metadataKeys = computed(() => {
    const keys = new Set<string>();
    for (const result of Object.values(this.data().results || {})) {
      for (const hit of [...(result.hits || []), ...(result.negative_hits || [])]) {
        for (const key of Object.keys(hit.custom_metadata || {})) keys.add(key);
      }
    }
    return [...keys];
  });

  /** The columns the list can show now, ticked or not, in the order they are
   *  first offered (the checkboxes' order). Label only lists both sides;
   *  Detector only a run of several; Thumbnail only a type with them;
   *  Similarity only results that name a dataset to order. The metadata
   *  columns and Similarity start unticked, the rest ticked. */
  readonly columns = computed<ResultsColumn[]>(() => {
    const ticks = this.columnTicks();
    const col = (key: string, label: string, track: string, extra: Partial<ResultsColumn> = {}, on = true) => ({
      key,
      label,
      track,
      sortable: true,
      ...extra,
      enabled: ticks[key] ?? on,
    });
    const cols: ResultsColumn[] = [];
    if (this.hasThumbnails()) {
      cols.push(
        col('thumbnail', 'Thumbnail', `${this.thumbPx() + THUMB_CELL_PAD_PX}px`, {
          displayOnly: true,
          sortable: false,
          title: 'Shown in the list; not copied or exported',
        }),
      );
    }
    cols.push(col('origin_name', 'Name', 'minmax(6rem, 2fr)'));
    cols.push(col('score', 'Score', '5rem'));
    if (this.data().dataset_id) {
      cols.push(
        col(
          'similarity',
          'Similarity',
          '6rem',
          { title: "Each item's place in the dataset's 1-D similarity order; sort by it for look-alikes side by side" },
          false,
        ),
      );
    }
    if (this.sides() === 'both') cols.push(col('label', 'Label', '4.5rem'));
    if (Object.keys(this.data().results || {}).length > 1) cols.push(col('detector', 'Detector', 'minmax(5rem, 1fr)'));
    cols.push(col('filename', 'Filename', 'minmax(6rem, 2fr)'));
    cols.push(col('origin', 'Origin', 'minmax(6rem, 2fr)'));
    cols.push(col('md5', 'MD5', 'minmax(6rem, 2fr)'));
    for (const key of this.metadataKeys()) {
      cols.push(col(key, humanizeKey(key), 'minmax(5rem, 1fr)', {}, false));
    }
    return cols;
  });

  /** The ticked columns, in the user's order: what the list shows. */
  readonly listedColumns = computed(() => {
    const offered = new Map(this.columns().map((c) => [c.key, c]));
    const order = this.order();
    const keys = [...order.filter((k) => offered.has(k)), ...[...offered.keys()].filter((k) => !order.includes(k))];
    return keys.map((k) => offered.get(k)!).filter((c) => c.enabled);
  });
  /** The ticked columns that are data: what the clipboard copies and an export sends. */
  private readonly dataColumns = computed(() => this.listedColumns().filter((c) => !c.displayOnly));
  /** The list's (and its header's) CSS grid columns: each column's track, or
   *  its share of the list once one has been resized. The thumbnail keeps its
   *  tile's width either way. */
  readonly gridTemplate = computed(() => {
    const { fixed, widths } = this.layout();
    return this.listedColumns()
      .map((c) => (c.key === 'thumbnail' || !fixed ? c.track : `minmax(0, ${widths[c.key] ?? LATE_COL_PCT}fr)`))
      .join(' ');
  });
  /** Whether the Similarity column is listed, which is what asks for the order. */
  readonly showsSimilarity = computed(() => this.listedColumns().some((c) => c.key === 'similarity'));

  setColumnEnabled(key: string, enabled: boolean): void {
    this.columnTicks.update((ticks) => ({ ...ticks, [key]: enabled }));
  }

  /** From [x] (all ticked), clear every column; from [ ]/[-], tick them all. */
  toggleAllColumns(): void {
    const columns = this.columns();
    const on = !columns.every((c) => c.enabled);
    this.columnTicks.update((ticks) => ({ ...ticks, ...Object.fromEntries(columns.map((c) => [c.key, on])) }));
  }

  /** The heading's tooltip: what the column is, and what a click and a drag do. */
  headerTitle(col: ResultsColumn): string {
    const what = col.title || col.label;
    return col.sortable ? `${what} (click to sort, drag to reorder)` : `${what} (drag to reorder)`;
  }

  onHeaderClick(col: ResultsColumn): void {
    if (col.sortable) this.cols.sortBy(col.key);
  }

  onHeaderKeydown(event: KeyboardEvent, col: ResultsColumn): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onHeaderClick(col);
    }
  }

  dropColumn(event: DragEvent, key: string): void {
    this.cols.onColDrop(event, key);
    this.order.set([...this.cols.columnOrder]);
  }

  /** Whether listed column *index* carries the handle that resizes it against
   *  the column before it: not the first, and not the thumbnail on either
   *  side, whose tile keeps its width. */
  canResize(index: number): boolean {
    const listed = this.listedColumns();
    return index > 0 && listed[index].key !== 'thumbnail' && listed[index - 1].key !== 'thumbnail';
  }

  private stopResize: (() => void) | null = null;

  /** Start dragging a column divider, following the mouse for as long as it
   *  is held wherever it goes. The controller keeps the widths; copying them
   *  into {@link layout} on each move is what repaints the grid. */
  startResize(event: MouseEvent): void {
    this.stopResize?.();
    this.cols.startResize(event);
    this.syncLayout();
    const move = (e: MouseEvent) => {
      this.cols.onResizeMove(e);
      this.syncLayout();
    };
    const stop = () => {
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', stop);
      this.stopResize = null;
      this.cols.onResizeEnd();
      this.syncLayout();
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', stop);
    this.stopResize = stop;
  }

  private syncLayout(): void {
    this.layout.set({ fixed: this.cols.tableFixed, widths: { ...this.cols.colWidths } });
  }

  /** What *row* shows in column *key*, as the list, the clipboard and a
   *  tooltip read it. */
  cellValue(row: ResultRow, key: string): string {
    const hit = row.hit;
    switch (key) {
      case 'origin_name':
        return hit.origin_name || hit.filename || '';
      case 'score':
        return hit.score != null ? String(hit.score) : '';
      case 'similarity': {
        const similarity = this.similarity();
        const rank = similarity.status === 'ready' && hit.id != null ? similarity.rank.get(hit.id) : undefined;
        return rank == null ? '' : String(rank + 1);
      }
      case 'label':
        return row.label === 'good' ? 'Good' : 'Bad';
      case 'detector':
        return row.detector;
      case 'origin':
        return this.formatOrigin(hit);
      case 'md5':
        return hit.md5 || '';
      case 'filename':
        return hit.filename || '';
      default: {
        const value = hit.custom_metadata?.[key];
        return value == null ? '' : String(value);
      }
    }
  }

  // --- Thumbnails and the virtual list ------------------------------------------

  /** The thumbnail size, from the list's own smaller/bigger buttons. */
  private readonly sizePref = this.settingsState.perMediaType<string>('grid_icon_size_results', this.mediaType, {
    fallback: 'M',
    coerce: coerceNonEmptyString,
  });
  /** Side of one thumbnail (px). */
  readonly thumbPx = computed(() => iconSizeToGoalWidth(this.sizePref.value()));
  /** Whether the list shows thumbnails: the column is offered and ticked. */
  readonly showThumbs = computed(() => this.listedColumns().some((c) => c.key === 'thumbnail'));
  /** Pixel height of one virtual row: a line of text, or a thumbnail's. */
  readonly rowHeight = computed(() =>
    this.showThumbs() ? Math.max(TEXT_ROW_PX, this.thumbPx() + 2 * THUMB_PAD_PX) : TEXT_ROW_PX,
  );

  private readonly viewport = viewChild(CdkVirtualScrollViewport);
  private rangeSub: Subscription | null = null;
  private rangeViewport: CdkVirtualScrollViewport | null = null;
  /** Media ids whose thumbnails may load: on screen when the scroll last
   *  rested. Never shrinks, so a thumbnail loaded once stays loaded. */
  private readonly settledIds = signal<ReadonlySet<number>>(new Set());
  private settleTimer: ReturnType<typeof setTimeout> | null = null;
  /** Thumbnails that failed to load; their cells stay empty. */
  private readonly failedThumbs = signal<ReadonlySet<number>>(new Set());

  /** Whether *id*'s thumbnail may load: it rested on screen, and has not failed. */
  thumbReady(id: number | undefined): boolean {
    return id != null && this.settledIds().has(id) && !this.failedThumbs().has(id);
  }

  thumbnailUrl(id: number): string {
    return this.activeContext.mediaUrlIn(this.data().dataset_id ?? '', `/api/medias/${id}/thumbnail`);
  }

  onThumbnailError(id: number): void {
    this.failedThumbs.update((failed) => new Set(failed).add(id));
  }

  trackRow(_index: number, row: ResultRow): string {
    return row.key;
  }

  /** The viewport lives behind the template's `@if`s, so re-wire on each new one. */
  private wireViewport(): void {
    const vp = this.viewport() ?? null;
    if (vp === this.rangeViewport) return;
    this.rangeViewport = vp;
    this.rangeSub?.unsubscribe();
    this.rangeSub = vp ? vp.renderedRangeStream.subscribe(() => this.scheduleSettle()) : null;
  }

  /** Load the thumbnails on screen once the scroll has rested {@link THUMB_SETTLE_MS}. */
  private scheduleSettle(): void {
    if (!this.showThumbs()) return;
    if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      this.settle();
    }, THUMB_SETTLE_MS);
  }

  private settle(): void {
    const vp = this.viewport();
    if (!vp) return;
    const { start, end } = vp.getRenderedRange();
    const settled = this.settledIds();
    const fresh = this.sortedRows()
      .slice(start, end)
      .map((row) => row.hit.id)
      .filter((id): id is number => id != null && !settled.has(id));
    if (fresh.length === 0) return;
    this.settledIds.update((ids) => new Set([...ids, ...fresh]));
  }

  // --- The Similarity column -----------------------------------------------------

  readonly similarity = signal<SimilarityState>({ status: 'idle' });
  /** How often to ask again while the dataset's order is being fit (ms). */
  static readonly SIMILARITY_POLL_MS = 1500;
  private similaritySub: Subscription | null = null;
  private similarityTimer: ReturnType<typeof setTimeout> | null = null;

  /** Untick the Similarity column and stop waiting on the fit (which carries
   *  on on the server, so ticking it again later finds it further along). */
  cancelSimilarity(): void {
    this.stopSimilarity();
    this.similarity.set({ status: 'idle' });
    this.setColumnEnabled('similarity', false);
  }

  /** Ask for the dataset's order (the Similarity column was ticked, a new run
   *  came in, or Try again was pressed after a failure). */
  requestSimilarity(): void {
    const datasetId = this.data().dataset_id;
    if (!datasetId) {
      this.similarity.set({ status: 'error', message: 'These results name no dataset to order.' });
      return;
    }
    const ids = new Set<number>();
    for (const result of Object.values(this.data().results || {})) {
      for (const hit of [...(result.hits || []), ...(result.negative_hits || [])]) {
        if (hit.id != null) ids.add(hit.id);
      }
    }
    this.stopSimilarity();
    this.similarity.set({ status: 'building', progress: null });
    this.askSimilarity(datasetId, [...ids]);
  }

  /** One ask; a `building` answer schedules the next. Its own failures are
   *  shown in the dialog, so the error toast is skipped. */
  private askSimilarity(datasetId: string, ids: number[]): void {
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    this.similaritySub = this.mediasApi
      .getSimilarityOrderIn(datasetId, ids, context)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (answer) => {
          if (answer.status === 'ready') {
            this.similarity.set({ status: 'ready', rank: new Map((answer.ids ?? []).map((id, i) => [id, i])) });
          } else if (answer.status === 'error') {
            this.similarity.set({ status: 'error', message: answer.error || 'Ordering by similarity failed.' });
          } else {
            this.similarity.set({ status: 'building', progress: answer });
            this.similarityTimer = setTimeout(() => {
              this.similarityTimer = null;
              this.askSimilarity(datasetId, ids);
            }, AutoDetectResultsModalComponent.SIMILARITY_POLL_MS);
          }
        },
        error: (err) =>
          this.similarity.set({
            status: 'error',
            message: apiErrorMessage(err, 'Could not order these results by similarity.'),
          }),
      });
  }

  private stopSimilarity(): void {
    this.similaritySub?.unsubscribe();
    this.similaritySub = null;
    if (this.similarityTimer !== null) clearTimeout(this.similarityTimer);
    this.similarityTimer = null;
  }

  /** Why the order could not be had, while the Similarity column is in error. */
  readonly similarityError = computed(() => {
    const state = this.similarity();
    return state.status === 'error' ? state.message : '';
  });

  /** The fit's progress for the bar: a fraction once it reports one. */
  similarityFraction(state: SimilarityState): number | null {
    return state.status === 'building' ? (state.progress?.overall ?? null) : null;
  }

  // --- Where the rows go ------------------------------------------------------

  // Signals: these are written from the getExporters() subscribe callback (not a
  // zoneless CD trigger) yet read in the template, so they must repaint on emit.
  /** The exporters offered a tab each: those that can read a scored run. */
  readonly exporters = signal<ExporterEntry[]>([]);
  /** Every exporter the server has, for naming the one AutoFind is set to. */
  private readonly allExporters = signal<ExporterEntry[]>([]);
  /** The open destination tab: `'clipboard'` (the default), an exporter's name, or {@link AUTOFIND_TAB}. */
  readonly activeTab = signal('clipboard');
  /** Every field the active exporter declares, ``hidden`` ones included. */
  private readonly declaredExporterFields = signal<ImporterField[]>([]);
  exportFieldValues: Record<string, string> = {};

  /** The active exporter's *renderable* fields. ``hidden`` fields are left
   *  out (their values are the plugin author's, fixed via ``default``) but
   *  are still seeded into ``exportFieldValues``; so is any field whose
   *  ``visible_when`` fails on the current ``exportFieldValues``. */
  exporterFields(): ImporterField[] {
    return visibleFields(this.declaredExporterFields(), this.exportFieldValues);
  }
  /** An Export click is in flight (disables the buttons). */
  readonly exporting = signal(false);

  /** The Clipboard tab's control, which the footer's Export runs. */
  private readonly clipboard = viewChild(ClipboardCopyComponent);

  /** The ticked data columns, for the Clipboard tab's control. */
  readonly clipboardColumns = computed<ClipboardColumn[]>(() =>
    this.dataColumns().map((c) => ({ key: c.key, label: c.label })),
  );

  /** The listed rows flattened to `{ columnKey: value }`, as the list shows them. */
  readonly clipboardRows = computed<Record<string, string>[]>(() => {
    const cols = this.dataColumns();
    return this.sortedRows().map((row) => Object.fromEntries(cols.map((c) => [c.key, this.cellValue(row, c.key)])));
  });

  /** The icon on an exporter's tab, as the Export dialog's tabs wear it. */
  exporterIcon(exp: ExporterEntry): string {
    const kind = exporterIconKind(exp);
    return kind === 'webhook' ? 'globe' : kind === 'external-link' ? 'pop-out' : kind;
  }

  // --- The AutoFind tab (#5584) ---------------------------------------------------

  /** The exporter set up on the Settings AutoFind tab, by name; `''` for none. */
  readonly autofindExporterName = computed(() => this.settingsState.settingsSignal()?.autofind_exporter || '');
  /** That exporter's entry, once the exporter list has landed. */
  readonly autofindExporter = computed(
    () => this.allExporters().find((e) => e.name === this.autofindExporterName()) ?? null,
  );
  /** What it is called to the user. */
  readonly autofindExporterLabel = computed(() => {
    const exp = this.autofindExporter();
    return exp?.display_name || exp?.name || this.autofindExporterName();
  });
  /** The field values saved for it on the Settings AutoFind tab. */
  private readonly autofindFieldValues = computed<Record<string, string>>(
    () => this.settingsState.settingsSignal()?.autofind_exporter_field_values?.[this.autofindExporterName()] ?? {},
  );
  /** Its fields the user can see there, with those values: what the export runs with. */
  readonly autofindFields = computed(() =>
    visibleFields((this.autofindExporter()?.fields ?? []) as ImporterField[], this.autofindFieldValues()),
  );
  /** Whether finished runs are held back from sending (#5170); a press here still sends. */
  readonly autofindPaused = computed(() => this.settingsState.settingsSignal()?.autofind_export_enabled === false);
  /** What a finished run sends: every detector's Good at the run's own cut. */
  readonly autofindHitCount = computed(() =>
    Object.values(this.data().results || {}).reduce((n, result) => n + (result.hits || []).length, 0),
  );

  /** *field*'s saved value as the read-only tab shows it: a password hidden,
   *  a checkbox as Yes / No, a blank as a dash. A blank saved value is the
   *  field's declared default, as the server reads it (`normalize_field_values`). */
  autofindFieldText(field: ImporterField): string {
    const saved = this.autofindFieldValues()[field.key];
    const value = saved == null || saved === '' ? (field.default ?? '') : saved;
    if (field.field_type === 'checkbox') return /^(true|1|yes|on)$/i.test(String(value)) ? 'Yes' : 'No';
    if (value === '') return '—';
    return field.field_type === 'password' ? '••••••••' : String(value);
  }

  private readonly destroyRef = inject(DestroyRef);

  /** How often to ask whether the server is free to lay out the Good results
   *  for Browse: quickly at first, easing off while the answer stays `busy`. */
  static readonly WARM_FAST_MS = 3000;
  static readonly WARM_SLOW_MS = 15000;
  /** The run whose Good results are being readied for Browse, and the loop
   *  asking the server to; see {@link warmBrowse}. */
  private warmRunId = '';
  private warmSub: Subscription | null = null;

  constructor() {
    this.mediaTypeCaps.ensureLoaded();
    // Per run, not once: a toast's View Results can swap another run into
    // the open dialog.
    effect(() => {
      const { run_id: runId, dataset_id: datasetId } = this.data();
      untracked(() => this.warmBrowse(runId ?? '', datasetId ?? ''));
    });
    // A new run's ids need their own order: ask again if the column is up. It
    // is listed at its own notch, which another run need not share.
    effect(() => {
      this.data();
      untracked(() => {
        this.notch.set(null);
        this.stopSimilarity();
        this.similarity.set({ status: 'idle' });
        if (this.showsSimilarity()) this.requestSimilarity();
      });
    });
    // The Similarity column asks for the order when it is ticked, and stops
    // waiting when it is unticked (a landed order is kept for next time). A
    // failure waits for Try again rather than asking again on its own.
    effect(() => {
      const shown = this.showsSimilarity();
      untracked(() => {
        const status = this.similarity().status;
        if (shown && status === 'idle') this.requestSimilarity();
        if (!shown && status === 'building') this.cancelSimilarity();
      });
    });
    // A run's metadata keys are columns the controller learns of here; and a
    // column that leaves the list (unticked, or Label with Both) cannot stay
    // the sort, which goes back to Score, highest first.
    effect(() => {
      const keys = this.columns().map((c) => c.key);
      const listed = this.listedColumns();
      untracked(() => {
        if (this.cols.addColumns(keys)) this.order.set([...this.cols.columnOrder]);
        if (listed.length > 0 && !listed.some((c) => c.key === this.sort().column)) this.cols.setSort('score', false);
      });
    });
    effect(() => {
      this.viewport();
      untracked(() => this.wireViewport());
    });
    // A new order, side or size can put unseen rows where the rendered range
    // already was, which the range stream does not report.
    effect(() => {
      this.sortedRows();
      this.rowHeight();
      untracked(() => this.scheduleSettle());
    });
    this.destroyRef.onDestroy(() => {
      this.stopWarm();
      this.stopSimilarity();
      this.stopResize?.();
      this.rangeSub?.unsubscribe();
      if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    });
  }

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (list) => {
        this.allExporters.set(list);
        // Drop exporters the plugin author flagged hidden_from_picker, and
        // those that can't read a scored run - this picker's destination is
        // always find results (matches the export modal, which filters on
        // `labelset` instead).
        this.exporters.set(
          list.filter((exp) => !exp.hidden_from_picker && (exp.supported_payloads ?? []).includes('find_results')),
        );
      },
    });
  }

  /**
   * The AutoFind auto-export's `open_url`, if it returned an openable one.
   *
   * An exporter can format the run's results into a third-party site's URL
   * instead of (or as well as) delivering them somewhere; the same key drives
   * the Export modal. Surfaced as an "Open" button rather than opened on
   * arrival, because these results land from an async response and a popup
   * blocker would swallow an unprompted `window.open()`.
   */
  autoExportUrl(): string | null {
    const status = this.data().auto_export;
    return status?.success ? safeExternalUrl(status.open_url) : null;
  }

  /** Open the auto-export's URL in a new tab (the click is the user gesture). */
  openExternal(url: string): void {
    openExternalUrl(url);
  }

  /** The failed auto-export's line, worded as the run's toast words it. */
  readonly autoExportFailure = autoExportFailure;

  /**
   * The detectors that ran as the Goods' centroid (#4643): too few labels for
   * a trained detector and no typed query, so their hits are the centroid's.
   * One line each, with the labels still owed.
   */
  get centroidDetectors(): string[] {
    return this.detectorsServedAs('centroid');
  }

  /**
   * The detectors that ran as their typed query (#4758): too few labels for a
   * trained detector, so their hits are the set their query's text sort keeps.
   * One line each, with the labels still owed. Not the held ones
   * ({@link heldDetectors}), whose labels meet the quota.
   */
  get queryDetectors(): string[] {
    return this.detectorsServedAs('query', false);
  }

  /**
   * The detectors that ran as their typed query past the quota (#4736, #4889):
   * a detector made from a typed query keeps its query's set until it has
   * {@link holdLabels} labels. One line each, with the labels still owed.
   */
  get heldDetectors(): string[] {
    return this.detectorsServedAs('query', true);
  }

  /**
   * Tune's training-domain warning, for each detector whose check says this
   * dataset looks unlike the one it was trained on (#5501): the same line, so
   * the warning reaches the results the user acts on, not only the view where
   * they check calls by hand. Its next step is the Browse button under the
   * list (#5545).
   */
  readonly domainWarnings = computed(() => {
    const dataset = this.data().dataset_name;
    const warnings: DomainShiftWarning[] = [];
    for (const [name, result] of Object.entries(this.data().results || {})) {
      const warning = domainShiftWarning(result.training_domain, result.detector_name || name, dataset, 'results');
      if (warning) warnings.push(warning);
    }
    return warnings;
  });

  /** The labels the typed query's hold lasts until, from the first held detector's quota. */
  get holdLabels(): number {
    const held = Object.values(this.data().results || {}).find((r) => (r.label_quota?.hold_owed ?? 0) > 0);
    return held?.label_quota?.hold_labels ?? 0;
  }

  private detectorsServedAs(tier: 'centroid' | 'query', held = false): string[] {
    const lines: string[] = [];
    for (const [name, result] of Object.entries(this.data().results || {})) {
      const quota = result.label_quota;
      if (quota?.tier !== tier || (quota.hold_owed > 0) !== held) continue;
      const owed = held
        ? `${quota.hold_owed} more label${quota.hold_owed === 1 ? '' : 's'}`
        : labelsOwed(quota);
      lines.push(owed ? `${result.detector_name || name} (${owed})` : result.detector_name || name);
    }
    return lines;
  }

  /** Whether the summary names the run's media type: not on a server locked
   *  to that one type, where it could be nothing else (#5106). */
  get showMediaType(): boolean {
    return !isSoloMediaType(this.settingsState.settingsSignal()?.solo_media_type, this.data().media_type);
  }

  /** The Good at the {@link notch}, across every detector. */
  get goodCount(): number {
    return this.splits().reduce((total, { split }) => total + split.good.length, 0);
  }

  /** The Bad at the {@link notch}, across every detector. */
  get badCount(): number {
    return this.splits().reduce((total, { split }) => total + split.bad.length, 0);
  }

  formatOrigin(hit: AutoDetectHit): string {
    const origin = hit.origin;
    if (!origin) return '';
    if (origin.params) {
      const firstVal = Object.values(origin.params)[0];
      if (firstVal) return `${origin.importer}(${firstVal})`;
    }
    return origin.importer || '';
  }

  /** Open a destination tab: Clipboard, AutoFind, or an exporter, its form seeded. */
  selectTab(name: string): void {
    this.activeTab.set(name);
    this.updateExporterFields();
  }

  private updateExporterFields(): void {
    const fields = (this.activeExporter?.fields ?? []) as ImporterField[];
    this.declaredExporterFields.set(fields);
    this.exportFieldValues = {};
    for (const field of fields) {
      if (field.default) {
        // Resolve the field's declared `template_vars` so a default like
        // `"{detector_name}"` shows the real name instead of the placeholder
        // (issue #3199); the server substitutes again, idempotently, on run.
        this.exportFieldValues[field.key] = this.templateVars.resolveDefault(field);
      }
    }
  }

  /** The exporter the open tab sends to, or none on the Clipboard and AutoFind tabs. */
  get activeExporter(): ExporterEntry | undefined {
    return this.exporters().find((e) => e.name === this.activeTab());
  }

  /** Why Export can't run now, or `''` when it can. */
  get exportBlocker(): string {
    if (this.activeTab() === AUTOFIND_TAB) {
      if (!this.data().run_id) return 'These results name no run to send';
      if (!this.autofindExporterName()) return 'Set up a results exporter on the Settings AutoFind tab';
      return '';
    }
    if (this.displayHits().length === 0) return 'Nothing is listed to export';
    if (this.dataColumns().length === 0) return 'Tick a column to export';
    return '';
  }

  /** What Export does from the open tab, for its tooltip. */
  get exportAction(): string {
    if (this.activeTab() === AUTOFIND_TAB) {
      return `Send this run's results via ${this.autofindExporterLabel()}, as a finished run does`;
    }
    const exp = this.activeExporter;
    return exp ? `Send the listed rows to ${exp.display_name || exp.name}` : 'Copy the listed rows to the clipboard';
  }

  /**
   * The run reshaped to hold exactly the rows the list shows, in its order.
   *
   * Exporters write a detector's `hits` and conventionally ignore
   * `negative_hits` (see `ResultsExporter.export_find_results`), so the listed
   * side is moved into `hits`: Bad exports the below-threshold rows, Both
   * exports every row with its `label` stamped, and `total_hits` counts what
   * is being sent. `selected_columns` carries the ticked columns, in the
   * user's order, which an exporter writing a table honours (#5464). A
   * detector split at another notch of the Threshold (#5540) sends that
   * notch's cut as `threshold` and its balance as `beta`, and no `balance`,
   * which describes the run's own cut.
   */
  exportPayload(): AutoDetectResultsData {
    const data = this.data();
    const listed = new Map<string, AutoDetectHit[]>();
    const hits = this.displayHits();
    this.sortedRows().forEach((row, i) => {
      const list = listed.get(row.resultKey) ?? [];
      list.push(hits[i]);
      listed.set(row.resultKey, list);
    });
    const results: Record<string, AutoDetectDetectorResult> = {};
    const notch = this.notch();
    for (const { resultKey: name, result, split } of this.splits()) {
      const rows = listed.get(name) ?? [];
      results[name] = { ...result, hits: rows, negative_hits: [], total_hits: rows.length };
      if (split.moved && notch != null) {
        results[name] = { ...results[name], threshold: split.threshold, beta: notch, balance: undefined };
      }
    }
    return {
      media_type: data.media_type,
      detectors_run: data.detectors_run,
      results,
      missing_detectors: data.missing_detectors ?? [],
      selected_columns: this.dataColumns().map((c) => c.key),
    };
  }

  /** Send the listed rows on from the open tab, and close after if asked. */
  async exportResults(closeAfter = false): Promise<void> {
    if (this.exporting() || this.exportBlocker) return;
    if (this.activeTab() === AUTOFIND_TAB) {
      this.exportAsAutoFind(closeAfter);
      return;
    }
    const exporter = this.activeExporter;
    const rowCount = this.displayHits().length;
    const counted = `${rowCount.toLocaleString()} result${rowCount === 1 ? '' : 's'}`;
    if (!exporter) {
      const copied = (await this.clipboard()?.copy()) ?? false;
      if (!copied) return;
      this.toast.success({ message: `Copied ${counted} to the clipboard`, dedupKey: 'autofind-results-export' });
      if (closeAfter) this.close();
      return;
    }
    const exporterLabel = exporter.display_name || exporter.name;
    // Claim the tab now, while this click still counts as user activation;
    // see `openBlankTab` (#2898).
    const pendingTab = exporter.opens_url ? openBlankTab() : null;
    this.exporting.set(true);
    this.exportersApi
      .runExport({
        exporter_name: exporter.name,
        field_values: { ...this.exportFieldValues },
        results: this.exportPayload() as unknown as Record<string, unknown>,
        payload_kind: 'find_results',
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (response) => {
          this.exporting.set(false);
          this.confirmExport(counted, exporterLabel, response?.open_url, response?.message, pendingTab, closeAfter);
        },
        error: () => {
          // The error interceptor toasts the server's reason.
          pendingTab?.close();
          this.exporting.set(false);
        },
      });
  }

  /**
   * Send the run through the exporter set up for AutoFind, as its finish did
   * (#5584): the server runs the same export with the saved field values, on
   * the run's own results (every detector's Good at the run's cut, whatever
   * is listed), and answers with the status a run records as `auto_export`.
   * The exporter's own failure comes back in that status, so it is toasted
   * here in the words the run's notice uses.
   */
  private exportAsAutoFind(closeAfter: boolean): void {
    const runId = this.data().run_id;
    if (!runId) return;
    const exporter = this.autofindExporter();
    const label = this.autofindExporterLabel();
    const count = this.autofindHitCount();
    const counted = `${count.toLocaleString()} result${count === 1 ? '' : 's'}`;
    const pendingTab = exporter?.opens_url ? openBlankTab() : null;
    this.exporting.set(true);
    this.findApi
      .exportAutofindRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (status: AutoFindExportStatus) => {
          this.exporting.set(false);
          if (!status.success) {
            pendingTab?.close();
            this.toast.error({
              message: 'The AutoFind export failed',
              detail: autoExportFailure(status),
              dedupKey: 'autofind-results-export',
            });
            return;
          }
          this.confirmExport(counted, label, status.open_url, status.message, pendingTab, closeAfter);
        },
        error: () => {
          pendingTab?.close();
          this.exporting.set(false);
        },
      });
  }

  /** Toast a finished export, opening its URL if it returned one, and close after if asked. */
  private confirmExport(
    counted: string,
    exporterLabel: string,
    url: string | undefined,
    message: string | undefined,
    pendingTab: ReturnType<typeof openBlankTab>,
    closeAfter: boolean,
  ): void {
    const openUrl = safeExternalUrl(url);
    let opened = false;
    if (openUrl) {
      opened = pendingTab ? pendingTab.navigate(openUrl) : openExternalUrl(openUrl);
    } else {
      pendingTab?.close();
    }
    this.toast.success({
      message: opened ? `Opened ${counted} in ${exporterLabel}` : `Exported ${counted} to ${exporterLabel}`,
      detail: openUrl && !opened ? 'Your browser blocked the new tab.' : message,
      action: openUrl ? { label: 'Open', title: openUrl, onClick: () => openExternalUrl(openUrl) } : undefined,
      autoDismissMs: openUrl && !opened ? 0 : undefined,
      dedupKey: 'autofind-results-export',
    });
    if (closeAfter) this.close();
  }

  /** The listed rows' media ids, once each: what Browse lays out. A media
   *  two detectors both called Good is one item on the map. */
  get browseIds(): number[] {
    return runBrowseIds(this.data(), this.sides(), this.notch());
  }

  /** Why Browse is unavailable, or `''` when it isn't. */
  get browseBlocker(): string {
    if (!this.data().dataset_id || !this.data().run_id) return 'These results name no dataset to browse';
    if (this.browseIds.length === 0) return 'No listed items to browse';
    return '';
  }

  /**
   * Open the listed items in Browse, as a map of their own. As in the Tune
   * view, the map is built here, behind a progress bar with a Cancel, and the
   * dialog hands off to Browse only once it is ready; the run's dataset is
   * made active first, since the build scores against the active dataset.
   * Browse's Back reopens these results.
   */
  browse(): void {
    const datasetId = this.data().dataset_id;
    const runId = this.data().run_id;
    const ids = this.browseIds;
    const sides = this.sides();
    const notch = this.notch();
    if (this.browseBlocker || !datasetId || !runId || this.browseStarting() || this.browsePrep.preparing()) return;
    // Browse's own build takes over from here: it attaches to a fit the
    // warm-up already started, or finds the map built.
    this.stopWarm();
    this.browseStarting.set(true);
    // Completes without emitting when the load fails; its own toast says why.
    this.contextSwitch
      .applyActivePair(datasetId, this.activeContext.modelId || '')
      .pipe(
        take(1),
        finalize(() => this.browseStarting.set(false)),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(() =>
        this.browsePrep.start(
          datasetId,
          ids,
          { kind: 'results', runId, sides, ...(notch != null ? { notch } : {}) },
          () => this.closed.emit(),
        ),
      );
  }

  close(): void {
    // A map still building would navigate away from wherever the user went.
    if (this.browsePrep.preparing()) this.browsePrep.cancel();
    this.closed.emit();
  }

  /**
   * Get run *runId*'s Good results ready to Browse while the user reads them
   * (#4683): the server lays them out as the Browse button would, but only
   * when it has nothing else to do, answering `busy` otherwise. So this asks
   * again, easing off, for as long as the dialog stays open and the answer
   * stays `busy`; any other answer (`building`, `ready`, `skipped`) or a
   * failure ends it. Nothing is shown: the user only notices a Browse that
   * opens sooner.
   */
  private warmBrowse(runId: string, datasetId: string): void {
    if (runId === this.warmRunId) return;
    this.stopWarm();
    this.warmRunId = runId;
    if (!runId || !datasetId) return;
    // A hint, not an action of the user's: never toast its failures.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    const ask = (): Observable<AutoFindBrowsePrepResponse> =>
      this.viewingSubsetOf(datasetId)
        ? of<AutoFindBrowsePrepResponse>({ status: 'busy' })
        : this.findApi
            .prepAutofindRunBrowse(runId, context)
            .pipe(catchError(() => of<AutoFindBrowsePrepResponse>({ status: 'skipped' })));
    this.warmSub = adaptivePoll(ask, {
      fastMs: AutoDetectResultsModalComponent.WARM_FAST_MS,
      slowMs: AutoDetectResultsModalComponent.WARM_SLOW_MS,
    })
      .pipe(takeWhile((answer) => answer.status === 'busy'))
      .subscribe();
  }

  private stopWarm(): void {
    this.warmSub?.unsubscribe();
    this.warmSub = null;
  }

  /**
   * Whether this tab is showing a subset map of *datasetId*. A dataset holds
   * one subset layout at a time, so laying out these results would replace
   * the map under the user's eyes; that waits until they have moved on.
   */
  private viewingSubsetOf(datasetId: string): boolean {
    const tree = this.router.parseUrl(this.router.url);
    const segments = tree.root.children['primary']?.segments.map((s) => s.path) ?? [];
    return segments[0] === 'browse' && segments[1] === datasetId && !!tree.queryParams['subset'];
  }
}

/** ``"file_size"`` → ``"File size"``: a metadata key as a column name. */
function humanizeKey(key: string): string {
  const spaced = key.replace(/_/g, ' ').trim();
  return spaced ? spaced[0].toUpperCase() + spaced.slice(1) : key;
}
