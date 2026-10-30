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
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
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
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import type { MediaSimilarityOrder } from '../../../generated/api-client/models/media-similarity-order';
import { IconComponent } from '../../icon/icon.component';
import { apiErrorMessage } from '../../../utils/api-error';
import { autoExportFailure } from '../../../utils/auto-export-status';
import { exporterIconKind } from '../../../utils/exporter-icon';
import { openBlankTab, openExternalUrl, safeExternalUrl } from '../../../utils/external-url';
import { iconSizeToGoalWidth } from '../../../utils/grid-icon-size';
import { visibleFields } from '../../../utils/plugin-fields';
import { labelsOwed } from '../../../utils/label-quota';
import { coerceNonEmptyString } from '../../../utils/settings-coerce';
import { isSoloMediaType } from '../../../utils/solo-media-type';
import { PluginCheckboxComponent } from '../../plugin-checkbox/plugin-checkbox.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

/** Which side of each detector's threshold the list shows. */
export type ResultsSides = 'good' | 'bad' | 'both';
/** What the list is ordered by: score, or look-alikes side by side. */
export type ResultsSort = 'score' | 'similarity';

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
  /** The column's track in the list's CSS grid. */
  track: string;
  displayOnly?: boolean;
}

/** The Similarity sort's state: the run's ids ranked once the order lands. */
type SimilarityState =
  | { status: 'idle' }
  | { status: 'building'; progress: MediaSimilarityOrder | null }
  | { status: 'ready'; rank: ReadonlyMap<number, number> }
  | { status: 'error'; message: string };

/** How long the scroll must rest before the thumbnails on screen load (ms). */
export const THUMB_SETTLE_MS = 120;
/** A row with no thumbnail column (px): one line of text. */
const TEXT_ROW_PX = 30;
/** Space above and below a thumbnail in its row (px). */
const THUMB_PAD_PX = 2;
/** The thumbnail column's track: its tile plus the cell's side padding (px). */
const THUMB_CELL_PAD_PX = 12;

/** The Find Results dialog: one Find or AutoFind run's hits on one dataset.
 *
 * Laid out by what each control acts on (#5464): above the list, what the list
 * shows (Good / Bad / Both, the sort, the columns, the thumbnail size); the
 * list; below it, what acts on the listed rows (Browse, which opens them as a
 * map of their own, #4615; and where they go, a tab per destination with
 * Clipboard first); and the footer's Close / Export / Export and Close. While
 * the dialog is open, the server lays out the Good results for Browse whenever
 * it has nothing else to do (#4683). Mounted once in `AppComponent`, fed by
 * `AutoFindService`.
 *
 * The list is virtual (a run's Both can be the whole dataset), its rows a CSS
 * grid of the chosen columns; thumbnails load once the scroll rests and stay
 * loaded, as New Detector's picker grid does. The Similarity sort orders the
 * rows by the dataset's 1-D UMAP, fit with its Browse layout: one the dataset
 * has answers at once, one it lacks is fit while the dialog waits on it. */
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

  // --- What the list shows --------------------------------------------------

  /** Which side of the threshold is listed (and exported, and browsed). */
  readonly sides = signal<ResultsSides>('good');
  readonly sortMode = signal<ResultsSort>('score');

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
    for (const [resultKey, result] of Object.entries(this.data().results || {})) {
      const detector = result.detector_name || resultKey;
      const add = (hit: AutoDetectHit, label: 'good' | 'bad') =>
        rows.push({ hit, label, resultKey, detector, key: `${resultKey}|${label}|${hit.id ?? hit.md5}` });
      if (sides !== 'bad') for (const hit of result.hits || []) add(hit, 'good');
      if (sides !== 'good') for (const hit of result.negative_hits || []) add(hit, 'bad');
    }
    return rows;
  });

  /** {@link rows} in the chosen order. Score: highest first, across every
   *  detector. Similarity: the dataset's 1-D order, once it has landed (score
   *  until then); ties, and rows the order doesn't place, by score. */
  readonly sortedRows = computed<ResultRow[]>(() => {
    const byScore = (a: ResultRow, b: ResultRow) => (b.hit.score ?? -Infinity) - (a.hit.score ?? -Infinity);
    const rows = [...this.rows()];
    const similarity = this.similarity();
    if (this.sortMode() === 'similarity' && similarity.status === 'ready') {
      const rank = (row: ResultRow) => (row.hit.id != null ? similarity.rank.get(row.hit.id) : undefined) ?? Infinity;
      return rows.sort((a, b) => rank(a) - rank(b) || byScore(a, b));
    }
    return rows.sort(byScore);
  });

  /** The listed hits in order, each stamped with its side when both are listed. */
  readonly displayHits = computed<AutoDetectHit[]>(() =>
    this.sortedRows().map((row) => (this.sides() === 'both' ? { ...row.hit, label: row.label } : row.hit)),
  );

  // --- Columns ----------------------------------------------------------------

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

  /** The columns the list can show now, ticked or not. Label only lists both
   *  sides; Detector only a run of several; Thumbnail only a type with them.
   *  The metadata columns start unticked, the rest ticked. */
  readonly columns = computed<ResultsColumn[]>(() => {
    const ticks = this.columnTicks();
    const col = (key: string, label: string, track: string, extra: Partial<ResultsColumn> = {}, on = true) => ({
      key,
      label,
      track,
      ...extra,
      enabled: ticks[key] ?? on,
    });
    const cols: ResultsColumn[] = [];
    if (this.hasThumbnails()) {
      cols.push(
        col('thumbnail', 'Thumbnail', `${this.thumbPx() + THUMB_CELL_PAD_PX}px`, {
          displayOnly: true,
          title: 'Shown in the list; not copied or exported',
        }),
      );
    }
    cols.push(col('origin_name', 'Name', 'minmax(6rem, 2fr)'));
    cols.push(col('score', 'Score', '5rem'));
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

  /** The ticked columns, in order: what the list shows. */
  readonly listedColumns = computed(() => this.columns().filter((c) => c.enabled));
  /** The ticked columns that are data: what the clipboard copies and an export sends. */
  private readonly dataColumns = computed(() => this.listedColumns().filter((c) => !c.displayOnly));
  /** The list's (and its header's) CSS grid columns. */
  readonly gridTemplate = computed(() => this.listedColumns().map((c) => c.track).join(' '));

  setColumnEnabled(key: string, enabled: boolean): void {
    this.columnTicks.update((ticks) => ({ ...ticks, [key]: enabled }));
  }

  /** From [x] (all ticked), clear every column; from [ ]/[-], tick them all. */
  toggleAllColumns(): void {
    const columns = this.columns();
    const on = !columns.every((c) => c.enabled);
    this.columnTicks.update((ticks) => ({ ...ticks, ...Object.fromEntries(columns.map((c) => [c.key, on])) }));
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

  // --- The Similarity sort -------------------------------------------------------

  readonly similarity = signal<SimilarityState>({ status: 'idle' });
  /** How often to ask again while the dataset's order is being fit (ms). */
  static readonly SIMILARITY_POLL_MS = 1500;
  private similaritySub: Subscription | null = null;
  private similarityTimer: ReturnType<typeof setTimeout> | null = null;

  /** Order the list by score, or by similarity (asking for the order if it
   *  isn't in hand, or asking again after a failure). */
  setSort(mode: ResultsSort): void {
    this.sortMode.set(mode);
    const status = this.similarity().status;
    if (mode === 'similarity' && (status === 'idle' || status === 'error')) this.requestSimilarity();
  }

  /** Back to the score order, and stop waiting on the fit (which carries on
   *  on the server, so a later Similarity finds it further along). */
  cancelSimilarity(): void {
    this.stopSimilarity();
    this.similarity.set({ status: 'idle' });
    this.sortMode.set('score');
  }

  private requestSimilarity(): void {
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

  /** Why the order could not be had, while the Similarity sort is in error. */
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
  readonly exporters = signal<ExporterEntry[]>([]);
  /** The open destination tab: `'clipboard'` (the default) or an exporter's name. */
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
    // A new run's ids need their own order: ask again if it is the sort.
    effect(() => {
      this.data();
      untracked(() => {
        this.stopSimilarity();
        this.similarity.set({ status: 'idle' });
        if (this.sortMode() === 'similarity') this.requestSimilarity();
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
      this.rangeSub?.unsubscribe();
      if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    });
  }

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (list) => {
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

  get goodCount(): number {
    let total = 0;
    for (const result of Object.values(this.data().results || {})) {
      total += (result.hits || []).length;
    }
    return total;
  }

  get badCount(): number {
    let total = 0;
    for (const result of Object.values(this.data().results || {})) {
      total += (result.negative_hits || []).length;
    }
    return total;
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

  /** Open a destination tab: Clipboard, or an exporter, its form seeded. */
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

  /** The exporter the open tab sends to, or none on the Clipboard tab. */
  get activeExporter(): ExporterEntry | undefined {
    return this.exporters().find((e) => e.name === this.activeTab());
  }

  /** Why Export can't run now, or `''` when it can. */
  get exportBlocker(): string {
    if (this.displayHits().length === 0) return 'Nothing is listed to export';
    if (this.dataColumns().length === 0) return 'Tick a column to export';
    return '';
  }

  /** What Export does from the open tab, for its tooltip. */
  get exportAction(): string {
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
   * is being sent. `selected_columns` carries the ticked columns, which an
   * exporter writing a table honours (#5464).
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
    for (const [name, result] of Object.entries(data.results || {})) {
      const rows = listed.get(name) ?? [];
      results[name] = { ...result, hits: rows, negative_hits: [], total_hits: rows.length };
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
          const openUrl = safeExternalUrl(response?.open_url);
          let opened = false;
          if (openUrl) {
            opened = pendingTab ? pendingTab.navigate(openUrl) : openExternalUrl(openUrl);
          } else {
            pendingTab?.close();
          }
          this.toast.success({
            message: opened ? `Opened ${counted} in ${exporterLabel}` : `Exported ${counted} to ${exporterLabel}`,
            detail: openUrl && !opened ? 'Your browser blocked the new tab.' : response?.message,
            action: openUrl ? { label: 'Open', title: openUrl, onClick: () => openExternalUrl(openUrl) } : undefined,
            autoDismissMs: openUrl && !opened ? 0 : undefined,
            dedupKey: 'autofind-results-export',
          });
          if (closeAfter) this.close();
        },
        error: () => {
          // The error interceptor toasts the server's reason.
          pendingTab?.close();
          this.exporting.set(false);
        },
      });
  }

  /** The listed rows' media ids, once each: what Browse lays out. A media
   *  two detectors both called Good is one item on the map. */
  get browseIds(): number[] {
    return runBrowseIds(this.data(), this.sides());
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
        this.browsePrep.start(datasetId, ids, { kind: 'results', runId, sides }, () => this.closed.emit()),
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
