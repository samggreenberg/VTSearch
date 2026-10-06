import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';

import type { DatasetDomainShiftResponse } from '../../../generated/api-client/models/dataset-domain-shift-response';
import type { FindEvidenceCoverageResponse } from '../../../generated/api-client/models/find-evidence-coverage-response';
import type { FindStatsResponse } from '../../../generated/api-client/models/find-stats-response';
import type { LineTestEdge } from '../../../generated/api-client/models/line-test-edge';
import type { LineTestResponse } from '../../../generated/api-client/models/line-test-response';
import type { DatasetRegistryEntry } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import { ActiveDatasetService } from '../../../services/active-dataset.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DetectorsFindApiService } from '../../../services/detectors-find-api.service';
import { PairScopeService } from '../../../services/pair-scope.service';
import { apiErrorMessage } from '../../../utils/api-error';
import { formatTimestamp } from '../../../utils/format-date';
import { BALANCE_PRESETS } from '../../../utils/line-balance';
import {
  estimatePercent,
  fbetaHeadline,
  lineTestPhase,
  widthLight,
  type LineTestLight,
  type LineTestPhase,
} from '../../../utils/line-test';

/** A tick on the chart's log-scale count axis. */
interface XTick {
  x: number;
  label: string;
}

/** A band row of the picks table. */
interface BandRow {
  index: number;
  side: 'above' | 'below';
  where: string;
  labelled: number;
  right: number;
  range: string;
}

/** Compact count label for an axis tick: 1, 10, 100, 1k, 10k, 1M. */
function compactCount(n: number): string {
  if (n >= 1_000_000) return `${n / 1_000_000}M`;
  if (n >= 1_000) return `${n / 1_000}k`;
  return `${n}`;
}

/**
 * The Test autopilot's right pane (#4524): the result as it forms. The picks
 * so far by band, the precision and recall ranges with their lights, the
 * F-beta headline, the band-resolution precision curve (which replaces the
 * retired Stats modal's *Checked by you* curve: unbiased, from the uniform
 * picks alone), and the two trust chips (Training-domain overlap, Evidence
 * coverage) that were behind the Stats button. At Done, the verdict with its
 * three exits, **Move to AutoRun**, **Lean the Threshold** (the ranges each
 * preset would ship, re-estimated from the picks already taken) and **Add
 * Corrections and retrain**; and *Nothing to test* when the line keeps fewer
 * items than one round.
 *
 * Every number here comes from the server's draws (`/api/line-test`); the
 * pane only draws them. The Stats modal's 2×2 of the session's checks stays,
 * folded under the result, read from `/api/find/stats` once the test is done.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-line-test-result',
  standalone: true,
  imports: [DecimalPipe],
  templateUrl: './line-test-result.component.html',
  styleUrl: './line-test-result.component.scss',
})
export class LineTestResultComponent {
  private readonly findApi = inject(DetectorsFindApiService);
  private readonly registryApi = inject(DatasetsRegistryApiService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly activeCtx = inject(ActiveContextService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly pairScope = inject(PairScopeService);
  private readonly destroyRef = inject(DestroyRef);

  /** The server's last word; null before any. */
  readonly response = input<LineTestResponse | null>(null);
  /** The Find pass is scoring: nothing to show yet. */
  readonly scoring = input(false);
  /** Bumped by the host when the session's checks changed under a finished test (corrections, Review votes). */
  readonly refresh = input(0);

  /** Move the detector to AutoRun. */
  readonly moveToAutoRun = output<void>();
  /** Lean the Threshold to a preset (a beta). */
  readonly lean = output<number>();
  /** Fold the session's corrections into the detector and retrain. */
  readonly addCorrections = output<void>();
  /** Test the line as it stands now (after it moved). */
  readonly testAgain = output<void>();

  readonly phase = computed<LineTestPhase>(() => (this.scoring() ? 'score' : lineTestPhase(this.response())));
  readonly test = computed(() => this.response()?.test ?? null);
  readonly estimates = computed(() => this.test()?.estimates ?? null);
  readonly done = computed(() => this.phase() === 'done');
  readonly nothing = computed(() => this.phase() === 'nothing');
  readonly moved = computed(() => this.response()?.moved ?? false);
  readonly stale = computed(() => this.response()?.stale ?? false);
  /**
   * The detector has a class model to count the matches below the bands the
   * test reached. A structural or document detector has none, so its recall
   * is unmeasured below them and is read in words only (#4523, #4542).
   */
  readonly classModel = computed(() => this.test()?.class_model ?? true);
  /** A test resumed from the verdict the detector keeps (#4526) says so: the user never voted these picks today. */
  readonly keptNote = computed(() => {
    const t = this.test();
    if (!t?.kept_at) return '';
    const date = formatTimestamp(t.kept_at, { withTime: false });
    return (
      `Kept from your test of ${date}: the detector and its ranking of this collection are unchanged since, ` +
      `so its ${t.labelled} picks still hold and are in the Review tab's piles.`
    );
  });

  readonly precisionText = computed(() => {
    const e = this.estimates();
    return e ? estimatePercent(e.precision) : '';
  });
  readonly recallText = computed(() => {
    const e = this.estimates();
    if (!e) return '';
    return this.classModel() ? `${e.found} (${estimatePercent(e.recall)})` : e.found;
  });
  readonly foundTitle = computed(() =>
    this.classModel()
      ? "Of all the matches in the collection, the share the line keeps: random picks below the line, read against the detector's own count of what the unchecked tail holds."
      : 'Of all the matches in the collection, the share the line keeps, as far as the picks below the line reach: this detector has no count of its own for the rest of the list.',
  );
  readonly fbetaText = computed(() => {
    const e = this.estimates();
    return e ? fbetaHeadline(e.fbeta) : '';
  });
  readonly precisionLight = computed<LineTestLight>(() => {
    const t = this.test();
    return widthLight(t?.report?.matches_width, t?.budgets?.matches_width ?? 0.2);
  });
  readonly recallLight = computed<LineTestLight>(() => {
    const t = this.test();
    return widthLight(t?.report?.misses_width, t?.budgets?.misses_width ?? 0.25);
  });
  /**
   * The deep tail the walk never reached is the model's word, and said to be;
   * with no class model nothing counts it, and the note says recall is
   * unmeasured there.
   */
  readonly tailNote = computed(() => {
    const e = this.estimates();
    const t = this.test();
    if (!e || !t) return '';
    if (!this.classModel()) {
      const reached = Math.max(t.line_count, ...t.bands.filter((b) => b.side === 'below' && b.labelled > 0).map((b) => b.hi));
      return (
        `This detector has no class model, so nothing estimates the matches below the top ${reached.toLocaleString()}: ` +
        `Found counts only those the picks turned up, and is a reading in words, not a measurement.`
      );
    }
    if (!e.tail_from_model) return '';
    return `The deepest part of the list was not checked: the ${Math.round(e.tail_positives)} matches it likely holds are the detector's own estimate.`;
  });

  readonly bandRows = computed<BandRow[]>(() => {
    const t = this.test();
    if (!t) return [];
    return t.bands
      .filter((b) => b.labelled > 0)
      .map((b) => ({
        index: b.index,
        side: b.side,
        where: b.lo === 1 ? `top ${b.hi.toLocaleString()}` : `${b.lo.toLocaleString()}–${b.hi.toLocaleString()}`,
        labelled: b.labelled,
        right: b.right,
        range: b.range ? estimatePercent(b.range) : '',
      }));
  });

  /** The line each preset would draw, from the picks already taken, left to right as the control offers them. */
  readonly presetRows = computed(() => {
    const presets = this.response()?.presets ?? [];
    const current = this.test()?.beta ?? null;
    return BALANCE_PRESETS.map((preset) => {
      const p = presets.find((row) => row.beta === preset.value) ?? null;
      return { beta: preset.value, hint: preset.hint, current: current === preset.value, estimate: p };
    }).filter((row) => row.estimate !== null);
  });

  /** The verdict in one line: a reading of the ranges, not a threshold the app enforces. */
  readonly verdict = computed(() => {
    const e = this.estimates();
    const t = this.test();
    if (!e || !t) return '';
    const found = this.classModel() ? `${e.found} (${estimatePercent(e.recall)} of all the matches)` : e.found;
    return (
      `On this collection the line keeps the top ${t.line_count.toLocaleString()}: likely ${estimatePercent(e.precision)} ` +
      `of them are right, with ${found}. ` +
      `That is what AutoRun would ship from a collection like this one.`
    );
  });

  // --- The session's checks: the Stats modal's 2×2, kept under the result ---
  readonly stats = signal<FindStatsResponse | null>(null);
  readonly statsError = signal('');

  // --- Training-domain overlap (coverage-atlas domain-shift report) --------
  // Compares the active dataset against a reference dataset's coverage atlas,
  // the dataset the detector was trained on. The reference can't be inferred
  // (a handed-over detector may not carry its haystack), so the user picks it
  // from the loaded datasets with a matching embedder. See
  // docs/plans/coverage-atlas.md §6.5.
  readonly refCandidates = signal<DatasetRegistryEntry[]>([]);
  readonly selectedRefId = signal<string>('');
  readonly domainLoading = signal(false);
  readonly domainError = signal('');
  readonly domainShift = signal<DatasetDomainShiftResponse | null>(null);

  // --- Evidence coverage (labelset-kNN, cross-user by construction) ---------
  // The complement to the domain-shift report: it asks only what the detector
  // carries, so it works whenever a Find run has been scored. See
  // docs/plans/coverage-atlas.md §6.1.
  readonly evidence = signal<FindEvidenceCoverageResponse | null>(null);

  // --- The band-resolution precision curve ----------------------------------
  readonly hoverEdge = signal<LineTestEdge | null>(null);
  readonly chartWidth = signal(280);
  readonly chartHeight = 150;
  private readonly chartSvg = viewChild<ElementRef<SVGSVGElement>>('chartSvg');
  private resizeObserver: ResizeObserver | null = null;
  private readonly padLeft = 36;
  private readonly padRight = 10;
  private readonly padTop = 8;
  private readonly padBottom = 30;

  /** The edges with picks behind them, in count order: the points of the curve. */
  readonly edges = computed<LineTestEdge[]>(() => {
    const e = this.estimates();
    return e ? [...e.at_edges].sort((a, b) => a.count - b.count) : [];
  });

  private loadedFor: string | null = null;

  constructor() {
    this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
    // The chart mounts once a test exists, so observe it when it appears.
    effect(() => {
      const svg = this.chartSvg()?.nativeElement;
      this.resizeObserver?.disconnect();
      this.resizeObserver = null;
      if (!svg || typeof ResizeObserver === 'undefined') return;
      this.resizeObserver = new ResizeObserver((entries) => {
        const width = Math.round(entries[0]?.contentRect.width ?? 0);
        if (width > 0) this.chartWidth.set(width);
      });
      this.resizeObserver.observe(svg);
    });
    // The checks' 2×2 and the trust chips are read once the verdict is up, and
    // again when the host says the checks changed under it.
    effect(() => {
      const done = this.done();
      const refresh = this.refresh();
      const key = `${this.activeCtx.datasetId}/${this.activeCtx.modelId}/${refresh}`;
      if (!done) {
        untracked(() => {
          if (this.loadedFor !== null) this.forget();
        });
        return;
      }
      if (this.loadedFor === key) return;
      this.loadedFor = key;
      untracked(() => this.loadChecks(refresh === 0 || this.evidence() === null));
    });
  }

  private forget(): void {
    this.loadedFor = null;
    this.stats.set(null);
    this.statsError.set('');
  }

  private loadChecks(withTrust: boolean): void {
    this.findApi
      .getFindStats()
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (data) => {
          this.stats.set(data);
          this.statsError.set('');
        },
        error: (err) => this.statsError.set(apiErrorMessage(err, 'Could not read the checks.')),
      });
    if (!withTrust) return;
    this.initDomainOverlap();
    // Best-effort: the chip stays hidden on error or when unavailable.
    this.findApi
      .getEvidenceCoverage()
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (data) => this.evidence.set(data.available ? data : null),
        error: () => this.evidence.set(null),
      });
  }

  /** Populate the reference-dataset picker from the loaded registry and, with a candidate, run the check. */
  private initDomainOverlap(): void {
    const activeId = this.activeCtx.datasetId;
    const activeEmbedder = this.activeDataset.dataset()?.embedder ?? '';
    const candidates = this.datasetState.datasets.filter(
      (d) => d.loaded && d.id !== activeId && (d.embedder ?? '') === activeEmbedder,
    );
    this.refCandidates.set(candidates);
    if (candidates.length > 0) this.selectRef(candidates[0].id);
  }

  /** Run the domain-shift report against *refId*'s coverage atlas, or clear it. */
  selectRef(refId: string): void {
    this.selectedRefId.set(refId);
    this.domainShift.set(null);
    this.domainError.set('');
    if (!refId) return;
    this.domainLoading.set(true);
    this.registryApi
      .domainShift(refId)
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (data) => {
          this.domainShift.set(data);
          this.domainLoading.set(false);
        },
        error: (err) => {
          this.domainError.set(apiErrorMessage(err, 'Domain check unavailable'));
          this.domainLoading.set(false);
        },
      });
  }

  onRefChange(event: Event): void {
    this.selectRef((event.target as HTMLSelectElement).value);
  }

  get selectedRefName(): string {
    const id = this.selectedRefId();
    return this.refCandidates().find((d) => d.id === id)?.name ?? id;
  }

  get atypicalPct(): number {
    const d = this.domainShift();
    return d ? Math.round(d.frac_atypical * 100) : 0;
  }

  get overlapShifted(): boolean {
    return this.domainShift()?.shifted ?? false;
  }

  get unsupportedPct(): number {
    const e = this.evidence();
    return e ? Math.round(e.frac_unsupported * 100) : 0;
  }

  get evidenceUnsupported(): boolean {
    return this.evidence()?.unsupported ?? false;
  }

  get agreementPct(): string {
    const s = this.stats();
    return s ? `${Math.round(s.agreement_rate * 100)}%` : '-';
  }

  // --- chart geometry -------------------------------------------------------

  private get plotW(): number {
    return this.chartWidth() - this.padLeft - this.padRight;
  }

  private get plotH(): number {
    return this.chartHeight - this.padTop - this.padBottom;
  }

  /** Right end of the log-scale count axis (at least 10, so one decade shows). */
  private get xMax(): number {
    return Math.max(this.test()?.size ?? 0, 10);
  }

  xFor(count: number): number {
    return this.padLeft + (Math.log10(Math.max(count, 1)) / Math.log10(this.xMax)) * this.plotW;
  }

  yFor(precision: number): number {
    return this.padTop + (1 - precision) * this.plotH;
  }

  get xTicks(): XTick[] {
    const ticks: XTick[] = [];
    for (let n = 1; n <= this.xMax; n *= 10) ticks.push({ x: this.xFor(n), label: compactCount(n) });
    return ticks;
  }

  get curvePolyline(): string {
    return this.edges()
      .map((e) => `${this.xFor(e.count).toFixed(1)},${this.yFor(e.precision.point).toFixed(1)}`)
      .join(' ');
  }

  /** X position of the line's marker. */
  get cutX(): number | null {
    const t = this.test();
    return t && t.line_count > 0 ? this.xFor(t.line_count) : null;
  }

  get axisTop(): number {
    return this.padTop;
  }

  get axisBottom(): number {
    return this.chartHeight - this.padBottom;
  }

  get axisLeft(): number {
    return this.padLeft;
  }

  get axisRight(): number {
    return this.chartWidth() - this.padRight;
  }

  /** Snap the hover readout to the band edge nearest the pointer (in log space). */
  onChartMove(event: MouseEvent): void {
    const edges = this.edges();
    const svg = event.currentTarget as SVGSVGElement | null;
    if (!svg || edges.length === 0) return;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0) return;
    const x = ((event.clientX - rect.left) / rect.width) * this.chartWidth();
    let best = edges[0];
    let bestDist = Infinity;
    for (const e of edges) {
      const d = Math.abs(this.xFor(e.count) - x);
      if (d < bestDist) {
        best = e;
        bestDist = d;
      }
    }
    this.hoverEdge.set(best);
  }

  onChartLeave(): void {
    this.hoverEdge.set(null);
  }

  /** The edge the readout describes: the one under the pointer, else the line's own. */
  get readoutEdge(): LineTestEdge | null {
    const hover = this.hoverEdge();
    if (hover) return hover;
    const t = this.test();
    return t ? (this.edges().find((e) => e.count === t.line_count) ?? null) : null;
  }

  edgeReadout(e: LineTestEdge): string {
    return `Top ${e.count.toLocaleString()}: likely ${estimatePercent(e.precision)} right, ${e.found}`;
  }

  edgeTitle(e: LineTestEdge): string {
    const found = this.classModel() ? `${estimatePercent(e.recall)} of all the matches found` : e.found;
    return (
      `If the line kept the top ${e.count.toLocaleString()}: likely ${estimatePercent(e.precision)} right, ` +
      `${found}, F-beta ${fbetaHeadline(e.fbeta)}.`
    );
  }

  presetTitle(row: { hint: string; estimate: { count: number; precision: { lo: number; hi: number }; recall: { lo: number; hi: number } } | null }): string {
    const e = row.estimate;
    if (!e) return row.hint;
    return `${row.hint}. The line would keep the top ${e.count.toLocaleString()}: likely ${estimatePercent(e.precision)} right, ${estimatePercent(e.recall)} of all the matches found.`;
  }

  pct(p: number | null | undefined): string {
    return p == null ? '-' : `${Math.round(p * 100)}%`;
  }
}
