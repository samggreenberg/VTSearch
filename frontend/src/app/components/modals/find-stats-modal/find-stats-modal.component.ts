import { ChangeDetectionStrategy, Component, DestroyRef, effect, ElementRef, inject, OnInit, output, signal, viewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsFindApiService } from '../../../services/detectors-find-api.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ActiveContextService } from '../../../services/active-context.service';
import { ActiveDatasetService } from '../../../services/active-dataset.service';
import type { FindStatsResponse } from '../../../generated/api-client/models/find-stats-response';
import type { FindStatsPrecisionPoint } from '../../../generated/api-client/models/find-stats-precision-point';
import type { FindEvidenceCoverageResponse } from '../../../generated/api-client/models/find-evidence-coverage-response';
import type { DatasetDomainShiftResponse } from '../../../generated/api-client/models/dataset-domain-shift-response';
import type { DatasetRegistryEntry } from '../../../models/api.models';
import { apiErrorMessage } from '../../../utils/api-error';
import { foundWords, lineBalanceFrom, rangePercent, rangeTitle, type LikelyRange } from '../../../utils/line-balance';

/** A tick on the precision chart's log-scale x axis. */
interface XTick {
  x: number;
  label: string;
}

/** Compact count label for an axis tick: 1, 10, 100, 1k, 10k, 1M. */
function compactCount(n: number): string {
  if (n >= 1_000_000) return `${n / 1_000_000}M`;
  if (n >= 1_000) return `${n / 1_000}k`;
  return `${n}`;
}

/**
 * Detector-evaluation Stats for a Find run: the 2×2 confusion of the adopted
 * label set against the detector's original call, the derived agreement /
 * kept rates, and the headline precision-vs-returned chart (#4242) rendered as
 * a dependency-free inline SVG line chart: the verified precision of the
 * checked items in the top N, on a log-scale count axis with the line (the
 * current cut) marked. At the line, the spot check's likely range for the
 * share of the set the line keeps that is right stands as a bar (#4273); the
 * note under the chart adds how many of all the matches it found, in words
 * (#4413). There is no horizontal line to draw: a balance is a preference,
 * not a precision the line is asked to keep. No model-based estimate is
 * drawn: the #4220 estimator could not back an "at least" (#4360).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-find-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './find-stats-modal.component.html',
  styleUrl: './find-stats-modal.component.scss',
})
export class FindStatsModalComponent implements OnInit {
  private findApi = inject(DetectorsFindApiService);
  private registryApi = inject(DatasetsRegistryApiService);
  private datasetState = inject(DatasetStateService);
  private activeCtx = inject(ActiveContextService);
  private activeDataset = inject(ActiveDatasetService);

  private destroyRef = inject(DestroyRef);

  readonly closed = output<void>();

  // Signalized so the `ngOnInit` subscribe (an unpatched callback under zoneless)
  // schedules CD when the stats land.
  readonly loading = signal(true);
  readonly error = signal('');
  readonly stats = signal<FindStatsResponse | null>(null);

  // --- Training-domain overlap (coverage-atlas domain-shift report) --------
  // Compares the active (Find) dataset against a reference dataset's coverage
  // atlas — the dataset the detector was trained on. The reference can't be
  // inferred reliably (a handed-over detector may not carry its haystack), so
  // the user picks it from the datasets currently loaded with a matching
  // embedder. See docs/plans/coverage-atlas.md §6.5.
  readonly refCandidates = signal<DatasetRegistryEntry[]>([]);
  readonly selectedRefId = signal<string>('');
  readonly domainLoading = signal(false);
  readonly domainError = signal('');
  readonly domainShift = signal<DatasetDomainShiftResponse | null>(null);

  // --- Evidence coverage (labelset-kNN, cross-user by construction) ---------
  // The complement to the domain-shift report above: that one needs the
  // *training* dataset loaded with a built atlas (absent in a real handoff);
  // this asks only what the detector carries — its labelset — so it works
  // whenever a Find run has been scored, with no reference dataset. Reports the
  // share of the dataset the detector is calling without labeled evidence
  // behind the call. See docs/plans/coverage-atlas.md §6.1 (phase v0).
  readonly evidence = signal<FindEvidenceCoverageResponse | null>(null);

  // Index into `precision_curve` of the point under the pointer, or null.
  readonly hoverIndex = signal<number | null>(null);

  // Chart geometry.  The viewBox tracks the chart's rendered width, so one user
  // unit is one CSS pixel: nothing is stretched however wide the modal grows
  // (a fixed viewBox under preserveAspectRatio="none" stretched the text and
  // dots sideways).  320 until the first measurement, and under jsdom.
  readonly chartWidth = signal(320);
  readonly chartHeight = 170;
  private readonly chartSvg = viewChild<ElementRef<SVGSVGElement>>('chartSvg');
  private resizeObserver: ResizeObserver | null = null;
  // The chart mounts only once the stats land, so observe it when it appears.
  private readonly observeChart = effect(() => {
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
  private readonly padLeft = 40;
  private readonly padRight = 16;
  private readonly padTop = 10;
  private readonly padBottom = 34;

  ngOnInit(): void {
    this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
    this.findApi.getFindStats().subscribe({
      next: (data) => {
        this.stats.set(data);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Failed to load stats'));
        this.loading.set(false);
      },
    });
    this.initDomainOverlap();
    // Best-effort: the section stays hidden on error or when unavailable.
    this.findApi.getEvidenceCoverage().subscribe({
      next: (data) => this.evidence.set(data.available ? data : null),
      error: () => this.evidence.set(null),
    });
  }

  /** Populate the reference-dataset picker from the loaded registry and, if a
   *  single candidate exists, run the domain-shift check right away. A
   *  candidate is any *other* loaded dataset sharing the active dataset's
   *  embedder (a domain check across embedding spaces is meaningless, and the
   *  backend refuses it anyway). */
  private initDomainOverlap(): void {
    const activeId = this.activeCtx.datasetId;
    const datasets = this.datasetState.datasets;
    const activeEmbedder = this.activeDataset.dataset()?.embedder ?? '';
    const candidates = datasets.filter(
      (d) => d.loaded && d.id !== activeId && (d.embedder ?? '') === activeEmbedder,
    );
    this.refCandidates.set(candidates);
    if (candidates.length > 0) {
      this.selectRef(candidates[0].id);
    }
  }

  /** Run the domain-shift report of the active dataset against *refId*'s
   *  coverage atlas, or clear it when the picker is set to "no reference". */
  selectRef(refId: string): void {
    this.selectedRefId.set(refId);
    this.domainShift.set(null);
    this.domainError.set('');
    if (!refId) return;
    this.domainLoading.set(true);
    this.registryApi.domainShift(refId).subscribe({
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

  /** `(change)` handler for the reference `<select>`. */
  onRefChange(event: Event): void {
    this.selectRef((event.target as HTMLSelectElement).value);
  }

  /** Display name of the currently-selected reference dataset. */
  get selectedRefName(): string {
    const id = this.selectedRefId();
    return this.refCandidates().find((d) => d.id === id)?.name ?? id;
  }

  /** Percent of the active dataset that looks atypical under the reference
   *  atlas (`frac_atypical`), rounded for display. */
  get atypicalPct(): number {
    const d = this.domainShift();
    return d ? Math.round(d.frac_atypical * 100) : 0;
  }

  /** Headline verdict class for the overlap chip. */
  get overlapShifted(): boolean {
    return this.domainShift()?.shifted ?? false;
  }

  /** Percent of scored items in an evidence vacuum for their predicted class
   *  (`frac_unsupported`), rounded for display. */
  get unsupportedPct(): number {
    const e = this.evidence();
    return e ? Math.round(e.frac_unsupported * 100) : 0;
  }

  /** Percent of scored items closer to the *other* class's evidence than to
   *  their own predicted class (`frac_low_trust`, trust score < 1). */
  get lowTrustPct(): number {
    const e = this.evidence();
    return e ? Math.round(e.frac_low_trust * 100) : 0;
  }

  /** Headline verdict class for the evidence-coverage chip. */
  get evidenceUnsupported(): boolean {
    return this.evidence()?.unsupported ?? false;
  }

  close(): void {
    this.closed.emit();
  }

  get agreementPct(): string {
    const s = this.stats();
    return s ? `${Math.round(s.agreement_rate * 100)}%` : '-';
  }

  /** The Kept rate: of the matches the user checked, the share kept Good. */
  get keptRatePct(): string {
    const p = this.stats()?.verified_precision;
    return p == null ? '-' : `${Math.round(p * 100)}%`;
  }

  // --- Precision-vs-returned chart ---------------------------------------

  private get plotW(): number {
    return this.chartWidth() - this.padLeft - this.padRight;
  }

  private get plotH(): number {
    return this.chartHeight - this.padTop - this.padBottom;
  }

  /** Right end of the log-scale count axis (at least 10, so one decade shows). */
  private get xMax(): number {
    return Math.max(this.stats()?.n_scored ?? 0, 10);
  }

  xFor(count: number): number {
    return this.padLeft + (Math.log10(Math.max(count, 1)) / Math.log10(this.xMax)) * this.plotW;
  }

  yFor(precision: number): number {
    return this.padTop + (1 - precision) * this.plotH;
  }

  /** Powers of ten up to the corpus size. */
  get xTicks(): XTick[] {
    const ticks: XTick[] = [];
    for (let n = 1; n <= this.xMax; n *= 10) {
      ticks.push({ x: this.xFor(n), label: compactCount(n) });
    }
    return ticks;
  }

  get verifiedPolyline(): string {
    const s = this.stats();
    if (!s) return '';
    return s.precision_curve
      .filter((p) => p.verified_precision != null)
      .map((p) => `${this.xFor(p.n_returned).toFixed(1)},${this.yFor(p.verified_precision as number).toFixed(1)}`)
      .join(' ');
  }

  get hasVerified(): boolean {
    return this.stats()?.precision_curve.some((p) => p.verified_precision != null) ?? false;
  }

  /** The balance's state as the sort state would hold it; null before the stats arrive. */
  get lineBalance() {
    return lineBalanceFrom(this.stats()?.balance);
  }

  /** The check's likely share of the set the line keeps that is right; null while unchecked. */
  get lineRange(): LikelyRange | null {
    return this.lineBalance?.precision ?? null;
  }

  /**
   * The range's tooltip, on the bar and its legend entry. A stale range is
   * drawn exactly as a current one: this is the only place it differs.
   */
  get rangeTooltip(): string {
    const r = this.lineRange;
    return r ? rangeTitle(r) : '';
  }

  /** The range's legend entry: "Likely 11–73% right (checked 5)". */
  get rangeLegend(): string {
    const r = this.lineRange;
    return r ? `Likely ${rangePercent(r)} right (checked ${r.labelled})` : '';
  }

  /** The line's legend entry: whether a check ended on the set it keeps, or never ran. */
  get lineLegend(): string {
    const balance = this.lineBalance;
    if (!balance) return 'Line';
    const kept = balance.count.toLocaleString();
    if (balance.status === 'checked') return `Line: checked (${kept} kept)`;
    return `Line: the top ${kept}, unchecked`;
  }

  /**
   * The note under a checked line: what the picks found on the set it keeps,
   * the share right as a number and the share of all the matches found in
   * words. Empty while unchecked or without ranges.
   */
  get checkedNote(): string {
    const b = this.lineBalance;
    const p = b?.precision;
    const r = b?.recall;
    if (!b || b.status !== 'checked' || !p || !r) return '';
    const kept = b.count.toLocaleString();
    return `Checked: ${p.labelled} random picks found the ${kept} the line keeps likely ${rangePercent(p)} right, with ${foundWords(r)}.`;
  }

  /** X position of the line's marker, or null when nothing clears it. */
  get cutX(): number | null {
    const s = this.stats();
    return s && s.n_returned > 0 ? this.xFor(s.n_returned) : null;
  }

  /** The curve's point at the current cut (the backend always samples it). */
  get cutPoint(): FindStatsPrecisionPoint | null {
    const s = this.stats();
    if (!s || s.n_returned <= 0) return null;
    return s.precision_curve.find((p) => p.n_returned === s.n_returned) ?? null;
  }

  get hoverPoint(): FindStatsPrecisionPoint | null {
    const i = this.hoverIndex();
    const s = this.stats();
    return i == null || !s ? null : (s.precision_curve[i] ?? null);
  }

  /** Snap the hover readout to the sampled count nearest the pointer (in log space). */
  onChartMove(event: MouseEvent): void {
    const s = this.stats();
    const svg = event.currentTarget as SVGSVGElement | null;
    if (!s || !svg || s.precision_curve.length === 0) return;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0) return;
    const x = ((event.clientX - rect.left) / rect.width) * this.chartWidth();
    let best = 0;
    let bestDist = Infinity;
    s.precision_curve.forEach((p, i) => {
      const d = Math.abs(this.xFor(p.n_returned) - x);
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    });
    this.hoverIndex.set(best);
  }

  onChartLeave(): void {
    this.hoverIndex.set(null);
  }

  pct(p: number | null | undefined): string {
    return p == null ? '-' : `${Math.round(p * 100)}%`;
  }

  /** The readout under the chart for one point: the checked precision there, and the checked count. */
  readout(p: FindStatsPrecisionPoint): { main: string; detail: string } {
    return {
      main: p.checked > 0 ? `checked ${this.pct(p.verified_precision)}` : 'nothing checked',
      detail: p.checked > 0 ? `(${p.checked_good.toLocaleString()} of ${p.checked.toLocaleString()} Good)` : '',
    };
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
}
