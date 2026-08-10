import { ChangeDetectionStrategy, Component, inject, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsFindApiService } from '../../../services/detectors-find-api.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ActiveContextService } from '../../../services/active-context.service';
import type { FindStatsResponse } from '../../../generated/api-client/models/find-stats-response';
import type { FindEvidenceCoverageResponse } from '../../../generated/api-client/models/find-evidence-coverage-response';
import type { DatasetDomainShiftResponse } from '../../../generated/api-client/models/dataset-domain-shift-response';
import type { DatasetRegistryEntry } from '../../../models/api.models';
import { apiErrorMessage } from '../../../utils/api-error';

/** A scaled point on the FP/FN-vs-inclusion chart. */
interface ChartPoint {
  inclusion: number;
  x: number;
  yFp: number;
  yFn: number;
}

/**
 * Detector-evaluation Stats for a Find run: the 2×2 confusion of the adopted
 * label set against the detector's original call, the derived agreement /
 * precision rates, and the headline FP/FN-vs-inclusion sweep rendered as a
 * dependency-free inline SVG line chart (current inclusion marked).
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

  // Chart geometry (SVG user units; the viewBox scales to the container width).
  readonly chartWidth = 320;
  readonly chartHeight = 150;
  private readonly padLeft = 34;
  private readonly padRight = 10;
  private readonly padTop = 10;
  private readonly padBottom = 22;

  ngOnInit(): void {
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
    const activeEmbedder = datasets.find((d) => d.id === activeId)?.embedder ?? '';
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

  get precisionPct(): string {
    const s = this.stats();
    return s ? `${Math.round(s.precision * 100)}%` : '-';
  }

  // --- FP/FN-vs-inclusion chart -------------------------------------------

  private get plotW(): number {
    return this.chartWidth - this.padLeft - this.padRight;
  }

  private get plotH(): number {
    return this.chartHeight - this.padTop - this.padBottom;
  }

  /** Largest FP/FN count in the sweep; the chart's y-axis top (min 1). */
  get maxCount(): number {
    const s = this.stats();
    if (!s) return 1;
    let m = 1;
    for (const p of s.sweep) {
      m = Math.max(m, p.false_pos, p.false_neg);
    }
    return m;
  }

  private xFor(inclusion: number): number {
    return this.padLeft + ((inclusion + 10) / 20) * this.plotW;
  }

  private yFor(count: number): number {
    return this.padTop + (1 - count / this.maxCount) * this.plotH;
  }

  get points(): ChartPoint[] {
    const s = this.stats();
    if (!s) return [];
    return s.sweep.map((p) => ({
      inclusion: p.inclusion,
      x: this.xFor(p.inclusion),
      yFp: this.yFor(p.false_pos),
      yFn: this.yFor(p.false_neg),
    }));
  }

  get fpPolyline(): string {
    return this.points.map((p) => `${p.x.toFixed(1)},${p.yFp.toFixed(1)}`).join(' ');
  }

  get fnPolyline(): string {
    return this.points.map((p) => `${p.x.toFixed(1)},${p.yFn.toFixed(1)}`).join(' ');
  }

  /** X position of the current-inclusion marker line. */
  get currentX(): number {
    const s = this.stats();
    return s ? this.xFor(s.inclusion) : 0;
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
    return this.chartWidth - this.padRight;
  }
}
