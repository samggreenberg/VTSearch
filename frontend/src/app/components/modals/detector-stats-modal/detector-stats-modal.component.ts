import { ChangeDetectionStrategy, Component, inject, input, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsRegistryApiService } from '../../../services/detectors-registry-api.service';
import type { DetectorRegistryStatsResponse } from '../../../generated/api-client/models/detector-registry-stats-response';
import { formatTimestamp as formatTs } from '../../../utils/format-date';
import { apiErrorMessage } from '../../../utils/api-error';
import { verdictLine, verdictTitle } from '../../../utils/line-test';
import { isSoloMediaType } from '../../../utils/solo-media-type';
import { NO_VALUE } from '../../../utils/no-value';
import type { OriginRow } from '../../../generated/api-client/models/origin-row';

/** Read-only stats for a registered detector. Mirrors the dataset stats
 *  modal: labelset composition (positives / negatives / total, plus how
 *  many positives currently resolve into the active dataset), the test
 *  verdicts the detector keeps (*Tested on*, #4526), how a label importer
 *  created it when one did (*Import*, #5334), and the detector's
 *  creation/provenance metadata. Counts only — no embeddings or MLP
 *  weights are read (see the "No Persisted Vectors" rule). */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-detector-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './detector-stats-modal.component.html',
  styleUrl: './detector-stats-modal.component.scss',
})
export class DetectorStatsModalComponent implements OnInit {
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);

  readonly detectorId = input('');
  readonly detectorName = input('');
  /** The server's one media type when it is locked to one, from the
   *  Dashboard; see {@link showMediaType}. */
  readonly soloMediaType = input<string | null>(null);
  readonly closed = output<void>();

  /** What a row with no value shows. */
  readonly noValue = NO_VALUE;

  // Signalized so the `ngOnInit` subscribe (an unpatched callback under zoneless)
  // schedules CD when the stats land.
  readonly loading = signal(true);
  readonly error = signal('');
  readonly stats = signal<DetectorRegistryStatsResponse | null>(null);

  protected readonly verdictLine = verdictLine;
  protected readonly verdictTitle = verdictTitle;

  ngOnInit(): void {
    this.detectorsRegistryApi.getDetectorStats(this.detectorId()).subscribe({
      next: (data) => {
        this.stats.set(data);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Failed to load stats'));
        this.loading.set(false);
      },
    });
  }

  /** Whether the Media type row shows: not on a server locked to the
   *  detector's type, where it could be nothing else (#5106). */
  get showMediaType(): boolean {
    return !isSoloMediaType(this.soloMediaType(), this.stats()?.media_type);
  }

  /** The label importer's display name, as the New Detector › Trained picker names it. */
  get importerName(): string {
    const s = this.stats();
    return s?.importer_display_name || s?.imported?.importer || '';
  }

  /** The fields the detector's label importer ran with, labelled and worded
   *  by the server (#5620), the way the dataset Stats window's rows are. */
  get importParams(): OriginRow[] {
    return this.stats()?.imported_rows ?? [];
  }

  /** The row's tooltip: the importer field's description, else its raw key. */
  paramTitle(row: OriginRow): string {
    return row.description || `Importer field: ${row.key}`;
  }

  close(): void {
    this.closed.emit();
  }

  formatTimestamp(ts: number | null): string {
    return ts ? formatTs(ts) : NO_VALUE;
  }

  /** When the detector last trained, or "Never" for one that has not. */
  get lastTrained(): string {
    const ts = this.stats()?.last_trained_at;
    return ts ? formatTs(ts) : 'Never';
  }

  /** "N of M (in \"dataset\")" for the resolved-positives row, or a hint
   *  when no dataset is loaded to resolve against. */
  get resolvedSummary(): string {
    const s = this.stats();
    if (!s) return NO_VALUE;
    if (!s.active_dataset_name) return 'No dataset loaded';
    return `${s.num_positive_resolved} of ${s.num_positive} in "${s.active_dataset_name}"`;
  }
}
