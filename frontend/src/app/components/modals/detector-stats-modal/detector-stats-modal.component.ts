import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsRegistryApiService } from '../../../services/detectors-registry-api.service';
import type { DetectorRegistryStatsResponse } from '../../../generated/api-client/models/detector-registry-stats-response';
import { formatTimestamp as formatTs } from '../../../utils/format-date';

/** Read-only stats for a registered detector. Mirrors the dataset stats
 *  modal: labelset composition (positives / negatives / total, plus how
 *  many positives currently resolve into the active dataset) and the
 *  detector's creation/provenance metadata. Counts only — no embeddings
 *  or MLP weights are read (see the "No Persisted Vectors" rule). */
@Component({
  selector: 'vt-detector-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './detector-stats-modal.component.html',
  styleUrl: './detector-stats-modal.component.scss',
})
export class DetectorStatsModalComponent implements OnInit {
  @Input() detectorId = '';
  @Input() detectorName = '';
  @Output() closed = new EventEmitter<void>();

  loading = true;
  error = '';
  stats: DetectorRegistryStatsResponse | null = null;

  constructor(private detectorsRegistryApi: DetectorsRegistryApiService) {}

  ngOnInit(): void {
    this.detectorsRegistryApi.getDetectorStats(this.detectorId).subscribe({
      next: (data) => {
        this.stats = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.error || err.error?.message || 'Failed to load stats';
        this.loading = false;
      },
    });
  }

  close(): void {
    this.closed.emit();
  }

  formatTimestamp(ts: number | null): string {
    return formatTs(ts);
  }

  /** "N of M (in \"dataset\")" for the resolved-positives row, or a hint
   *  when no dataset is loaded to resolve against. */
  get resolvedSummary(): string {
    const s = this.stats;
    if (!s) return '-';
    if (!s.active_dataset_name) return 'No dataset loaded';
    return `${s.num_positive_resolved} of ${s.num_positive} in "${s.active_dataset_name}"`;
  }
}
