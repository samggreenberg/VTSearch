import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import type { DatasetRegistryStatsResponse } from '../../../generated/api-client/models/dataset-registry-stats-response';
import { formatTimestamp as formatTs } from '../../../utils/format-date';

@Component({
  selector: 'vt-dataset-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './dataset-stats-modal.component.html',
  styleUrl: './dataset-stats-modal.component.scss',
})
export class DatasetStatsModalComponent implements OnInit {
  @Input() datasetId = '';
  @Input() datasetName = '';
  @Output() closed = new EventEmitter<void>();

  loading = true;
  error = '';
  stats: DatasetRegistryStatsResponse | null = null;

  constructor(private datasetsRegistryApi: DatasetsRegistryApiService) {}

  ngOnInit(): void {
    this.datasetsRegistryApi.getDatasetStats(this.datasetId).subscribe({
      next: (data) => {
        this.stats = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Failed to load stats';
        this.loading = false;
      },
    });
  }

  close(): void {
    this.closed.emit();
  }

  get fileTypes(): { ext: string; count: number }[] {
    if (!this.stats?.file_type_counts) return [];
    return Object.entries(this.stats.file_type_counts)
      .sort(([, a], [, b]) => b - a)
      .map(([ext, count]) => ({ ext, count }));
  }

  get importerName(): string {
    const src = this.stats?.source as { importer?: string } | undefined;
    return src?.importer || '';
  }

  get originParams(): { key: string; label: string; value: string }[] {
    const src = this.stats?.source as { params?: Record<string, unknown> } | undefined;
    const params = src?.params;
    if (!params) return [];
    return Object.entries(params)
      .filter(([, v]) => v !== '' && v != null)
      .map(([key, value]) => ({ key, label: this.formatParamKey(key), value: String(value) }));
  }

  /** Render a raw origin-param key (e.g. ``media_type``, ``name``) as a label
   *  that matches the hardcoded stat labels: underscores become spaces and the
   *  first letter is capitalized, so a demo dataset's ``name`` param reads
   *  "Name" alongside "Importer" / "Clipper" / "Embedder" instead of lowercase. */
  private formatParamKey(key: string): string {
    const spaced = key.replace(/_/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }

  formatTimestamp(ts: number | null): string {
    return formatTs(ts);
  }

  get duration(): string {
    if (!this.stats?.ingest_started_at || !this.stats?.ingest_finished_at) return '-';
    const seconds = Math.round(this.stats.ingest_finished_at - this.stats.ingest_started_at);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes < 60) return `${minutes}m ${secs}s`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m ${secs}s`;
  }
}
