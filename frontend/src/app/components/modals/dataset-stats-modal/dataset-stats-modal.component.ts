import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { DatasetStatsResponse } from '../../../models/api.models';
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
  stats: DatasetStatsResponse | null = null;

  constructor(private datasetsApi: DatasetsApiService) {}

  ngOnInit(): void {
    this.datasetsApi.getDatasetStats(this.datasetId).subscribe({
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

  get originParams(): { key: string; value: string }[] {
    const src = this.stats?.source as { params?: Record<string, unknown> } | undefined;
    const params = src?.params;
    if (!params) return [];
    return Object.entries(params)
      .filter(([, v]) => v !== '' && v != null)
      .map(([key, value]) => ({ key, value: String(value) }));
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
