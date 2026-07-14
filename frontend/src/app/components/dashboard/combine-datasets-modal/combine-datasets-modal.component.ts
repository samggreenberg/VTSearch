import { ChangeDetectionStrategy, Component, inject, Input, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetRegistryEntry, MediaTypeInfo } from '../../../models/api.models';
import { apiErrorMessage } from '../../../utils/api-error';

interface CombineRow {
  id: string;
  name: string;
  media_type: string;
  num_items: number;
  pkl_path: string;
}

/**
 * Payload emitted when a combine kicks off. Carries the pre-dedup source
 * counts alongside the task id so the dashboard can compute a post-combine
 * summary toast ("N unique kept, M duplicates dropped") once the background
 * task settles — the unique count is only knowable from the resulting
 * dataset, but the duplicate count is `totalItems - uniqueKept`.
 */
export interface CombineStartedInfo {
  taskId: string;
  numSources: number;
  totalItems: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-combine-datasets-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent],
  templateUrl: './combine-datasets-modal.component.html',
  styleUrl: './combine-datasets-modal.component.scss',
})
export class CombineDatasetsModalComponent implements OnInit {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);

  /** Datasets pre-selected on the dashboard when the modal was opened. */
  @Input() datasets: DatasetRegistryEntry[] = [];

  readonly closed = output<void>();
  readonly combineStarted = output<CombineStartedInfo>();

  rows: CombineRow[] = [];
  // Signals: written from the media-types / combine subscribes (async, not a
  // zoneless CD trigger) yet read in the template, so they must repaint on emit.
  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  readonly submitting = signal(false);
  readonly error = signal('');
  name = '';

  ngOnInit(): void {
    this.rows = this.datasets
      .map((d) => ({
        id: d.id,
        name: d.name,
        media_type: d.media_type,
        num_items: Number(d['num_items'] ?? 0),
        pkl_path: String(d['pkl_path'] ?? ''),
      }))
      .filter((r) => !!r.pkl_path);

    this.name = this.defaultName();

    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes.set(res.media_types || []);
      },
    });
  }

  /** Default combined-dataset name: source names joined with " + ". */
  private defaultName(): string {
    return this.rows.map((r) => r.name).filter((n) => !!n).join(' + ');
  }

  /** Total media items across all selected datasets, before deduplication. */
  get totalItems(): number {
    return this.rows.reduce((sum, r) => sum + (r.num_items || 0), 0);
  }

  get distinctMediaTypes(): string[] {
    return Array.from(new Set(this.rows.map((r) => r.media_type)));
  }

  get sharedMediaType(): string {
    return this.distinctMediaTypes.length === 1 ? this.distinctMediaTypes[0] : '';
  }

  get canCombine(): boolean {
    return this.rows.length >= 2 && this.distinctMediaTypes.length === 1;
  }

  /** Tooltip / inline reason describing why the Combine button is disabled. */
  get disabledReason(): string {
    if (this.rows.length < 2) {
      return 'Need at least two datasets to combine.';
    }
    if (this.distinctMediaTypes.length > 1) {
      return `All datasets must share a media type (got ${this.distinctMediaTypes.join(', ')}).`;
    }
    return '';
  }

  mediaTypeLabel(typeId: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === typeId);
    return mt?.name || typeId;
  }

  mediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  removeRow(id: string): void {
    this.rows = this.rows.filter((r) => r.id !== id);
  }

  submit(): void {
    if (!this.canCombine) return;
    this.submitting.set(true);
    this.error.set('');
    const paths = this.rows.map((r) => r.pkl_path);
    const name = (this.name || '').trim() || this.defaultName();
    const numSources = this.rows.length;
    const totalItems = this.totalItems;
    this.datasetsCrudApi.combineDatasets({ datasets: paths, name }).subscribe({
      next: (res) => {
        this.submitting.set(false);
        this.combineStarted.emit({ taskId: res.task_id, numSources, totalItems });
      },
      error: (err) => {
        this.submitting.set(false);
        this.error.set(apiErrorMessage(err, 'Combine failed'));
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
