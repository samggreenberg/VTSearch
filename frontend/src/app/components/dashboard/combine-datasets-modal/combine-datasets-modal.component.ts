import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { DatasetRegistryEntry, MediaTypeInfo } from '../../../models/api.models';

interface CombineRow {
  id: string;
  name: string;
  media_type: string;
  num_items: number;
  pkl_path: string;
}

@Component({
  selector: 'vt-combine-datasets-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent],
  templateUrl: './combine-datasets-modal.component.html',
  styleUrl: './combine-datasets-modal.component.scss',
})
export class CombineDatasetsModalComponent implements OnInit {
  /** Datasets pre-selected on the dashboard when the modal was opened. */
  @Input() datasets: DatasetRegistryEntry[] = [];

  @Output() closed = new EventEmitter<void>();
  @Output() combineStarted = new EventEmitter<void>();

  rows: CombineRow[] = [];
  mediaTypes: MediaTypeInfo[] = [];
  submitting = false;
  error = '';
  name = '';

  constructor(private datasetsApi: DatasetsApiService) {}

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

    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes = res.media_types || [];
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
    const mt = this.mediaTypes.find((m) => m.type_id === typeId);
    return mt?.name || typeId;
  }

  mediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  removeRow(id: string): void {
    this.rows = this.rows.filter((r) => r.id !== id);
  }

  submit(): void {
    if (!this.canCombine) return;
    this.submitting = true;
    this.error = '';
    const paths = this.rows.map((r) => r.pkl_path);
    const name = (this.name || '').trim() || this.defaultName();
    this.datasetsApi.combineDatasets({ datasets: paths, name }).subscribe({
      next: () => {
        this.submitting = false;
        this.combineStarted.emit();
      },
      error: (err) => {
        this.submitting = false;
        this.error = (err?.error?.error as string) || 'Combine failed';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
