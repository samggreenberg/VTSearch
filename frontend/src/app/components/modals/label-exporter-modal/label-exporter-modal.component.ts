import { Component, EventEmitter, Input, Output, computed, inject, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';

@Component({
  selector: 'vt-label-exporter-modal',
  standalone: true,
  imports: [ModalComponent, IconComponent],
  templateUrl: './label-exporter-modal.component.html',
  styleUrl: './label-exporter-modal.component.scss',
})
export class LabelExporterModalComponent {
  @Input() goodsOnly = false;
  @Input() customTitle = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exportComplete = new EventEmitter<void>();

  private readonly exportersApi = inject(ExportersApiService);
  private readonly sortingApi = inject(SortingApiService);

  // Eager `rxResource`: loads the exporter list once on creation (no request
  // signal), wrapping the generated-client read so the interceptor chain still
  // applies. Replaces the old `ngOnInit` subscribe + `destroy$` plumbing.
  private readonly exportersResource = rxResource({
    stream: () => this.exportersApi.getExporters(),
  });
  readonly exporters = computed<ExporterEntry[]>(() =>
    (this.exportersResource.value() ?? []).filter((exp) => !exp.hidden_from_picker),
  );
  readonly loading = computed(() => this.exportersResource.isLoading());

  /** Error from a failed export action; the list-load failure is merged in. */
  private readonly exportError = signal('');
  readonly error = computed(
    () =>
      this.exportError() || (this.exportersResource.error() ? 'Failed to load exporters' : ''),
  );

  get title(): string {
    if (this.customTitle) return this.customTitle;
    return this.goodsOnly ? 'Export Labels (Goods)' : 'Export Labels';
  }

  selectExporter(exporter: ExporterEntry): void {
    this.sortingApi.exportLabels(this.goodsOnly).subscribe({
      next: (labelsData) => {
        this.exportersApi
          .runExport({
            exporter_name: exporter.name,
            results: labelsData,
          })
          .subscribe({
            next: () => {
              this.exportComplete.emit();
              this.closed.emit();
            },
            error: () => {
              this.exportError.set('Export failed');
            },
          });
      },
      error: () => {
        this.exportError.set('Failed to fetch labels');
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
