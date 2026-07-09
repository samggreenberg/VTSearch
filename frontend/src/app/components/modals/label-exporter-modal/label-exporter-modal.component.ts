import { ChangeDetectionStrategy, Component, computed, inject, Input, input, output, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ToastService } from '../../../services/toast.service';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-label-exporter-modal',
  standalone: true,
  imports: [ModalComponent, IconComponent],
  templateUrl: './label-exporter-modal.component.html',
  styleUrl: './label-exporter-modal.component.scss',
})
export class LabelExporterModalComponent {
  @Input() goodsOnly = false;
  readonly customTitle = input('');
  readonly closed = output<void>();
  readonly exportComplete = output<void>();

  private readonly exportersApi = inject(ExportersApiService);
  private readonly sortingApi = inject(SortingApiService);
  private readonly toast = inject(ToastService);

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
    const customTitle = this.customTitle();
    if (customTitle) return customTitle;
    return this.goodsOnly ? 'Export Labels (Goods)' : 'Export Labels';
  }

  selectExporter(exporter: ExporterEntry): void {
    const exporterLabel = exporter.display_name || exporter.name;
    this.sortingApi.exportLabels(this.goodsOnly).subscribe({
      next: (labelsData) => {
        const labelCount = labelsData.labels?.length ?? 0;
        this.exportersApi
          .runExport({
            exporter_name: exporter.name,
            results: labelsData,
          })
          .subscribe({
            next: () => {
              // Selecting an exporter closes the modal immediately, so a toast
              // is the only durable confirmation the export succeeded (#2217).
              this.toast.success({
                message: `Exported ${labelCount.toLocaleString()} label${labelCount === 1 ? '' : 's'} to ${exporterLabel}`,
                dedupKey: 'label-export-success',
              });
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
