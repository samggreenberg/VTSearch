import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';

@Component({
  selector: 'vt-label-exporter-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, IconComponent],
  templateUrl: './label-exporter-modal.component.html',
  styleUrl: './label-exporter-modal.component.scss',
})
export class LabelExporterModalComponent implements OnInit, OnDestroy {
  @Input() goodsOnly = false;
  @Input() customTitle = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exportComplete = new EventEmitter<void>();

  exporters: ExporterEntry[] = [];
  loading = true;
  error = '';

  private destroy$ = new Subject<void>();

  constructor(
    private exportersApi: ExportersApiService,
    private sortingApi: SortingApiService,
  ) {}

  get title(): string {
    if (this.customTitle) return this.customTitle;
    return this.goodsOnly ? 'Export Labels (Goods)' : 'Export Labels';
  }

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntil(this.destroy$)).subscribe({
      next: (list) => {
        this.exporters = list.filter((exp) => !exp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load exporters';
      },
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
              this.error = 'Export failed';
            },
          });
      },
      error: () => {
        this.error = 'Failed to fetch labels';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
