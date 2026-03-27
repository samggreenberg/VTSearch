import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ExporterInfo } from '../../../models/api.models';

@Component({
  selector: 'vt-detector-export-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './detector-export-modal.component.html',
  styleUrl: './detector-export-modal.component.scss',
})
export class DetectorExportModalComponent implements OnInit, OnDestroy {
  @Input() detectorName = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  labelExporters: ExporterInfo[] = [];
  error = '';
  status = '';

  private destroy$ = new Subject<void>();

  constructor(
    private detectorsApi: DetectorsApiService,
    private exportersApi: ExportersApiService,
    private sortingApi: SortingApiService,
  ) {}

  get title(): string {
    return this.detectorName ? `Export "${this.detectorName}"` : 'Export Detector';
  }

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntil(this.destroy$)).subscribe({
      next: (list) => {
        this.labelExporters = list;
      },
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  exportBrowser(): void {
    this.status = 'Exporting...';
    this.detectorsApi.exportDetector(this.detectorName).subscribe({
      next: (data: any) => {
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.detectorName || 'detector'}.json`;
        a.click();
        URL.revokeObjectURL(url);
        this.status = 'Downloaded.';
        this.exported.emit();
      },
      error: () => {
        this.error = 'Export failed';
        this.status = '';
      },
    });
  }

  exportServer(): void {
    this.status = 'Saving to server...';
    this.detectorsApi.exportDetectorToServer(this.detectorName).subscribe({
      next: () => {
        this.status = 'Saved to server.';
        this.exported.emit();
      },
      error: () => {
        this.error = 'Failed to save to server';
        this.status = '';
      },
    });
  }

  exportLabels(exporter: ExporterInfo): void {
    this.sortingApi.exportLabels().subscribe({
      next: (labelsData) => {
        this.exportersApi
          .runExport({
            exporter_name: exporter.name,
            results: labelsData,
          })
          .subscribe({
            next: () => {
              this.status = 'Labels exported.';
              this.exported.emit();
            },
            error: () => {
              this.error = 'Label export failed';
            },
          });
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
