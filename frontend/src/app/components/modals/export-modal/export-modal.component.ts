import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ExporterInfo } from '../../../models/api.models';

@Component({
  selector: 'vt-export-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit {
  @Input() detectorName = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  exporters: ExporterInfo[] = [];
  loading = true;
  error = '';
  status = '';

  constructor(
    private detectorsApi: DetectorsApiService,
    private exportersApi: ExportersApiService,
    private sortingApi: SortingApiService,
  ) {}

  ngOnInit(): void {
    this.exportersApi.getExporters().subscribe({
      next: (list) => {
        this.exporters = list;
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load exporters';
      },
    });
  }

  exportLabelsWith(exporter: ExporterInfo): void {
    this.status = 'Exporting labels...';
    this.error = '';
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
              this.status = '';
              this.error = 'Label export failed';
            },
          });
      },
      error: () => {
        this.status = '';
        this.error = 'Failed to fetch labels';
      },
    });
  }

  exportDetectorBrowser(): void {
    this.status = 'Exporting...';
    this.error = '';
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

  exportDetectorServer(): void {
    this.status = 'Saving to server...';
    this.error = '';
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

  close(): void {
    this.closed.emit();
  }
}
