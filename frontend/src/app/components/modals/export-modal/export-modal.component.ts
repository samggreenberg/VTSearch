import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ExporterInfo } from '../../../models/api.models';

@Component({
  selector: 'vt-export-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit, OnDestroy {
  @Input() detectorName = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  exporters: ExporterInfo[] = [];
  loading = true;
  error = '';
  status = '';

  /** Current view: picker list or field form for a selected exporter. */
  view: 'picker' | 'form' = 'picker';
  selectedExporter: ExporterInfo | null = null;
  formValues: Record<string, string> = {};
  submitting = false;

  private destroy$ = new Subject<void>();

  constructor(
    private detectorsApi: DetectorsApiService,
    private exportersApi: ExportersApiService,
    private sortingApi: SortingApiService,
  ) {}

  ngOnInit(): void {
    this.exportersApi.getExporters().subscribe({
      next: (list) => {
        this.exporters = list.filter((e) => !e.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load exporters';
      },
    });
  }

  get modalTitle(): string {
    if (this.view === 'form' && this.selectedExporter) {
      return this.selectedExporter.display_name || this.selectedExporter.name;
    }
    return 'Export';
  }

  selectExporter(exporter: ExporterInfo): void {
    const fields = exporter.fields || [];
    if (fields.length === 0) {
      // No fields needed — export immediately
      this.exportLabelsWith(exporter, {});
      return;
    }
    // Show form for this exporter
    this.selectedExporter = exporter;
    this.formValues = {};
    for (const f of fields) {
      this.formValues[f.key] = f.default || '';
    }
    this.view = 'form';
    this.error = '';
    this.status = '';
  }

  back(): void {
    this.view = 'picker';
    this.selectedExporter = null;
    this.error = '';
    this.status = '';
  }

  submitForm(): void {
    if (!this.selectedExporter) return;
    this.exportLabelsWith(this.selectedExporter, { ...this.formValues });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  exportLabelsWith(exporter: ExporterInfo, fieldValues: Record<string, string>): void {
    this.status = 'Exporting labels...';
    this.error = '';
    this.submitting = true;
    this.sortingApi
      .exportLabels()
      .pipe(
        takeUntil(this.destroy$),
        switchMap((labelsData) =>
          this.exportersApi.runExport({
            exporter_name: exporter.name,
            field_values: fieldValues,
            results: labelsData,
          }),
        ),
      )
      .subscribe({
        next: () => {
          this.status = 'Labels exported.';
          this.submitting = false;
          this.exported.emit();
        },
        error: () => {
          this.status = '';
          this.error = 'Label export failed';
          this.submitting = false;
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
