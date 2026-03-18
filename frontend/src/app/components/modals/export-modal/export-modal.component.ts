import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ExporterInfo, LabelEntry } from '../../../models/api.models';

export type ExportColumn = 'label' | 'md5' | 'origin_name' | 'filename' | 'category';

export interface ColumnDef {
  key: ExportColumn;
  label: string;
  enabled: boolean;
}

@Component({
  selector: 'vt-export-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit, OnDestroy {
  @Input() detectorName = '';
  /** 'label' = Labeling mode (detector export allowed), 'find' = Finding mode (no detector export). */
  @Input() mode: 'label' | 'find' = 'label';
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  exporters: ExporterInfo[] = [];
  loading = true;
  error = '';
  status = '';

  /** Labels fetched from the server. */
  labels: LabelEntry[] = [];
  labelsLoaded = false;

  /** Column definitions with selection state. */
  columns: ColumnDef[] = [
    { key: 'label', label: 'Label', enabled: true },
    { key: 'md5', label: 'MD5', enabled: true },
    { key: 'origin_name', label: 'Origin Name', enabled: true },
    { key: 'filename', label: 'Filename', enabled: true },
    { key: 'category', label: 'Category', enabled: true },
  ];

  /** Delimiter for text export. */
  delimiter = ',';
  delimiterOptions = [
    { value: ',', label: 'Comma (,)' },
    { value: '\t', label: 'Tab' },
    { value: '|', label: 'Pipe (|)' },
    { value: ';', label: 'Semicolon (;)' },
  ];

  /** Exporter form state. */
  showExporterForm = false;
  selectedExporter: ExporterInfo | null = null;
  formValues: Record<string, string> = {};
  submitting = false;

  /** Copy feedback. */
  copySuccess = false;

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

    this.sortingApi.exportLabels().subscribe({
      next: (data) => {
        this.labels = data.labels || [];
        this.labelsLoaded = true;
      },
      error: () => {
        this.labelsLoaded = true;
        this.error = 'Failed to load labels';
      },
    });
  }

  get enabledColumns(): ColumnDef[] {
    return this.columns.filter((c) => c.enabled);
  }

  get previewLabels(): LabelEntry[] {
    return this.labels.slice(0, 50);
  }

  get hasLabels(): boolean {
    return this.labels.length > 0;
  }

  get showDetectorSection(): boolean {
    return this.mode === 'label';
  }

  get modalTitle(): string {
    if (this.showExporterForm && this.selectedExporter) {
      return this.selectedExporter.display_name || this.selectedExporter.name;
    }
    return 'Export';
  }

  getCellValue(entry: LabelEntry, col: ExportColumn): string {
    return String((entry as unknown as Record<string, unknown>)[col] ?? '');
  }

  /** Build delimited text from labels using selected columns. */
  buildExportText(): string {
    const cols = this.enabledColumns;
    if (cols.length === 0) return '';
    const header = cols.map((c) => c.label).join(this.delimiter);
    const rows = this.labels.map((entry) =>
      cols.map((c) => this.getCellValue(entry, c.key)).join(this.delimiter),
    );
    return [header, ...rows].join('\n');
  }

  /** Copy delimited text to clipboard. */
  copyToClipboard(): void {
    const text = this.buildExportText();
    navigator.clipboard.writeText(text).then(
      () => {
        this.copySuccess = true;
        this.status = `Copied ${this.labels.length} rows to clipboard.`;
        setTimeout(() => (this.copySuccess = false), 2000);
      },
      () => {
        this.error = 'Failed to copy to clipboard';
      },
    );
  }

  /** Start exporter flow — if no fields, export immediately. */
  startExporter(exporter: ExporterInfo): void {
    const fields = exporter.fields || [];
    if (fields.length === 0) {
      this.exportLabelsWith(exporter, {});
      return;
    }
    this.selectedExporter = exporter;
    this.formValues = {};
    for (const f of fields) {
      this.formValues[f.key] = f.default || '';
    }
    this.showExporterForm = true;
    this.error = '';
    this.status = '';
  }

  backFromForm(): void {
    this.showExporterForm = false;
    this.selectedExporter = null;
    this.error = '';
    this.status = '';
  }

  submitForm(): void {
    if (!this.selectedExporter) return;
    this.exportLabelsWith(this.selectedExporter, { ...this.formValues });
  }

  exportLabelsWith(exporter: ExporterInfo, fieldValues: Record<string, string>): void {
    this.status = 'Exporting...';
    this.error = '';
    this.submitting = true;

    // Build a results dict that includes selected columns info
    const labelsData = { labels: this.labels };
    this.exportersApi
      .runExport({
        exporter_name: exporter.name,
        field_values: fieldValues,
        results: labelsData,
      })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.status = 'Labels exported.';
          this.submitting = false;
          this.showExporterForm = false;
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

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
