import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ExporterInfo, LabelEntry } from '../../../models/api.models';

export interface ColumnDef {
  key: string;
  label: string;
  enabled: boolean;
  isMetadata?: boolean;
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

  /** Column definitions with selection state — built dynamically from API response. */
  columns: ColumnDef[] = [];

  /** Filter which labels to show/export. */
  labelFilter: 'good' | 'bad' | 'both' | 'corrections' = 'both';

  /** Delimiter for text export. */
  delimiter = ',';
  delimiterOptions = [
    { value: ',', label: 'Comma (,)' },
    { value: '\t', label: 'Tab (\u21E5)' },
    { value: '|', label: 'Pipe (|)' },
    { value: ';', label: 'Semicolon (;)' },
  ];

  /** Active export tab — 'clipboard' or an exporter name. */
  activeTab = 'clipboard';

  /** Exporter form state. */
  selectedExporter: ExporterInfo | null = null;
  formValues: Record<string, string> = {};
  submitting = false;

  /** Copy feedback. */
  copySuccess = false;

  private destroy$ = new Subject<void>();

  /** Base columns that are always present. */
  private static readonly BASE_COLUMNS: { key: string; label: string }[] = [
    { key: 'label', label: 'Label' },
    { key: 'md5', label: 'MD5' },
    { key: 'origin_name', label: 'Origin Name' },
    { key: 'filename', label: 'Filename' },
    { key: 'category', label: 'Category' },
  ];

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

    this.sortingApi.exportLabels(false, { enrich: true }).subscribe({
      next: (data) => {
        this.labels = data.labels || [];
        this.labelsLoaded = true;
        this.buildColumns(data.available_columns);
      },
      error: () => {
        this.labelsLoaded = true;
        this.error = 'Failed to load labels';
        this.buildColumns();
      },
    });
  }

  /** Build column definitions from available_columns or fall back to defaults. */
  private buildColumns(availableColumns?: string[]): void {
    const baseKeys = new Set(ExportModalComponent.BASE_COLUMNS.map((c) => c.key));
    // Start with base columns
    this.columns = ExportModalComponent.BASE_COLUMNS.map((c) => ({
      key: c.key,
      label: c.label,
      enabled: true,
    }));
    // Add metadata columns discovered from the data
    if (availableColumns) {
      for (const key of availableColumns) {
        if (!baseKeys.has(key)) {
          this.columns.push({
            key,
            label: key,
            enabled: true,
            isMetadata: true,
          });
        }
      }
    }
  }

  get enabledColumns(): ColumnDef[] {
    return this.columns.filter((c) => c.enabled);
  }

  get filteredLabels(): LabelEntry[] {
    if (this.labelFilter === 'good') {
      return this.labels.filter((e) => e.label === 'good');
    }
    if (this.labelFilter === 'bad') {
      return this.labels.filter((e) => e.label === 'bad');
    }
    if (this.labelFilter === 'corrections') {
      return this.labels.filter((e) => e.is_correction === true);
    }
    return this.labels;
  }

  /** Whether any labels are corrections (detector label was changed by user). */
  get hasCorrections(): boolean {
    return this.labels.some((e) => e.is_correction === true);
  }

  get previewLabels(): LabelEntry[] {
    return this.filteredLabels.slice(0, 50);
  }

  get hasLabels(): boolean {
    return this.filteredLabels.length > 0;
  }

  get showDetectorSection(): boolean {
    return this.mode === 'label';
  }

  get hasExporterForm(): boolean {
    return this.selectedExporter !== null;
  }

  getCellValue(entry: LabelEntry, col: ColumnDef): string {
    if (col.isMetadata) {
      const meta = entry.custom_metadata;
      if (meta && col.key in meta) {
        return String(meta[col.key] ?? '');
      }
      return '';
    }
    return String((entry as unknown as Record<string, unknown>)[col.key] ?? '');
  }

  /** Build delimited text from labels using selected columns. */
  buildExportText(): string {
    const cols = this.enabledColumns;
    if (cols.length === 0) return '';
    const header = cols.map((c) => c.label).join(this.delimiter);
    const rows = this.filteredLabels.map((entry) =>
      cols.map((c) => this.getCellValue(entry, c)).join(this.delimiter),
    );
    return [header, ...rows].join('\n');
  }

  /** Copy delimited text to clipboard. */
  copyToClipboard(): void {
    const text = this.buildExportText();
    navigator.clipboard.writeText(text).then(
      () => {
        this.copySuccess = true;
        this.status = `Copied ${this.filteredLabels.length} rows to clipboard.`;
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
    this.error = '';
    this.status = '';
  }

  /** Select an exporter tab and initialise its form values. */
  selectExporterTab(exporter: ExporterInfo): void {
    this.activeTab = exporter.name;
    this.selectedExporter = exporter;
    this.formValues = {};
    for (const f of exporter.fields || []) {
      this.formValues[f.key] = f.default || '';
    }
    this.error = '';
    this.status = '';
  }

  /** The exporter object for the currently active tab (null if clipboard). */
  get activeTabExporter(): ExporterInfo | null {
    if (this.activeTab === 'clipboard') return null;
    return this.exporters.find((e) => e.name === this.activeTab) || null;
  }

  /** Label for the action button on the active exporter tab. */
  get activeTabAction(): string {
    const exp = this.activeTabExporter;
    if (!exp) return 'Export';
    const name = (exp.display_name || exp.name).toLowerCase();
    if (name.includes('email') || name.includes('smtp')) return 'Send';
    if (name.includes('csv') || name.includes('file') || name.includes('json')) return 'Save';
    if (name.includes('webhook')) return 'Send';
    return 'Export';
  }

  /** Submit the currently active exporter tab. */
  submitExporterTab(): void {
    const exp = this.activeTabExporter;
    if (!exp) return;
    this.exportLabelsWith(exp, { ...this.formValues });
  }

  cancelExporterForm(): void {
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

    const labelsData = {
      labels: this.filteredLabels,
      selected_columns: this.enabledColumns.map((c) => c.key),
    };
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
          this.selectedExporter = null;
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
