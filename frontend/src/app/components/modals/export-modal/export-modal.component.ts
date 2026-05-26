import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { LabelSessionService } from '../../../services/label-session.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ImporterField } from '../../../models/api.models';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import type { LabeledElement } from '../../../generated/api-client/models/labeled-element';

export interface ColumnDef {
  key: string;
  label: string;
  enabled: boolean;
  isMetadata?: boolean;
}

@Component({
  selector: 'vt-export-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit, OnDestroy {
  @Input() detectorName = '';
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  exporters: ExporterEntry[] = [];
  loading = true;
  error = '';
  status = '';

  /** Labels fetched from the server. */
  labels: LabeledElement[] = [];
  labelsLoaded = false;

  /** Column definitions with selection state - built dynamically from API response. */
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

  /** Active export tab - 'clipboard' or an exporter name. */
  activeTab = 'clipboard';

  /** Exporter form state. */
  selectedExporter: ExporterEntry | null = null;
  formValues: Record<string, string> = {};
  submitting = false;

  /** Copy feedback. */
  copySuccess = false;

  /** Dataset display name for default filenames. */
  private datasetName = '';

  private destroy$ = new Subject<void>();

  /** Base columns that are always present. */
  private static readonly BASE_COLUMNS: { key: string; label: string }[] = [
    { key: 'label', label: 'Label' },
    { key: 'md5', label: 'MD5' },
    { key: 'filename', label: 'Filename' },
    { key: 'category', label: 'Category' },
  ];

  /** Columns excluded from checkboxes/preview but always appended to exports. */
  private static readonly ALWAYS_EXPORT_KEYS = ['origin', 'origin_name'];

  constructor(
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private exportersApi: ExportersApiService,
    private labelSession: LabelSessionService,
    private sortingApi: SortingApiService,
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
  ) {}

  /** Detector/model name from any available source. Falls back to the
   *  registry entry for the active detector id when the
   *  parent-supplied name and `labelSession.modelName` are both
   *  empty - typical when this modal opens from the Find view. */
  private get effectiveDetectorName(): string {
    if (this.detectorName) return this.detectorName;
    if (this.labelSession.modelName) return this.labelSession.modelName;
    const modelId = this.activeContext.modelId;
    if (!modelId) return '';
    return this.datasetState.detectors.find((d) => d.id === modelId)?.name || '';
  }

  ngOnInit(): void {
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => {
        this.datasetName = status.display_name || '';
      },
    });

    this.exportersApi.getExporters().pipe(takeUntil(this.destroy$)).subscribe({
      next: (list) => {
        this.exporters = list.filter((e) => !e.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load exporters';
      },
    });

    this.sortingApi.exportLabels(false, { enrich: true }).pipe(takeUntil(this.destroy$)).subscribe({
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
    const alwaysKeys = new Set(ExportModalComponent.ALWAYS_EXPORT_KEYS);
    // Start with base columns
    this.columns = ExportModalComponent.BASE_COLUMNS.map((c) => ({
      key: c.key,
      label: c.label,
      enabled: true,
    }));
    // Add metadata columns discovered from the data (skip always-export columns)
    if (availableColumns) {
      for (const key of availableColumns) {
        if (!baseKeys.has(key) && !alwaysKeys.has(key)) {
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

  get filteredLabels(): LabeledElement[] {
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

  get previewLabels(): LabeledElement[] {
    return this.filteredLabels.slice(0, 50);
  }

  get hasLabels(): boolean {
    return this.filteredLabels.length > 0;
  }

  get hasExporterForm(): boolean {
    return this.selectedExporter !== null;
  }

  getCellValue(entry: LabeledElement, col: ColumnDef): string {
    if (col.isMetadata) {
      const meta = entry.custom_metadata;
      if (meta && col.key in meta) {
        return String(meta[col.key] ?? '');
      }
      return '';
    }
    return String((entry as unknown as Record<string, unknown>)[col.key] ?? '');
  }

  /** Columns to export: user-selected columns plus always-export columns appended at the end. */
  private get exportColumns(): ColumnDef[] {
    const cols = [...this.enabledColumns];
    for (const key of ExportModalComponent.ALWAYS_EXPORT_KEYS) {
      cols.push({ key, label: key, enabled: true });
    }
    return cols;
  }

  /** Build delimited text from labels using selected columns. */
  buildExportText(): string {
    const cols = this.exportColumns;
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

  /** Build a descriptive default filename for export.
   *  e.g. "Good-MyDetector-MyDataset.json" */
  private buildDefaultFilename(ext: string): string {
    const parts: string[] = [];
    if (this.labelFilter === 'good') parts.push('Good');
    else if (this.labelFilter === 'bad') parts.push('Bad');
    else if (this.labelFilter === 'corrections') parts.push('Corrections');
    const detName = this.effectiveDetectorName;
    if (detName) parts.push(detName);
    if (this.datasetName) parts.push(this.datasetName);
    if (parts.length === 0) parts.push('labels');
    // Sanitise: replace characters unsafe for filenames with hyphens
    const stem = parts.join('-').replace(/[\\/:*?"<>|]+/g, '-');
    return `${stem}.${ext}`;
  }

  /** The exporter's plugin-defined fields, narrowed to the legacy
   *  ImporterField shape (the OpenAPI spec types it as an open dict
   *  because plugin field schemas aren't part of the generated client). */
  private exporterFieldsOf(exporter: ExporterEntry): ImporterField[] {
    return (exporter.fields ?? []) as ImporterField[];
  }

  /** Initial form value for *field*: its declared default, or the first
   *  option when a select field has no default (so the form is never sitting
   *  on a blank pulldown that the user has to actively populate). */
  private defaultFor(field: ImporterField): string {
    if (field.default) return field.default;
    if (
      field.field_type === 'select' &&
      !field.dynamic_options &&
      (field.options?.length ?? 0) > 0
    ) {
      return field.options![0];
    }
    return '';
  }

  /** Apply the dynamic default filename to the filepath form field if present. */
  private applyDefaultFilename(exporter: ExporterEntry): void {
    const filepathField = this.exporterFieldsOf(exporter).find((f) => f.key === 'filepath');
    if (filepathField) {
      const staticDefault = filepathField.default || '';
      // Derive extension from the static default (e.g. ".json", ".csv") or fall back
      const extMatch = staticDefault.match(/\.(\w+)$/);
      const ext = extMatch ? extMatch[1] : 'json';
      this.formValues['filepath'] = `data/${this.buildDefaultFilename(ext)}`;
    }
  }

  /** Start exporter flow - if no fields, export immediately. */
  startExporter(exporter: ExporterEntry): void {
    const fields = this.exporterFieldsOf(exporter);
    if (fields.length === 0) {
      this.exportLabelsWith(exporter, {});
      return;
    }
    this.selectedExporter = exporter;
    this.formValues = {};
    for (const f of fields) {
      this.formValues[f.key] = this.defaultFor(f);
    }
    this.applyDefaultFilename(exporter);
    this.error = '';
    this.status = '';
  }

  /** Select an exporter tab and initialise its form values. */
  selectExporterTab(exporter: ExporterEntry): void {
    this.activeTab = exporter.name;
    this.selectedExporter = exporter;
    this.formValues = {};
    for (const f of this.exporterFieldsOf(exporter)) {
      this.formValues[f.key] = this.defaultFor(f);
    }
    this.applyDefaultFilename(exporter);
    this.error = '';
    this.status = '';
  }

  /** Re-generate the default filename when the label filter changes. */
  onLabelFilterChange(): void {
    const exp = this.activeTabExporter;
    if (exp) {
      this.applyDefaultFilename(exp);
    }
  }

  /** The exporter object for the currently active tab (null if clipboard). */
  get activeTabExporter(): ExporterEntry | null {
    if (this.activeTab === 'clipboard') return null;
    return this.exporters.find((e) => e.name === this.activeTab) || null;
  }

  /** Typed view of the active tab's plugin fields for the template (the
   *  generated ExporterEntry types `fields` as an open dict because plugin
   *  field schemas aren't part of the OpenAPI client). */
  get activeTabExporterFields(): ImporterField[] {
    const exp = this.activeTabExporter;
    return exp ? this.exporterFieldsOf(exp) : [];
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

  exportLabelsWith(exporter: ExporterEntry, fieldValues: Record<string, string>): void {
    const exporterLabel = exporter.display_name || exporter.name;
    this.status = `Exporting ${this.filteredLabels.length.toLocaleString()} labels to ${exporterLabel}…`;
    this.error = '';
    this.submitting = true;

    const labelsData = {
      labels: this.filteredLabels,
      selected_columns: this.exportColumns.map((c) => c.key),
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

  /** Map an exporter's emoji icon to an SVG icon type. */
  getExporterIconType(exp: ExporterEntry): string {
    const icon = exp.icon || '';
    if (icon === '📧' || icon.includes('📧')) return 'email';
    if (icon === '🖥️' || icon === '\uD83D\uDDA5' || icon === '\uD83D\uDDA5\uFE0F') return 'server';
    if (icon === '🌐' || icon === '\uD83C\uDF10') return 'webhook';
    // Also match by name as fallback
    const name = (exp.name || '').toLowerCase();
    if (name.includes('email') || name.includes('smtp')) return 'email';
    if (name.includes('webhook')) return 'webhook';
    if (name.includes('server') || name.includes('file')) return 'server';
    return 'upload';
  }

  close(): void {
    this.closed.emit();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
