import {
  Component,
  OnInit,
  computed,
  effect,
  inject,
  signal,
  input,
  output
} from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { LabelSessionService } from '../../../services/label-session.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { LabelFilter } from '../../../services/sorting-api.service';
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
  imports: [
    CommonModule,
    FormsModule,
    ModalComponent,
    FieldHintIconComponent,
    ClipboardCopyComponent,
  ],
  templateUrl: './export-modal.component.html',
  styleUrl: './export-modal.component.scss',
})
export class ExportModalComponent implements OnInit {
  readonly detectorName = input('');
  /**
   * The filter the modal opens on. ``unverified`` / ``verified`` are
   * server-side partitions (by Find ``verified_ids``) that can't be derived
   * client-side, so the modal fetches them with that ``label_filter``; the
   * other values are client-side category filters over the full fetched set.
   */
  readonly initialFilter = input<LabelFilter>('good');
  readonly closed = output<void>();
  readonly exported = output<void>();

  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private readonly exportersApi = inject(ExportersApiService);
  private readonly labelSession = inject(LabelSessionService);
  private readonly sortingApi = inject(SortingApiService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly datasetState = inject(DatasetStateService);

  // Two eager reads (dataset status + exporter list) load on creation; the
  // labels read is input-derived, so it waits for `ngOnInit` to set
  // `serverFilter` and flip `labelsReady`. All three wrap the generated-client
  // methods so the interceptor chain still applies, replacing the old
  // `ngOnInit` subscribes + `destroy$` plumbing.
  private readonly statusResource = rxResource({
    stream: () => this.datasetsRegistryApi.getStatus(),
  });
  private readonly exportersResource = rxResource({
    stream: () => this.exportersApi.getExporters(),
  });
  private readonly labelsReady = signal(false);
  private readonly labelsResource = rxResource({
    params: () => (this.labelsReady() ? this.serverFilter : undefined),
    stream: () => {
      const labelFilter = this.serverFilter === 'both' ? undefined : this.serverFilter;
      return this.sortingApi.exportLabels(false, { enrich: true, labelFilter });
    },
  });

  readonly exporters = computed<ExporterEntry[]>(() =>
    (this.exportersResource.value() ?? []).filter((e) => !e.hidden_from_picker),
  );

  /** Labels fetched from the server. */
  private readonly labelsList = computed<LabeledElement[]>(
    () => this.labelsResource.value()?.labels ?? [],
  );
  readonly labelsLoaded = computed(
    () => this.labelsResource.hasValue() || this.labelsResource.error() !== undefined,
  );

  /** Error from a failed export action; the read failures are merged in below. */
  private readonly actionError = signal('');
  readonly error = computed(
    () =>
      this.actionError() ||
      (this.exportersResource.error() ? 'Failed to load exporters' : '') ||
      (this.labelsResource.error() ? 'Failed to load labels' : ''),
  );
  // Written from the export-run subscribe (async); template-bound.
  readonly status = signal('');

  /** Column definitions with selection state, built dynamically from API response. */
  columns: ColumnDef[] = [];

  /** Client-side category filter over the fetched set (radio buttons). */
  labelFilter: 'good' | 'bad' | 'both' | 'corrections' = 'good';

  /**
   * Server-side partition the labels were fetched with. ``both`` fetches
   * everything (the radios then slice it); ``unverified`` / ``verified`` fetch
   * the Find work-queue / confirmed pile, which can't be sliced client-side.
   */
  serverFilter: 'both' | 'unverified' | 'verified' = 'both';

  /** Active export tab: 'clipboard' or an exporter name. */
  activeTab = 'clipboard';

  /** Exporter form state. */
  selectedExporter: ExporterEntry | null = null;
  formValues: Record<string, string> = {};
  readonly submitting = signal(false);

  /** Dataset display name for default filenames. */
  private readonly datasetName = computed(() => this.statusResource.value()?.display_name || '');

  /** Base columns that are always present. */
  private static readonly BASE_COLUMNS: { key: string; label: string }[] = [
    { key: 'label', label: 'Label' },
    { key: 'md5', label: 'MD5' },
    { key: 'filename', label: 'Filename' },
    { key: 'category', label: 'Category' },
  ];

  /** Columns excluded from checkboxes/preview but always appended to exports. */
  private static readonly ALWAYS_EXPORT_KEYS = ['origin', 'origin_name'];

  constructor() {
    // Rebuild the column set when the labels read settles. The checkbox
    // `enabled` state lives on `columns`, so it stays a mutable field rather
    // than a pure computed; the effect mirrors the old subscribe's
    // `buildColumns(...)` call (with the no-arg fallback on error).
    effect(() => {
      if (this.labelsResource.hasValue()) {
        this.buildColumns(this.labelsResource.value()?.available_columns);
      } else if (this.labelsResource.error()) {
        this.buildColumns();
      }
    });
  }

  /** Detector/model name from any available source. Falls back to the
   *  registry entry for the active detector id when the
   *  parent-supplied name and `labelSession.modelName` are both
   *  empty (typical when this modal opens from the Find view). */
  private get effectiveDetectorName(): string {
    const detectorName = this.detectorName();
    if (detectorName) return detectorName;
    if (this.labelSession.modelName) return this.labelSession.modelName;
    const modelId = this.activeContext.modelId;
    if (!modelId) return '';
    return this.datasetState.detectors.find((d) => d.id === modelId)?.name || '';
  }

  ngOnInit(): void {
    // Split the requested filter into a server-side partition (unverified /
    // verified are fetched with that label_filter) and a client-side category.
    const initialFilter = this.initialFilter();
    if (initialFilter === 'unverified' || initialFilter === 'verified') {
      this.serverFilter = initialFilter;
      this.labelFilter = 'both';
    } else if (initialFilter === 'unverified_good') {
      // The left work-queue export: the unverified partition (server-side),
      // sliced to the above-threshold good category (client-side).
      this.serverFilter = 'unverified';
      this.labelFilter = 'good';
    } else {
      this.serverFilter = 'both';
      this.labelFilter = initialFilter;
    }

    // Now that the input-derived `serverFilter` is set, release the labels read
    // (the dataset-status and exporter-list reads are eager and already in
    // flight). `buildColumns` rides the constructor effect on resolution.
    this.labelsReady.set(true);
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
    const labels = this.labelsList();
    if (this.labelFilter === 'good') {
      return labels.filter((e) => e.label === 'good');
    }
    if (this.labelFilter === 'bad') {
      return labels.filter((e) => e.label === 'bad');
    }
    if (this.labelFilter === 'corrections') {
      return labels.filter((e) => e.is_correction === true);
    }
    return labels;
  }

  /** Whether any labels are corrections (detector label was changed by user). */
  get hasCorrections(): boolean {
    return this.labelsList().some((e) => e.is_correction === true);
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

  /** Columns passed to the shared clipboard control (table mode). */
  get clipboardColumns(): ClipboardColumn[] {
    return this.exportColumns.map((c) => ({ key: c.key, label: c.label }));
  }

  /** Labels flattened to `{ columnKey: value }` rows for the clipboard control. */
  get clipboardRows(): Record<string, string>[] {
    const cols = this.exportColumns;
    return this.filteredLabels.map((entry) => {
      const row: Record<string, string> = {};
      for (const c of cols) row[c.key] = this.getCellValue(entry, c);
      return row;
    });
  }

  /** Build a descriptive default filename for export.
   *  e.g. "Good-MyDetector-MyDataset.json" */
  private buildDefaultFilename(ext: string): string {
    const parts: string[] = [];
    if (this.serverFilter === 'unverified') parts.push('Unverified');
    else if (this.serverFilter === 'verified') parts.push('Verified');
    if (this.labelFilter === 'good') parts.push('Good');
    else if (this.labelFilter === 'bad') parts.push('Bad');
    else if (this.labelFilter === 'corrections') parts.push('Corrections');
    const detName = this.effectiveDetectorName;
    if (detName) parts.push(detName);
    const datasetName = this.datasetName();
    if (datasetName) parts.push(datasetName);
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

  /** Start exporter flow: if no fields, export immediately. */
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
    this.actionError.set('');
    this.status.set('');
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
    this.actionError.set('');
    this.status.set('');
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
    return this.exporters().find((e) => e.name === this.activeTab) || null;
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
    this.actionError.set('');
    this.status.set('');
  }

  submitForm(): void {
    if (!this.selectedExporter) return;
    this.exportLabelsWith(this.selectedExporter, { ...this.formValues });
  }

  exportLabelsWith(exporter: ExporterEntry, fieldValues: Record<string, string>): void {
    const exporterLabel = exporter.display_name || exporter.name;
    this.status.set(`Exporting ${this.filteredLabels.length.toLocaleString()} labels to ${exporterLabel}…`);
    this.actionError.set('');
    this.submitting.set(true);

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
      .subscribe({
        next: () => {
          this.status.set('Labels exported.');
          this.submitting.set(false);
          this.selectedExporter = null;
          this.exported.emit();
        },
        error: () => {
          this.status.set('');
          this.actionError.set('Label export failed');
          this.submitting.set(false);
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

  /** Modal heading, noting the server-side partition (and category slice) when present. */
  get modalTitle(): string {
    if (this.serverFilter === 'unverified') {
      // The left work-queue export opens on the above-threshold good slice.
      return this.labelFilter === 'good' ? 'Export Unverified Good' : 'Export Unverified';
    }
    if (this.serverFilter === 'verified') return 'Export Verified';
    return 'Export';
  }

  close(): void {
    this.closed.emit();
  }
}
