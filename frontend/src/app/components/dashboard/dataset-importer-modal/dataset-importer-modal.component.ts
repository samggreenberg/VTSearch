import { Component, EventEmitter, HostListener, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { ClipperChooserComponent, ClipperSelection } from '../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from './import-advanced/import-advanced.component';
import { ImportConfigComponent } from './import-config/import-config.component';
import { SourcePickerComponent } from './source-picker/source-picker.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { ImporterInfo, ImporterField, ImporterPickerTab, DemoDataset, MediaTypeInfo, MediaTypeDetectionResponse, ClipperInfo, ClipperParameter, EmbedderInfo, ConverterInfo, SourceSpec } from '../../../models/api.models';
import { ColMeta, ManagedColumns } from '../../../utils/managed-columns';

@Component({
  selector: 'vt-dataset-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, ClipperChooserComponent, ImportAdvancedComponent, ImportConfigComponent, SourcePickerComponent],
  templateUrl: './dataset-importer-modal.component.html',
  styleUrl: './dataset-importer-modal.component.scss',
})
export class DatasetImporterModalComponent implements OnInit {
  /** Media type_id guessed from existing datasets/models (e.g. "image"). */
  @Input() guessedMediaType = '';
  /** Embedder name guessed from existing datasets/in-progress loads (e.g. "siglip"). */
  @Input() guessedMediaEmbedder = '';
  /** Picker tab id to pre-select when the modal opens (e.g. "server" from
   *  the dashboard's first-run welcome banner CTA).  Empty leaves the
   *  picker in the default "no tab selected" state. */
  @Input() initialTab = '';

  @Output() closed = new EventEmitter<void>();
  @Output() importStarted = new EventEmitter<void>();
  @Output() demoSelected = new EventEmitter<DemoDataset>();

  importers: ImporterInfo[] = [];
  selectedImporter: ImporterInfo | null = null;
  formValues: Record<string, any> = {};
  selectedFile: File | null = null;
  submitting = false;
  error = '';
  /** Whether the user has manually edited the generic-form dataset_name
   *  input (so we stop auto-deriving it from path/url/file fields). */
  private formDatasetNameDirty = false;

  // Clipper state
  availableClippers: ClipperInfo[] = [];
  selectedClipper = '';
  clipperParamValues: Record<string, number | string> = {};

  // Embedder state
  allEmbedders: EmbedderInfo[] = [];
  availableEmbedders: EmbedderInfo[] = [];
  selectedEmbedder = '';

  // Demo picker state
  demos: DemoDataset[] = [];
  mediaTypes: MediaTypeInfo[] = [];
  demoTabs: string[] = [];
  activeTab = '';
  demoLoading = false;
  demoEmbedders: EmbedderInfo[] = [];
  selectedDemoEmbedder = '';
  demoEmbedder = '';
  demoClippers: ClipperInfo[] = [];
  selectedDemoClipper = '';
  /** Optional user-supplied dataset name for demo imports.  Empty means
   *  "use the demo entry's label". */
  demoDatasetName = '';

  // Demo table column metadata + controller. Mirrors the Dashboard datagrid:
  // percentage widths summing to 100, draggable column reorder, click-to-sort
  // headers, and column-resize handles between cells.
  static readonly DEMO_COL_META: Record<string, ColMeta> = {
    label: { label: 'Name', title: 'Demo dataset name (click to sort)', sortable: true },
    num_files: { label: '# Media', title: 'Number of media files in the demo dataset (click to sort)', sortable: true },
    num_categories: { label: '# Cat.', title: 'Number of distinct categories or classes in the dataset (click to sort)', sortable: true },
    description: { label: 'Description', title: 'Short description of the demo dataset contents (click to sort)', sortable: true },
    status: { label: 'Readiness', title: 'Whether the dataset is pre-downloaded and ready to load immediately, or needs to be fetched first (click to sort)', sortable: true },
  };
  static readonly DEMO_COLUMNS_DEFAULT = ['label', 'num_files', 'num_categories', 'description', 'status'];
  private static readonly DEMO_COL_ORDER_KEY = 'vtsearch.dashboard.demoColumnOrder';

  demoCols = new ManagedColumns(
    DatasetImporterModalComponent.DEMO_COLUMNS_DEFAULT,
    DatasetImporterModalComponent.DEMO_COL_META,
    { initialSort: 'num_files', storageKey: DatasetImporterModalComponent.DEMO_COL_ORDER_KEY },
  );

  /** Source-specs editor state for the form view (server_files +
   *  any future ``multi_media=True`` form-style importer). */
  formSourceSpecs: SourceSpec[] = [];

  /** Source-specs editor state for the local-folder / local-files
   *  views.  These uploads go through ``/api/dataset/import-local-folder``,
   *  which delegates to server_folder — so the same multi-media flow
   *  applies. */
  lfSourceSpecs: SourceSpec[] = [];

  // Local folder upload state — files come from the browser machine
  lfFiles: File[] = [];
  lfMediaType = '';
  lfMediaTypeOptions: string[] = [];
  lfEmbedders: EmbedderInfo[] = [];
  lfSelectedEmbedder = '';
  lfClippers: ClipperInfo[] = [];
  lfSelectedClipper = '';
  lfClipperParams: ClipperParameter[] = [];
  lfClipperParamValues: Record<string, number | string> = {};
  lfSubmitting = false;
  lfError = '';
  /** ``"folder"`` opens a directory picker (Local Folder card),
   *  ``"files"`` opens a multi-file picker (Local Files card). */
  lfPickerKind: 'folder' | 'files' = 'folder';
  /** Whether subfolders inside the picked local folder are included. */
  lfRecursive = true;
  /** Optional user-supplied dataset name for local-folder uploads. */
  lfDatasetName = '';
  /** Whether the user has manually edited :prop:`lfDatasetName` (so we stop
   *  auto-overwriting it from the picked folder name). */
  private lfDatasetNameDirty = false;

  // Server folder picker state. The user types an absolute server path; we
  // split it into ``sfBrowseRootPath`` (always ``/`` here) and ``sfBrowsePath``
  // (the typed path with the leading slash stripped) so the existing
  // detection / submit / dataset-name helpers keep working unchanged.
  sfBrowsePath = '';
  sfBrowseRootPath = '';
  sfBrowseError = '';
  sfMediaType = '';
  sfMediaTypeOptions: string[] = [];
  sfEmbedders: EmbedderInfo[] = [];
  sfSelectedEmbedder = '';
  sfClippers: ClipperInfo[] = [];
  sfSelectedClipper = '';
  sfClipperParams: ClipperParameter[] = [];
  sfClipperParamValues: Record<string, number | string> = {};
  sfSubmitting = false;
  /** Whether subdirectories of the picked server folder are scanned. */
  sfRecursive = true;
  /** Optional user-supplied dataset name for server-folder imports. */
  sfDatasetName = '';
  /** Whether the user has manually edited :prop:`sfDatasetName`. */
  private sfDatasetNameDirty = false;
  /** Multi-media import rows for the server_folder picker.  Each row is a
   *  ``(source_type, converter|null, params)`` triple — see
   *  ``docs/plans/multi-media-import.md``. */
  sfSourceSpecs: SourceSpec[] = [];

  /** Auto-detect result for the local-folder / local-files picker.  Set
   *  after the user picks files; ``null`` when no detection has been run
   *  for the current selection. */
  lfDetection: MediaTypeDetectionResponse | null = null;
  /** Same as :prop:`lfDetection` but for the server-folder picker.  Filled
   *  in by an API call after each successful directory load. */
  sfDetection: MediaTypeDetectionResponse | null = null;

  // Dynamic-options cache for the generic form view.  Keyed by ImporterField.key.
  /** Options last fetched from the backend for dynamic-options fields. */
  dynamicFieldOptions: Record<string, string[]> = {};
  /** Whether a dynamic-options fetch is currently in flight for a field. */
  dynamicFieldLoading: Record<string, boolean> = {};
  /** Last fetch error for a dynamic-options field, if any. */
  dynamicFieldError: Record<string, string> = {};

  // Clipper chooser modal state
  clipperChooserOpen = false;
  /** Which context opened the chooser: 'form' | 'demo' | 'sf' | 'lf' */
  clipperChooserContext: 'form' | 'demo' | 'sf' | 'lf' = 'form';
  clipperChooserClippers: ClipperInfo[] = [];

  constructor(
    private datasetsApi: DatasetsApiService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnInit(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.importers = (res.importers || []).filter((imp) => !imp['hidden_from_picker']);
        this.declaredTabs = res.tabs || [];
        if (this.initialTab && this.visibleImporterTabs.some((t) => t.id === this.initialTab)) {
          this.selectImporterTab(this.initialTab);
        }
      },
    });
    this.datasetsApi.getEmbedders().subscribe({
      next: (embedders) => {
        this.allEmbedders = embedders || [];
      },
    });
    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes = res.media_types || [];
      },
    });
    // Settings carry the per-media-type "last embedder" memory used to
    // pre-select an embedder when no loaded dataset can supply
    // ``guessedMediaEmbedder``.
    this.settingsState.load();
  }

  /** Pick the initial embedder for a picker view.  Priority:
   *  1. ``guessedMediaEmbedder`` (computed from currently loaded datasets)
   *  2. the user's last pick for this media type (per-user setting)
   *  3. first option, or empty when the list is empty.
   *
   *  ``mediaTypeFolderOrTypeId`` accepts either form — the importer form
   *  values use the folder name (e.g. ``"images"``) while the settings
   *  map is keyed by canonical type_id (e.g. ``"image"``).  We resolve to
   *  type_id before looking up the saved setting. */
  private pickInitialEmbedder(embedders: EmbedderInfo[], mediaTypeFolderOrTypeId: string): string {
    if (embedders.length === 0) return '';
    const guessedMatch = this.guessedMediaEmbedder
      ? embedders.find((e) => e.name === this.guessedMediaEmbedder)
      : null;
    if (guessedMatch) return guessedMatch.name;
    const typeId = this.toTypeId(mediaTypeFolderOrTypeId) || mediaTypeFolderOrTypeId;
    const savedMap = this.settingsState.settings?.last_embedder_per_media_type || {};
    const saved = savedMap[typeId];
    if (saved) {
      const savedMatch = embedders.find((e) => e.name === saved);
      if (savedMatch) return savedMatch.name;
    }
    return embedders[0].name;
  }

  /** Front-of-list order for the picker within each tab.  Importers not
   *  listed here come after these in registry order. */
  private static readonly PICKER_ORDER = [
    'local_folder',
    'local_files',
    'server_folder',
    'server_files',
    'demo',
    'synthetic',
  ];

  /** Tab declarations supplied by the backend (``/api/dataset/all-importers``). */
  declaredTabs: ImporterPickerTab[] = [];

  /** Currently selected picker tab.  Empty string means no tab is selected
   *  yet, so the content area below the tab bars stays blank. */
  activeImporterTab = '';

  get orderedImporters(): ImporterInfo[] {
    const order = DatasetImporterModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      const imp = this.importers.find((i) => i.name === name);
      if (imp && !imp['hidden_from_picker']) result.push(imp);
    }
    for (const imp of this.importers) {
      if (!order.includes(imp.name) && !imp['hidden_from_picker']) {
        result.push(imp);
      }
    }
    return result;
  }

  /** Title-case an importer category id when no backend declaration exists.
   *  ``"my_cloud"`` → ``"My Cloud"``. */
  private fallbackTabLabel(id: string): string {
    return id
      .split(/[\s_-]+/)
      .filter(Boolean)
      .map((part) => part[0].toUpperCase() + part.slice(1))
      .join(' ');
  }

  /** Picker tabs to display.  All tabs declared by the backend render in
   *  their declared order, regardless of whether any importers populate
   *  them — so categories like "Services" remain visible even when no
   *  extension importers are installed.  Categories used by importers but
   *  never declared get appended at the end with a title-cased label and
   *  no icon. */
  get visibleImporterTabs(): ImporterPickerTab[] {
    const visible: ImporterPickerTab[] = [];
    const seen = new Set<string>();
    const declared = [...this.declaredTabs].sort(
      (a, b) => (a.order ?? 100) - (b.order ?? 100),
    );
    for (const tab of declared) {
      visible.push(tab);
      seen.add(tab.id);
    }
    const usedCategories = new Set(
      this.orderedImporters.map((imp) => imp.category || '').filter(Boolean),
    );
    for (const id of usedCategories) {
      if (!seen.has(id)) {
        visible.push({ id, label: this.fallbackTabLabel(id) });
      }
    }
    return visible;
  }

  /** Importers belonging to the active tab. */
  get importersForActiveTab(): ImporterInfo[] {
    return this.orderedImporters.filter(
      (imp) => (imp.category || '') === this.activeImporterTab,
    );
  }

  /** Display label of the currently selected category tab, or empty when none. */
  get activeImporterTabLabel(): string {
    const tab = this.visibleImporterTabs.find((t) => t.id === this.activeImporterTab);
    return tab?.label || '';
  }

  selectImporterTab(tabId: string): void {
    this.activeImporterTab = tabId;
    this.selectedImporter = null;
    // When the tab has exactly one importer, the inner sub-tab row is
    // redundant — clicking the outer tab already declared the intent.
    // Auto-select the lone importer so the user lands directly on its
    // form instead of having to click a single-option card.
    const importers = this.importersForActiveTab;
    if (importers.length === 1 && importers[0]['enabled'] !== false) {
      this.selectImporter(importers[0]);
    }
  }

  /** Title shown at the top of the modal. */
  get modalTitle(): string {
    return 'Add Dataset';
  }

  /** ``picker_view`` of the currently selected importer, or empty when
   *  nothing is selected.  Drives which inline widget set is rendered
   *  below the inner tab row. */
  get activePickerView(): string {
    return this.selectedImporter?.picker_view || '';
  }

  selectImporter(importer: ImporterInfo): void {
    // Dispatch to the importer-specific setup logic.  Each helper
    // populates its own state slice but no longer changes a global
    // ``view`` because the modal renders all tab levels together.
    const pickerView = importer.picker_view || 'form';
    if (pickerView === 'local_folder' || pickerView === 'local_files') {
      this.openLocalFolderUploader(importer);
      return;
    }
    if (pickerView === 'server_folder') {
      this.openServerFolderBrowser(importer);
      return;
    }
    if (pickerView === 'demo') {
      this.openDemoPicker(importer);
      return;
    }

    this.selectedImporter = importer;
    this.formValues = {};
    this.error = '';
    this.selectedClipper = '';
    this.availableClippers = [];
    this.clipperParamValues = {};
    this.selectedEmbedder = '';
    this.availableEmbedders = [];
    this.dynamicFieldOptions = {};
    this.dynamicFieldLoading = {};
    this.dynamicFieldError = {};
    this.formDatasetNameDirty = false;

    // Pre-populate defaults
    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default !== undefined) {
          this.formValues[field.key] = field.default;
        }
      }
    }

    // Override media_type default with guessed type when available
    const mediaTypeField = importer.fields?.find((f) => f.key === 'media_type');
    if (mediaTypeField && this.guessedMediaType) {
      const folderName = this.toFolderName(this.guessedMediaType);
      if (folderName && mediaTypeField.options?.includes(folderName)) {
        this.formValues['media_type'] = folderName;
      }
    }

    // Load clippers and embedders for the default media type
    if (mediaTypeField) {
      const defaultType = this.formValues['media_type'] || mediaTypeField.default || '';
      this.loadClippers(defaultType);
      this.loadEmbedders(defaultType);
    }

    // Reset the multi-media source-specs list so this picker opens with
    // a single "include directly" row for the active output type.
    this.resetFormSourceSpecs();

    // Fetch initial options for dynamic fields whose deps are already filled.
    for (const field of importer.fields || []) {
      if (field.dynamic_options) {
        this.refreshDynamicFieldOptions(field);
      }
    }
  }

  onMediaTypeChange(mediaType: string): void {
    this.formValues['media_type'] = mediaType;
    this.loadClippers(mediaType);
    this.loadEmbedders(mediaType);
    this.resetFormSourceSpecs();
    this.onFormFieldChanged('media_type');
  }

  /** Called whenever a form field value changes.  Refreshes options for
   *  every dynamic-options field whose ``depends_on`` includes *changedKey*.
   *  Also clears the dependent field's current value so the user can't
   *  submit a now-stale selection.  Also re-derives a default
   *  ``dataset_name`` from the change unless the user has typed one. */
  onFormFieldChanged(changedKey: string): void {
    const importer = this.selectedImporter;
    if (!importer?.fields) return;
    for (const field of importer.fields) {
      if (!field.dynamic_options) continue;
      if (!(field.depends_on || []).includes(changedKey)) continue;
      this.formValues[field.key] = '';
      this.refreshDynamicFieldOptions(field);
    }
    if (changedKey !== 'dataset_name') {
      this.maybeApplyDerivedDatasetName();
    }
  }

  /** Called when the user types into the generic-form ``Dataset Name``
   *  input.  Marks the field as dirty so subsequent path/url/file changes
   *  do not overwrite the user's value. */
  formOnDatasetNameInput(value: string): void {
    this.formValues['dataset_name'] = value;
    this.formDatasetNameDirty = true;
  }

  /** Called when the user types into a ``server_path`` form field.
   *  Updates the form value and re-derives the dataset name when the
   *  user hasn't typed one yet. */
  formOnServerPathSelected(key: string, path: string): void {
    this.formValues[key] = path;
    this.onFormFieldChanged(key);
  }

  /** Re-derive a default dataset name from the current form values when
   *  the user hasn't manually edited the dataset_name input.  Inspects
   *  the importer's fields and uses the first source-style field with a
   *  value (url, server_path, file, or a field keyed ``path``). */
  private maybeApplyDerivedDatasetName(): void {
    if (this.formDatasetNameDirty) return;
    const derived = this.formDerivedDatasetName();
    if (derived) {
      this.formValues['dataset_name'] = derived;
    }
  }

  /** Compute a derived dataset name from the current form values. */
  private formDerivedDatasetName(): string {
    const fields = this.selectedImporter?.fields || [];
    for (const f of fields) {
      if (f.key === 'dataset_name') continue;
      const raw = this.formValues[f.key];
      if (typeof raw !== 'string' || !raw) continue;
      if (f.field_type === 'url') {
        const cleaned = raw.split('?')[0].replace(/\/+$/, '');
        const tail = cleaned.split('/').pop() || '';
        if (!tail) continue;
        const stripped = tail.replace(
          /\.(?:tar\.gz|tar\.bz2|tar\.xz|tar|zip|rar)$/i, '',
        );
        return stripped || tail;
      }
      if (f.field_type === 'server_path' || f.field_type === 'file') {
        const basename = raw.split(/[\\/]/).pop() || '';
        if (!basename) continue;
        const dot = basename.lastIndexOf('.');
        return dot > 0 ? basename.slice(0, dot) : basename;
      }
      if (f.key === 'path') {
        const parts = raw.split(/[\\/]/).filter(Boolean);
        if (parts.length > 0) return parts[parts.length - 1];
      }
    }
    return '';
  }

  /** Fetch the option list for a dynamic-options field from the backend. */
  private refreshDynamicFieldOptions(field: ImporterField): void {
    const importer = this.selectedImporter;
    if (!importer) return;
    const key = field.key;
    this.dynamicFieldLoading[key] = true;
    this.dynamicFieldError[key] = '';
    this.datasetsApi
      .getImporterFieldOptions(importer.name, key, { ...this.formValues })
      .subscribe({
        next: (res) => {
          this.dynamicFieldOptions[key] = res.options || [];
          this.dynamicFieldLoading[key] = false;
          // If the current value isn't in the new option list, clear it so
          // the displayed select matches the value.  Pre-select the first
          // option when the field is required and currently empty.
          const current = this.formValues[key];
          if (current && !this.dynamicFieldOptions[key].includes(String(current))) {
            this.formValues[key] = '';
          }
          if (!this.formValues[key] && field.required && this.dynamicFieldOptions[key].length > 0) {
            this.formValues[key] = this.dynamicFieldOptions[key][0];
          }
        },
        error: (err) => {
          this.dynamicFieldLoading[key] = false;
          this.dynamicFieldError[key] = err?.error?.error || 'Could not load options';
          this.dynamicFieldOptions[key] = [];
        },
      });
  }

  /** Effective option list for a select field — dynamic options when set,
   *  otherwise the static options declared on the field. */
  optionsFor(field: ImporterField): string[] {
    if (field.dynamic_options) {
      return this.dynamicFieldOptions[field.key] || [];
    }
    return field.options || [];
  }

  private loadClippers(mediaType: string): void {
    if (!mediaType) {
      this.availableClippers = [];
      this.selectedClipper = '';
      return;
    }
    this.datasetsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.availableClippers = clippers;
        // Default to the first clipper (the default/null clipper)
        this.selectedClipper = clippers.length > 0 ? clippers[0].name : '';
        this.resetClipperParams();
      },
    });
  }

  onClipperChange(clipperName: string): void {
    this.selectedClipper = clipperName;
    this.resetClipperParams();
  }

  get selectedClipperParams(): ClipperParameter[] {
    const clipper = this.availableClippers.find((c) => c.name === this.selectedClipper);
    return clipper?.parameters || [];
  }

  private resetClipperParams(): void {
    this.clipperParamValues = {};
    for (const param of this.selectedClipperParams) {
      this.clipperParamValues[param.key] = param.default;
    }
  }

  private loadEmbedders(mediaType: string): void {
    if (!mediaType) {
      this.availableEmbedders = [];
      this.selectedEmbedder = '';
      return;
    }
    this.datasetsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.availableEmbedders = embedders;
        this.selectedEmbedder = this.pickInitialEmbedder(embedders, mediaType);
      },
    });
  }

  openDemoPicker(importer?: ImporterInfo): void {
    this.selectedImporter = importer || this.importers.find((i) => i.name === 'demo') || null;
    this.demoLoading = true;
    this.demos = [];
    this.demoTabs = [];
    this.activeTab = '';
    this.demoDatasetName = '';

    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes = res.media_types || [];
        this.fetchDemos();
      },
      error: () => {
        this.fetchDemos();
      },
    });
  }

  private fetchDemos(embedder?: string): void {
    this.datasetsApi.getDemoList(embedder).subscribe({
      next: (demoRes) => {
        this.demos = demoRes.datasets || [];
        this.buildDemoTabs();
        this.demoLoading = false;
      },
      error: () => {
        this.demoLoading = false;
      },
    });
  }

  private buildDemoTabs(): void {
    const grouped = new Set(this.demos.map((d) => d.media_type));
    // Order by media type registry order, then any remaining
    const registryOrder = this.mediaTypes.map((mt) => mt.type_id);
    this.demoTabs = registryOrder.filter((mt) => grouped.has(mt));
    // Add any types not in registry
    for (const mt of grouped) {
      if (!this.demoTabs.includes(mt)) {
        this.demoTabs.push(mt);
      }
    }
    // Intentionally leave ``activeTab`` blank — no media-type tab is
    // auto-selected.  The demo table stays empty until the user clicks
    // one of the inner tabs.
  }

  private loadDemoEmbedders(mediaType: string): void {
    if (!mediaType) {
      this.demoEmbedders = [];
      this.selectedDemoEmbedder = '';
      this.demoClippers = [];
      this.selectedDemoClipper = '';
      return;
    }
    this.datasetsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.demoEmbedders = embedders;
        this.selectedDemoEmbedder = this.pickInitialEmbedder(embedders, mediaType);
        this.demoEmbedder = this.selectedDemoEmbedder;
        this.updateDemoStatuses();
        // The initial demo fetch had no embedder/clipper context, so re-fetch
        // with the now-known defaults for authoritative status values.
        if (this.selectedDemoEmbedder) {
          this.refetchDemoStatuses(this.selectedDemoEmbedder, this.selectedDemoClipper);
        }
      },
    });
    this.datasetsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.demoClippers = clippers;
        this.selectedDemoClipper = clippers.length > 0 ? clippers[0].name : '';
      },
    });
  }

  onDemoEmbedderChange(embedder: string): void {
    this.selectedDemoEmbedder = embedder;
    this.demoEmbedder = embedder;
    this.updateDemoStatuses();
    this.refetchDemoStatuses(embedder, this.selectedDemoClipper);
  }

  onDemoClipperChange(clipper: string): void {
    this.selectedDemoClipper = clipper;
    this.updateDemoStatuses();
    this.refetchDemoStatuses(this.selectedDemoEmbedder, this.selectedDemoClipper);
  }

  /**
   * Re-compute each demo's status client-side based on the selected embedder
   * and clipper.  A demo that has a cached pkl is only "ready" when both the
   * pkl embedder and clipper match the currently selected values; otherwise it
   * downgrades to "needs_embedding".
   *
   * Only processes demos for the active tab to avoid accidentally changing
   * statuses of demos from other media types.
   */
  private updateDemoStatuses(): void {
    const emb = this.selectedDemoEmbedder;
    const clip = this.selectedDemoClipper;
    for (const demo of this.demos) {
      if (demo.media_type !== this.activeTab) continue;
      if (demo.status === 'needs_download') continue;

      if (!demo.pkl_embedder) {
        if (emb && demo.status === 'ready') {
          demo.status = 'needs_embedding';
          demo.ready = false;
        }
        continue;
      }

      const embedderMatch = !emb || demo.pkl_embedder === emb;
      const clipperMatch = !clip || !demo.pkl_clipper || demo.pkl_clipper === clip;

      if (embedderMatch && clipperMatch) {
        demo.status = 'ready';
        demo.ready = true;
      } else {
        demo.status = 'needs_embedding';
        demo.ready = false;
      }
    }
  }

  /**
   * Re-fetch the demo list from the server with the given embedder and clipper
   * so the backend can authoritatively determine each demo's status.
   */
  private refetchDemoStatuses(embedder: string, clipper?: string): void {
    this.datasetsApi.getDemoList(embedder, clipper).subscribe({
      next: (demoRes) => {
        this.demos = demoRes.datasets || [];
      },
    });
  }

  get filteredDemos(): DemoDataset[] {
    const items = this.demos.filter((d) => d.media_type === this.activeTab);
    const statusOrder: Record<string, number> = { ready: 0, needs_embedding: 1, needs_download: 2 };
    const sortKey = this.demoCols.sortColumn;
    const asc = this.demoCols.sortAsc;
    return items.sort((a, b) => {
      const key = sortKey as keyof DemoDataset;
      let va: any = a[key];
      let vb: any = b[key];
      if (key === 'status') {
        va = statusOrder[va as string] ?? 3;
        vb = statusOrder[vb as string] ?? 3;
      }
      if (typeof va === 'number' && typeof vb === 'number') {
        return asc ? va - vb : vb - va;
      }
      va = String(va || '').toLowerCase();
      vb = String(vb || '').toLowerCase();
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  selectDemoTab(tab: string): void {
    this.activeTab = tab;
    this.loadDemoEmbedders(tab);
  }

  // --- Document-level resize tracking ---

  @HostListener('document:mousemove', ['$event'])
  onDocResizeMove(event: MouseEvent): void {
    this.demoCols.onResizeMove(event);
  }

  @HostListener('document:mouseup')
  onDocResizeEnd(): void {
    this.demoCols.onResizeEnd();
  }

  /** Convert a type_id (e.g. "image") to the corresponding folder_import_name (e.g. "images"). */
  private toFolderName(typeId: string): string {
    if (!typeId) return '';
    const mt = this.mediaTypes.find((m) => m.type_id === typeId);
    return mt?.folder_import_name || typeId;
  }

  getTabLabel(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    if (mt) {
      return mt.name.trim();
    }
    return mediaType;
  }

  /** Cached map of type_id → human label, rebuilt only when
   *  ``mediaTypes`` is reassigned.  Passed to
   *  ``<vt-source-specs-picker>`` so its child change-detection sees a
   *  stable input reference. */
  private _typeLabelsSource: MediaTypeInfo[] | null = null;
  private _typeLabelsCache: Record<string, string> = {};
  get mediaTypeLabels(): Record<string, string> {
    if (this._typeLabelsSource !== this.mediaTypes) {
      const out: Record<string, string> = {};
      for (const mt of this.mediaTypes) out[mt.type_id] = mt.name.trim();
      this._typeLabelsCache = out;
      this._typeLabelsSource = this.mediaTypes;
    }
    return this._typeLabelsCache;
  }

  /** Cached map of ``folder_import_name`` → human label, rebuilt only
   *  when ``mediaTypes`` is reassigned.  Feeds the media-type dropdown
   *  inside :component:`ImportConfigComponent` so the parent does not
   *  need to pass a per-render label function. */
  private _optionLabelsSource: MediaTypeInfo[] | null = null;
  private _optionLabelsCache: Record<string, string> = {};
  get mediaTypeOptionLabels(): Record<string, string> {
    if (this._optionLabelsSource !== this.mediaTypes) {
      const out: Record<string, string> = {};
      for (const mt of this.mediaTypes) {
        if (mt.folder_import_name) out[mt.folder_import_name] = mt.name.trim();
      }
      this._optionLabelsCache = out;
      this._optionLabelsSource = this.mediaTypes;
    }
    return this._optionLabelsCache;
  }

  /** Embedders available for the currently active demo tab's media type. */
  get demoEmbeddersForTab(): EmbedderInfo[] {
    return this.allEmbedders.filter((e) => e.media_type_id === this.activeTab);
  }

  selectDemo(demo: DemoDataset): void {
    const userName = (this.demoDatasetName || '').trim();
    this.demoSelected.emit({
      ...demo,
      embedder: this.selectedDemoEmbedder,
      clipper: this.selectedDemoClipper,
      dataset_name: userName,
    } as any);
    this.closed.emit();
  }

  selectDemoTabWithEmbedder(tab: string): void {
    this.selectDemoTab(tab);
    // Reset embedder selection for the new tab
    const embedders = this.allEmbedders.filter((e) => e.media_type_id === tab);
    this.demoEmbedder = embedders.length > 0 ? embedders[0].name : '';
    // Clipper is reset by loadDemoEmbedders (called from selectDemoTab)
  }

  /** Read the ``recursive`` field's declared default ("true"/"false") from
   *  the importer metadata; defaults to ``true`` when the field is absent. */
  private readRecursiveDefault(importer: ImporterInfo | null): boolean {
    const field = importer?.fields?.find((f) => f.key === 'recursive');
    if (!field) return true;
    return String(field.default ?? 'true').toLowerCase() !== 'false';
  }

  // --- Local folder upload (files come from the browser machine) ---

  openLocalFolderUploader(importer?: ImporterInfo): void {
    const resolved = importer
      || this.importers.find((i) => i.name === 'local_folder')
      || null;
    this.selectedImporter = resolved;
    this.lfPickerKind = resolved?.name === 'local_files' ? 'files' : 'folder';
    this.lfFiles = [];
    this.lfDetection = null;
    this.lfError = '';
    this.lfSubmitting = false;
    this.lfRecursive = this.readRecursiveDefault(resolved);
    this.lfDatasetName = '';
    this.lfDatasetNameDirty = false;

    // Reuse the server_folder importer's media_type options for consistency.
    const folderImporter = this.importers.find((imp) => imp.name === 'server_folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    this.lfMediaTypeOptions = mtField?.options || [];

    const guessedFolder = this.toFolderName(this.guessedMediaType);
    if (guessedFolder && this.lfMediaTypeOptions.includes(guessedFolder)) {
      this.lfMediaType = guessedFolder;
    } else {
      this.lfMediaType = mtField?.default || this.lfMediaTypeOptions[0] || 'audio';
    }

    this.lfLoadEmbedders(this.lfMediaType);
    this.lfLoadClippers(this.lfMediaType);
    this.lfResetSourceSpecs();
  }

  lfOnFilesDropped(files: File[]): void {
    this.lfAcceptFiles(files);
  }

  private lfAcceptFiles(files: File[]): void {
    if (files.length === 0) {
      this.lfFiles = [];
      this.lfDetection = null;
      return;
    }
    // Local Files uploads a single paths file (not media), so type detection
    // doesn't apply — the user picks the media type explicitly.
    if (this.lfPickerKind === 'files') {
      this.lfFiles = [files[0]];
      this.lfDetection = null;
      this.lfError = '';
      if (!this.lfDatasetNameDirty) {
        this.lfDatasetName = this.lfDerivedDatasetName();
      }
      return;
    }
    this.lfFiles = files;
    this.lfError = '';
    if (!this.lfDatasetNameDirty) {
      this.lfDatasetName = this.lfDerivedDatasetName();
    }
    this.lfDetection = this.detectFromFiles(this.lfFiles, this.lfRecursive);
    this.lfApplyDetection();
  }

  lfOnRecursiveChange(recursive: boolean): void {
    this.lfRecursive = recursive;
    if (this.lfFiles.length > 0) {
      this.lfDetection = this.detectFromFiles(this.lfFiles, this.lfRecursive);
      this.lfApplyDetection();
    }
  }

  sfOnRecursiveChange(recursive: boolean): void {
    this.sfRecursive = recursive;
    if (this.sfBrowseRootPath) {
      this.sfRunDetection();
    }
  }

  /** Apply :prop:`lfDetection` to the lf-* view: set the output media-type
   *  dropdown and rebuild the source-spec rows.  Re-fetches embedders and
   *  clippers via :prop:`lfOnMediaTypeChange` when the chosen type changed
   *  so the rest of the form stays consistent. */
  private lfApplyDetection(): void {
    if (!this.lfDetection) return;
    const { mediaType, sourceSpecs } = this.autofillFromDetection(
      this.lfDetection,
      this.lfMediaTypeOptions,
      (typeId) => this.availableConvertersFor('server_folder', typeId),
    );
    if (mediaType && mediaType !== this.lfMediaType) {
      this.lfMediaType = mediaType;
      this.lfLoadEmbedders(this.lfMediaType);
      this.lfLoadClippers(this.lfMediaType);
    }
    if (sourceSpecs) {
      this.lfSourceSpecs = sourceSpecs;
    }
  }

  /** Derive a default dataset name from the currently picked files / folder.
   *  Used to pre-fill the Dataset Name input until the user edits it. */
  private lfDerivedDatasetName(): string {
    if (this.lfPickerKind === 'folder' && this.lfFolderName) {
      return this.lfFolderName;
    }
    if (this.lfFiles.length === 1) {
      const name = this.lfFiles[0].name || '';
      const dot = name.lastIndexOf('.');
      return dot > 0 ? name.slice(0, dot) : name;
    }
    return '';
  }

  lfOnDatasetNameInput(value: string): void {
    this.lfDatasetName = value;
    this.lfDatasetNameDirty = true;
  }

  lfOnMediaTypeChange(mediaType: string): void {
    this.lfMediaType = mediaType;
    this.lfLoadEmbedders(mediaType);
    this.lfLoadClippers(mediaType);
    this.lfResetSourceSpecs();
  }

  private lfLoadEmbedders(mediaType: string): void {
    if (!mediaType) {
      this.lfEmbedders = [];
      this.lfSelectedEmbedder = '';
      return;
    }
    this.datasetsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.lfEmbedders = embedders;
        this.lfSelectedEmbedder = this.pickInitialEmbedder(embedders, mediaType);
      },
    });
  }

  private lfLoadClippers(mediaType: string): void {
    if (!mediaType) {
      this.lfClippers = [];
      this.lfSelectedClipper = '';
      return;
    }
    this.datasetsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.lfClippers = clippers;
        this.lfSelectedClipper = clippers.length > 0 ? clippers[0].name : '';
        this.lfResetClipperParams();
      },
    });
  }

  lfOnClipperChange(clipperName: string): void {
    this.lfSelectedClipper = clipperName;
    this.lfResetClipperParams();
  }

  private lfResetClipperParams(): void {
    const clipper = this.lfClippers.find((c) => c.name === this.lfSelectedClipper);
    this.lfClipperParams = clipper?.parameters || [];
    this.lfClipperParamValues = {};
    for (const param of this.lfClipperParams) {
      this.lfClipperParamValues[param.key] = param.default;
    }
  }

  /** First selected file's webkitRelativePath top-level segment, for display. */
  get lfFolderName(): string {
    if (this.lfFiles.length === 0) return '';
    const rel = (this.lfFiles[0] as any).webkitRelativePath as string | undefined;
    if (!rel) return '';
    const idx = rel.indexOf('/');
    return idx >= 0 ? rel.slice(0, idx) : rel;
  }

  lfSubmit(): void {
    if (this.lfFiles.length === 0) {
      this.lfError = this.lfPickerKind === 'files'
        ? 'Please select a paths file to upload.'
        : 'Please select a folder to upload.';
      return;
    }

    if (this.lfPickerKind === 'files') {
      this.lfSubmitFiles();
      return;
    }
    this.lfSubmitFolder();
  }

  private lfSubmitFolder(): void {
    // When recursion is disabled in folder mode, drop any files that live
    // inside subdirectories of the picked folder.  ``webkitRelativePath``
    // looks like ``"top/file.wav"`` for top-level entries and
    // ``"top/sub/file.wav"`` for files inside a subdirectory.
    let filesToUpload = this.lfFiles;
    if (!this.lfRecursive) {
      filesToUpload = this.lfFiles.filter((file) => {
        const rel = ((file as any).webkitRelativePath as string | undefined) || '';
        return rel.split('/').length <= 2;
      });
      if (filesToUpload.length === 0) {
        this.lfError = 'No files at the top level of the selected folder. Enable "Include subfolders" to import nested files.';
        return;
      }
    }

    this.lfSubmitting = true;
    this.lfError = '';

    const formData = new FormData();
    formData.append('media_type', this.lfMediaType);
    formData.append('recursive', this.lfRecursive ? 'true' : 'false');
    this.lfAppendCommonFormFields(formData);
    for (const file of filesToUpload) {
      const rel = (file as any).webkitRelativePath as string | undefined;
      // Browsers only populate webkitRelativePath when the input has the
      // `webkitdirectory` attribute; fall back to the file's own name.
      formData.append('files', file, rel && rel.length > 0 ? rel : file.name);
    }
    if (this.lfSourceSpecs.length > 0) {
      // Multipart form fields are flat strings; encode as JSON.
      formData.append('source_specs', JSON.stringify(this.lfSourceSpecs));
    }

    this.datasetsApi.importLocalFolder(formData).subscribe({
      next: () => {
        this.lfSubmitting = false;
        this.importStarted.emit();
      },
      error: (err) => {
        this.lfSubmitting = false;
        this.lfError = err.error?.error || 'Upload failed';
      },
    });
  }

  private lfSubmitFiles(): void {
    this.lfSubmitting = true;
    this.lfError = '';

    const pathsFile = this.lfFiles[0];
    const formData = new FormData();
    formData.append('media_type', this.lfMediaType);
    formData.append('paths_file', pathsFile, pathsFile.name);
    this.lfAppendCommonFormFields(formData);
    if (this.lfSourceSpecs.length > 0) {
      formData.append('source_specs', JSON.stringify(this.lfSourceSpecs));
    }

    this.datasetsApi.importLocalFiles(formData).subscribe({
      next: () => {
        this.lfSubmitting = false;
        this.importStarted.emit();
      },
      error: (err) => {
        this.lfSubmitting = false;
        this.lfError = err.error?.error || 'Upload failed';
      },
    });
  }

  private lfAppendCommonFormFields(formData: FormData): void {
    const lfName = (this.lfDatasetName || '').trim();
    if (lfName) {
      formData.append('dataset_name', lfName);
    }
    if (this.lfSelectedEmbedder) {
      formData.append('embedder', this.lfSelectedEmbedder);
    }
    if (this.lfSelectedClipper) {
      formData.append('clipper', this.lfSelectedClipper);
      if (this.lfClipperParams.length > 0 && Object.keys(this.lfClipperParamValues).length > 0) {
        formData.append('clipper_params', JSON.stringify(this.lfClipperParamValues));
      }
    }
  }

  // --- Server folder browser ---

  openServerFolderBrowser(importer?: ImporterInfo): void {
    this.selectedImporter = importer || this.importers.find((i) => i.name === 'server_folder') || null;
    this.sfBrowsePath = '';
    this.sfBrowseRootPath = '';
    this.sfBrowseError = '';
    this.sfDetection = null;
    this.sfSubmitting = false;
    this.sfRecursive = this.readRecursiveDefault(this.selectedImporter);
    this.sfDatasetName = '';
    this.sfDatasetNameDirty = false;
    this.sfPathInputValue = '';

    // Load media type options from the folder importer's fields
    const folderImporter = this.importers.find((imp) => imp.name === 'server_folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    this.sfMediaTypeOptions = mtField?.options || [];

    // Prefer guessed media type when available
    const guessedFolder = this.toFolderName(this.guessedMediaType);
    if (guessedFolder && this.sfMediaTypeOptions.includes(guessedFolder)) {
      this.sfMediaType = guessedFolder;
    } else {
      this.sfMediaType = mtField?.default || this.sfMediaTypeOptions[0] || 'audio';
    }

    this.sfLoadEmbedders(this.sfMediaType);
    this.sfLoadClippers(this.sfMediaType);
    this.sfResetSourceSpecs();
  }

  /** Current value of the editable absolute-path input. Two-way bound to
   *  the typed path input; ``sfApplyPathInput`` splits it into the
   *  ``sfBrowseRootPath`` / ``sfBrowsePath`` pair the submit + detection
   *  helpers already consume. */
  sfPathInputValue = '';

  /** Apply the value typed into the absolute-path input. The path is not
   *  verified here — the server validates it on submit. */
  sfApplyPathInput(): void {
    const raw = (this.sfPathInputValue || '').trim();
    if (!raw) {
      this.sfBrowsePath = '';
      this.sfBrowseRootPath = '';
      this.sfBrowseError = '';
      this.sfDetection = null;
      if (!this.sfDatasetNameDirty) {
        this.sfDatasetName = '';
      }
      return;
    }
    // Treat the typed value as an absolute server path. Anchor the root
    // at "/" and put the rest into sfBrowsePath so sfAbsolutePath returns
    // the user-typed value verbatim.
    const rel = raw.replace(/^\/+/, '').replace(/\/+$/, '');
    this.sfBrowseRootPath = '/';
    this.sfBrowsePath = rel;
    this.sfBrowseError = '';
    if (!this.sfDatasetNameDirty) {
      this.sfDatasetName = this.sfDerivedDatasetName();
    }
    this.sfRunDetection();
  }

  /** Token guarding overlapping detection responses for the sf-* picker.
   *  Each :meth:`sfRunDetection` invocation bumps this and stamps it onto
   *  its closure; the response handler bails when the token has moved on,
   *  so rapidly clicking through directories never lets an older request
   *  overwrite a newer one. */
  private sfDetectionToken = 0;

  /** Detect the dominant media type for the currently picked server folder
   *  and apply it to the sf-* form (output media-type + source specs). */
  private sfRunDetection(): void {
    const token = ++this.sfDetectionToken;
    this.datasetsApi.detectMediaType('server_fs', this.sfBrowsePath, this.sfRecursive).subscribe({
      next: (res) => {
        if (token !== this.sfDetectionToken) return;
        this.sfDetection = res;
        this.sfApplyDetection();
      },
      error: () => {
        if (token !== this.sfDetectionToken) return;
        this.sfDetection = null;
      },
    });
  }

  /** Apply :prop:`sfDetection` to the sf-* view. */
  private sfApplyDetection(): void {
    if (!this.sfDetection) return;
    const { mediaType, sourceSpecs } = this.autofillFromDetection(
      this.sfDetection,
      this.sfMediaTypeOptions,
      (typeId) => this.availableConvertersFor('server_folder', typeId),
    );
    if (mediaType && mediaType !== this.sfMediaType) {
      this.sfMediaType = mediaType;
      this.sfLoadEmbedders(this.sfMediaType);
      this.sfLoadClippers(this.sfMediaType);
    }
    if (sourceSpecs) {
      this.sfSourceSpecs = sourceSpecs;
    }
  }

  /** Derive a default dataset name from the currently selected server folder.
   *  Returns the leaf path component, or empty string when at the root. */
  private sfDerivedDatasetName(): string {
    if (!this.sfBrowsePath) return '';
    const parts = this.sfBrowsePath.split('/').filter(Boolean);
    return parts.length > 0 ? parts[parts.length - 1] : '';
  }

  sfOnDatasetNameInput(value: string): void {
    this.sfDatasetName = value;
    this.sfDatasetNameDirty = true;
  }

  get sfAbsolutePath(): string {
    if (!this.sfBrowseRootPath) return '';
    if (!this.sfBrowsePath) return this.sfBrowseRootPath;
    // Avoid "//foo" when the picker is rooted at the filesystem root.
    if (this.sfBrowseRootPath === '/') return '/' + this.sfBrowsePath;
    return this.sfBrowseRootPath + '/' + this.sfBrowsePath;
  }

  sfOnMediaTypeChange(mediaType: string): void {
    this.sfMediaType = mediaType;
    this.sfLoadEmbedders(mediaType);
    this.sfLoadClippers(mediaType);
    this.sfResetSourceSpecs();
  }

  // -------------------------------------------------------------------
  // Source-specs editor — shared logic across the three picker views.
  //
  // Each view ("sf" / "lf" / "form") owns its own SourceSpec[] and its
  // own "output media-type" form value.  The helpers below take that
  // state in as arguments so the same edit logic works for all three.
  // -------------------------------------------------------------------

  /** Map a folder_import_name (e.g. "images") to a type_id (e.g. "image"). */
  private toTypeId(folderName: string): string {
    if (!folderName) return '';
    const mt = this.mediaTypes.find((m) => m.folder_import_name === folderName);
    return mt?.type_id || folderName;
  }

  // -------------------------------------------------------------------
  // Media-type auto-detection
  //
  // Two flavours: the server-folder picker hits an API endpoint that
  // walks the chosen path on the server filesystem; the local-folder /
  // local-files picker counts extensions on the in-browser File[] (no
  // HTTP round-trip needed because the files haven't been uploaded yet).
  //
  // Once a detection result is available we (a) pre-fill the output
  // media-type dropdown with the dominant type and (b) auto-populate
  // multi-media SourceSpec rows for every non-dominant type present in
  // the sample that has a converter to the dominant type — so a folder
  // of "47 images + 3 videos" opens with "include images directly" plus
  // a "video → image" converter row pre-added.
  // -------------------------------------------------------------------

  /** Lowercase ``ext → type_id`` map derived from registered media types.
   *  ``.jpg → "image"``.  Rebuilt on each call (cheap; one Map per type). */
  private extensionToTypeId(): Map<string, string> {
    const map = new Map<string, string>();
    for (const mt of this.mediaTypes) {
      for (const pattern of mt.file_extensions || []) {
        const dot = pattern.lastIndexOf('.');
        if (dot < 0) continue;
        map.set(pattern.slice(dot).toLowerCase(), mt.type_id);
      }
    }
    return map;
  }

  /** Count media types in a browser-side ``File[]`` and shape the result
   *  like :type:`MediaTypeDetectionResponse` so the rest of the modal
   *  can treat local- and server-side detections identically.
   *
   *  When ``recursive`` is ``false`` files whose ``webkitRelativePath``
   *  lies in a sub-directory of the picked folder are skipped, matching
   *  the importer's "Include subfolders" toggle. */
  private detectFromFiles(files: File[], recursive: boolean, limit = 50): MediaTypeDetectionResponse {
    const extMap = this.extensionToTypeId();
    const countsByType: Record<string, number> = {};
    const extensions: Record<string, number> = {};
    let examined = 0;
    for (const file of files) {
      if (examined >= limit) break;
      const rel = ((file as any).webkitRelativePath as string | undefined) || '';
      if (!recursive && rel && rel.split('/').length > 2) continue;
      const name = rel || file.name || '';
      const slash = name.lastIndexOf('/');
      const base = slash >= 0 ? name.slice(slash + 1) : name;
      if (base.startsWith('.')) continue;
      const dot = base.lastIndexOf('.');
      const ext = dot > 0 ? base.slice(dot).toLowerCase() : '';
      extensions[ext] = (extensions[ext] || 0) + 1;
      const typeId = (ext && extMap.get(ext)) || 'unknown';
      countsByType[typeId] = (countsByType[typeId] || 0) + 1;
      examined += 1;
    }
    let dominant: string | null = null;
    let bestCount = 0;
    for (const [typeId, count] of Object.entries(countsByType)) {
      if (typeId === 'unknown') continue;
      if (count > bestCount) {
        bestCount = count;
        dominant = typeId;
      }
    }
    return { sample_size: examined, counts_by_type: countsByType, extensions, dominant };
  }

  /** Human-readable description of a detection result, suitable for a
   *  hint chip next to the dropdown. */
  detectionHint(detection: MediaTypeDetectionResponse | null): string {
    if (!detection || detection.sample_size === 0) return '';
    const entries = Object.entries(detection.counts_by_type)
      .filter(([typeId]) => typeId !== 'unknown')
      .sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) {
      return `No recognised media files in ${detection.sample_size} sampled.`;
    }
    const total = detection.sample_size;
    const fileWord = total === 1 ? 'file' : 'files';
    if (entries.length === 1) {
      const [typeId, count] = entries[0];
      return `Detected: ${this.getTabLabel(typeId)} (${count} of ${total} ${fileWord})`;
    }
    const head = entries
      .map(([typeId, count]) => `${this.getTabLabel(typeId)} (${count})`)
      .join(' + ');
    return `Detected: ${head} of ${total} ${fileWord}`;
  }

  /** Apply a detection result to a (mediaType, sourceSpecs) pair.
   *
   *  Sets ``mediaType`` to the dominant type's ``folder_import_name`` when
   *  it's a valid option, then rebuilds the source-spec list: one direct
   *  row for the dominant type plus one converter row per non-dominant
   *  recognised type that has at least one matching converter to the
   *  dominant type.  Returns the new ``(mediaType, sourceSpecs)`` pair —
   *  the caller decides which view's state to update.
   */
  private autofillFromDetection(
    detection: MediaTypeDetectionResponse,
    availableOptions: string[],
    convertersForType: (outputTypeId: string) => ConverterInfo[],
  ): { mediaType: string | null; sourceSpecs: SourceSpec[] | null } {
    const dominant = detection.dominant;
    if (!dominant) return { mediaType: null, sourceSpecs: null };
    const folderName = this.mediaTypes.find((m) => m.type_id === dominant)?.folder_import_name || dominant;
    if (!availableOptions.includes(folderName)) {
      return { mediaType: null, sourceSpecs: null };
    }
    const converters = convertersForType(dominant);
    const sourceSpecs: SourceSpec[] = [
      { source_type: dominant, converter: null, params: {} },
    ];
    const seenSourceTypes = new Set<string>([dominant]);
    const orderedNonDominant = Object.entries(detection.counts_by_type)
      .filter(([typeId, count]) => typeId !== 'unknown' && typeId !== dominant && count > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([typeId]) => typeId);
    for (const sourceType of orderedNonDominant) {
      if (seenSourceTypes.has(sourceType)) continue;
      const converter = converters.find((c) => c.source_type === sourceType);
      if (!converter) continue;
      const params: Record<string, string> = {};
      for (const f of converter.fields || []) {
        params[f.key] = String(f.default ?? '');
      }
      sourceSpecs.push({ source_type: sourceType, converter: converter.name, params });
      seenSourceTypes.add(sourceType);
    }
    return { mediaType: folderName, sourceSpecs };
  }

  /** Converters whose ``target_type`` matches *outputTypeId*.  The map
   *  comes from the importer's ``to_dict()`` so each importer can
   *  declare its own filtered list. */
  availableConvertersFor(importerName: string, outputTypeId: string): ConverterInfo[] {
    const importer = this.importers.find((i) => i.name === importerName);
    const byType = (importer?.available_converters_by_media_type as Record<string, ConverterInfo[]> | undefined) || {};
    return byType[outputTypeId] || [];
  }

  /** Build the "default" spec list for a freshly-opened picker: one
   *  "include directly" row whose source matches the output type. */
  private defaultSpecListFor(outputTypeId: string): SourceSpec[] {
    return outputTypeId
      ? [{ source_type: outputTypeId, converter: null, params: {} }]
      : [];
  }

  // Per-view native-type ids and converter lists fed to
  // ``<vt-source-specs-picker>``.  Local uploads stream to the
  // server-side temp dir and re-enter ``server_folder.run()`` — so
  // ``lf`` uses the same converter list as ``sf``.

  get sfOutputTypeId(): string { return this.toTypeId(this.sfMediaType); }
  get sfAvailableConverters(): ConverterInfo[] {
    return this.availableConvertersFor('server_folder', this.sfOutputTypeId);
  }
  private sfResetSourceSpecs(): void {
    this.sfSourceSpecs = this.defaultSpecListFor(this.sfOutputTypeId);
  }
  onSfSpecsChange(specs: SourceSpec[]): void { this.sfSourceSpecs = specs; }

  get lfOutputTypeId(): string { return this.toTypeId(this.lfMediaType); }
  get lfAvailableConverters(): ConverterInfo[] {
    return this.availableConvertersFor('server_folder', this.lfOutputTypeId);
  }
  private lfResetSourceSpecs(): void {
    this.lfSourceSpecs = this.defaultSpecListFor(this.lfOutputTypeId);
  }
  onLfSpecsChange(specs: SourceSpec[]): void { this.lfSourceSpecs = specs; }

  get formOutputTypeId(): string {
    return this.toTypeId(String(this.formValues['media_type'] || ''));
  }
  get formAvailableConverters(): ConverterInfo[] {
    if (!this.selectedImporter) return [];
    return this.availableConvertersFor(this.selectedImporter.name, this.formOutputTypeId);
  }
  resetFormSourceSpecs(): void {
    this.formSourceSpecs = this.defaultSpecListFor(this.formOutputTypeId);
  }
  onFormSpecsChange(specs: SourceSpec[]): void { this.formSourceSpecs = specs; }

  private sfLoadEmbedders(mediaType: string): void {
    if (!mediaType) {
      this.sfEmbedders = [];
      this.sfSelectedEmbedder = '';
      return;
    }
    this.datasetsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.sfEmbedders = embedders;
        this.sfSelectedEmbedder = this.pickInitialEmbedder(embedders, mediaType);
      },
    });
  }

  private sfLoadClippers(mediaType: string): void {
    if (!mediaType) {
      this.sfClippers = [];
      this.sfSelectedClipper = '';
      return;
    }
    this.datasetsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.sfClippers = clippers;
        this.sfSelectedClipper = clippers.length > 0 ? clippers[0].name : '';
        this.sfResetClipperParams();
      },
    });
  }

  sfOnClipperChange(clipperName: string): void {
    this.sfSelectedClipper = clipperName;
    this.sfResetClipperParams();
  }

  private sfResetClipperParams(): void {
    const clipper = this.sfClippers.find((c) => c.name === this.sfSelectedClipper);
    this.sfClipperParams = clipper?.parameters || [];
    this.sfClipperParamValues = {};
    for (const param of this.sfClipperParams) {
      this.sfClipperParamValues[param.key] = param.default;
    }
  }

  // --- Clipper chooser ---

  openClipperChooser(context: 'form' | 'demo' | 'sf' | 'lf'): void {
    this.clipperChooserContext = context;
    if (context === 'form') {
      this.clipperChooserClippers = this.availableClippers;
    } else if (context === 'demo') {
      this.clipperChooserClippers = this.demoClippers;
    } else if (context === 'sf') {
      this.clipperChooserClippers = this.sfClippers;
    } else {
      this.clipperChooserClippers = this.lfClippers;
    }
    this.clipperChooserOpen = true;
  }

  onClipperChooserSelected(selection: ClipperSelection): void {
    this.clipperChooserOpen = false;
    const ctx = this.clipperChooserContext;
    if (ctx === 'form') {
      this.selectedClipper = selection.name;
      this.clipperParamValues = { ...selection.params };
    } else if (ctx === 'demo') {
      this.selectedDemoClipper = selection.name;
      this.updateDemoStatuses();
      this.refetchDemoStatuses(this.selectedDemoEmbedder, this.selectedDemoClipper);
    } else if (ctx === 'sf') {
      this.sfSelectedClipper = selection.name;
      this.sfClipperParamValues = { ...selection.params };
    } else {
      this.lfSelectedClipper = selection.name;
      this.lfClipperParamValues = { ...selection.params };
    }
  }

  onClipperChooserCancelled(): void {
    this.clipperChooserOpen = false;
    // Cancel returns the default clipper for the media type
    const ctx = this.clipperChooserContext;
    const clippers = this.clipperChooserClippers;
    const defaultClipper = clippers.find((c) => c.name.endsWith('_default')) || clippers[0];
    const defaultName = defaultClipper?.name || '';
    if (ctx === 'form') {
      this.selectedClipper = defaultName;
      this.resetClipperParams();
    } else if (ctx === 'demo') {
      this.selectedDemoClipper = defaultName;
      this.updateDemoStatuses();
      this.refetchDemoStatuses(this.selectedDemoEmbedder, this.selectedDemoClipper);
    } else if (ctx === 'sf') {
      this.sfSelectedClipper = defaultName;
      this.sfResetClipperParams();
    } else {
      this.lfSelectedClipper = defaultName;
      this.lfResetClipperParams();
    }
  }

  sfSubmit(): void {
    this.sfSubmitting = true;
    this.sfBrowseError = '';

    const params: Record<string, unknown> = {
      path: this.sfAbsolutePath,
      media_type: this.sfMediaType,
      recursive: this.sfRecursive,
    };
    const sfName = (this.sfDatasetName || '').trim();
    if (sfName) {
      params['dataset_name'] = sfName;
    }
    if (this.sfSelectedEmbedder) {
      params['embedder'] = this.sfSelectedEmbedder;
    }
    if (this.sfSelectedClipper) {
      params['clipper'] = this.sfSelectedClipper;
      if (this.sfClipperParams.length > 0 && Object.keys(this.sfClipperParamValues).length > 0) {
        params['clipper_params'] = { ...this.sfClipperParamValues };
      }
    }
    if (this.sfSourceSpecs.length > 0) {
      params['source_specs'] = this.sfSourceSpecs;
    }

    this.datasetsApi.runImporter('server_folder', params).subscribe({
      next: () => {
        this.sfSubmitting = false;
        this.importStarted.emit();
      },
      error: (err) => {
        this.sfSubmitting = false;
        this.sfBrowseError = err.error?.error || 'Import failed';
      },
    });
  }

  onFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.formValues[fieldName] = input.files[0].name;
      this.maybeApplyDerivedDatasetName();
    }
  }

  submit(): void {
    if (!this.selectedImporter) return;
    this.submitting = true;
    this.error = '';

    // Include clipper and embedder in form values if selected
    const submitValues = { ...this.formValues };
    if (this.selectedClipper) {
      submitValues['clipper'] = this.selectedClipper;
      if (this.selectedClipperParams.length > 0 && Object.keys(this.clipperParamValues).length > 0) {
        submitValues['clipper_params'] = { ...this.clipperParamValues };
      }
    }
    if (this.selectedEmbedder) {
      submitValues['embedder'] = this.selectedEmbedder;
    }
    if (this.selectedImporter.multi_media && this.formSourceSpecs.length > 0) {
      submitValues['source_specs'] = this.formSourceSpecs;
    }

    // If there's a file field, use loadFile; otherwise runImporter
    const fileField = this.selectedImporter.fields?.find((f) => f.field_type === 'file');
    if (fileField && this.selectedFile) {
      this.datasetsApi.loadFile(this.selectedFile).subscribe({
        next: () => {
          this.submitting = false;
          this.importStarted.emit();
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Import failed';
        },
      });
    } else {
      this.datasetsApi.runImporter(this.selectedImporter.name, submitValues).subscribe({
        next: () => {
          this.submitting = false;
          this.importStarted.emit();
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Import failed';
        },
      });
    }
  }

  close(): void {
    this.closed.emit();
  }
}
