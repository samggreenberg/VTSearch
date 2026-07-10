import { ChangeDetectionStrategy, Component, HostListener, inject, Input, input, OnInit, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { DetectorsRegistryApiService } from '../../../services/detectors-registry-api.service';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { DatasetsUiApiService } from '../../../services/datasets-ui-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import {
  EmbedderCapabilityService,
  EMBEDDER_TYPE_LABELS,
  EMBEDDER_TYPE_ORDER,
  type EmbedderType,
} from '../../../services/embedder-capability.service';
import { MediaStateService } from '../../../services/media-state.service';
import {
  ImporterField,
  ImporterInfo,
  ImporterPickerTab,
  Media,
  MediaTypeInfo,
} from '../../../models/api.models';
import type { LabelImporterEntry } from '../../../generated/api-client/models/label-importer-entry';
import { DemoDatasetEntry } from '../../../generated/api-client/models/demo-dataset-entry';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../../modals/media-crop-modal/media-crop-modal.component';
import { DropZoneComponent } from '../../drop-zone/drop-zone.component';
import { SourcePickerComponent } from '../dataset-importer-modal/source-picker/source-picker.component';
import { ColMeta, ManagedColumns } from '../../../utils/managed-columns';
import { apiErrorMessage } from '../../../utils/api-error';

type ModalView = 'main' | 'media-picker';
type ModalTab = 'blank' | 'trained';
type TrainedSubView = 'picker' | 'form';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-new-detector-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, MediaCropModalComponent, DropZoneComponent, SourcePickerComponent, FieldHintIconComponent],
  templateUrl: './new-detector-modal.component.html',
  styleUrl: './new-detector-modal.component.scss',
})
export class NewDetectorModalComponent implements OnInit {
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
  private sortingApi = inject(SortingApiService);
  private labelImportersApi = inject(LabelImportersApiService);
  private settingsState = inject(SettingsStateService);
  private embedderCaps = inject(EmbedderCapabilityService);
  private mediaState = inject(MediaStateService);

  /** Media type of the currently active dataset, if any. */
  @Input() defaultMediaType = '';

  /** Embedder of the active dataset, if one is in context. When it can't
   *  search by text, a text-only detector won't be able to start in Autopilot
   *  or use Text sort on that dataset; the form surfaces a warning. Empty when
   *  unknown, which suppresses the warning. */
  @Input() datasetEmbedder = '';

  /** When set, the modal opens with this loaded-media id materialised into
   *  example_media/ as the seed example. The picker is bypassed and the
   *  user lands directly on the form. Cleared by callers via
   *  ``NewThingFlowsService.closeNewDetector``. */
  @Input() seedMediaId: number | null = null;

  /** Optional crop bounds applied when materialising ``seedMediaId``. */
  readonly seedCropParams = input<Record<string, unknown> | null>(null);

  readonly closed = output<void>();
  readonly created = output<string>();

  readonly view = signal<ModalView>('main');
  tab: ModalTab = 'blank';
  readonly name = signal('');
  /** True once the user has typed into the name field. While false, the
   *  name auto-tracks ``pendingText`` (sanitised) so users don't have to
   *  type the same string twice. */
  nameTouched = false;
  readonly mediaType = signal('audio');
  readonly pendingText = signal('');
  /** Which example kind the user is filling in. Text and media examples are
   *  mutually exclusive, so the blank-detector form shows one at a time behind
   *  a tab; this tracks the active tab. Defaults to text (the quick path). */
  readonly exampleTab = signal<'text' | 'media'>('text');
  readonly mediaTypes = signal<string[]>([]);
  readonly mediaTypeInfos = signal<MediaTypeInfo[]>([]);
  readonly submitting = signal(false);
  readonly error = signal('');

  /** The user's *explicit* embedder-type pick (the detector's locked scoring
   *  space *kind*). Empty means "not chosen": the displayed value falls back to
   *  the active dataset's default type via {@link effectiveEmbedderType}, and an
   *  untouched single-/no-dataset create defers to the server's auto-resolution
   *  (see {@link submittedEmbedderType}). Set only when the user changes the
   *  Advanced picker. */
  readonly embedderType = signal<EmbedderType | ''>('');
  /** Whether the collapsible "Advanced" section (which holds the embedder-type
   *  picker) is expanded. Collapsed by default — the common single-embedder
   *  create needs no interaction; advanced users open it to lock a type. */
  readonly advancedOpen = signal(false);
  mediaTypeDropdownOpen = false;
  /** True when the media-type field is locked to the active dataset's type.
   *  Set on init whenever `defaultMediaType` is provided; cleared when the
   *  user clicks the unlock button. The dropdown trigger is disabled while
   *  locked so the user can't accidentally change it. */
  mediaTypeLocked = false;

  // Single example (text or media, not both)
  readonly exampleType = signal<'text' | 'media' | null>(null);
  readonly exampleValue = signal('');
  readonly exampleDisplay = signal('');
  exampleMediaType = '';
  readonly exampleThumbFailed = signal(false);

  // --- Media picker state (shares structure with the Add Dataset modal) ---

  /** All importers discovered from the backend, filtered to picker_views we
   *  can use to select a single example file (demo, server_folder,
   *  local_folder, local_files). */
  readonly mediaImporters = signal<ImporterInfo[]>([]);
  /** Tab declarations (categories) returned by the backend. */
  readonly declaredImporterTabs = signal<ImporterPickerTab[]>([]);
  /** Currently selected category tab (e.g. ``"demo"``, ``"server"``). */
  activeImporterTab = '';
  /** Currently selected importer within the active category. */
  selectedImporter: ImporterInfo | null = null;

  /** Front-of-list order for sub-importer tabs.  Mirrors the Add Dataset
   *  modal's ordering so users see the same layout in both places. */
  private static readonly PICKER_ORDER = [
    'local_folder',
    'local_files',
    'server_folder',
    'server_files',
    'demo',
  ];

  /** Picker views supported by the single-file example picker.  Importers
   *  whose ``picker_view`` is anything else (e.g. ``"form"``) are hidden
   *  here; the user can still use them via the Add Dataset modal. */
  private static readonly SUPPORTED_PICKER_VIEWS = new Set([
    'demo',
    'server_folder',
    'local_folder',
    'local_files',
  ]);

  // --- Demo picker state ---
  readonly demos = signal<DemoDatasetEntry[]>([]);
  readonly demoTabs = signal<string[]>([]);
  activeDemoTab = '';
  readonly demoLoading = signal(false);

  /** Demo table column metadata + controller.  Reuses the same storage key
   *  as the Add Dataset modal so column order and sort preferences stay in
   *  sync between the two demo tables. */
  static readonly DEMO_COL_META: Record<string, ColMeta> = {
    label: { label: 'Name', title: 'Demo dataset name (click to sort)', sortable: true },
    num_files: { label: '# Media', title: 'Number of media files in the demo dataset (click to sort)', sortable: true },
    description: { label: 'Description', title: 'Short description of the demo dataset contents (click to sort)', sortable: true },
    status: { label: 'Readiness', title: 'Whether the dataset is pre-downloaded and ready to browse, or still needs to be fetched (click to sort)', sortable: true },
  };
  static readonly DEMO_COLUMNS_DEFAULT = ['label', 'num_files', 'description', 'status'];
  private static readonly DEMO_COL_ORDER_KEY = 'vtsearch.dashboard.demoColumnOrder';

  demoCols = new ManagedColumns(
    NewDetectorModalComponent.DEMO_COLUMNS_DEFAULT,
    NewDetectorModalComponent.DEMO_COL_META,
    { initialSort: 'num_files', storageKey: NewDetectorModalComponent.DEMO_COL_ORDER_KEY },
  );

  // --- Demo example-media picker (shown after picking a demo from the table) ---
  demoFileBrowsing = false;
  demoFileBrowseSource = '';
  demoFileBrowseLabel = '';
  readonly demoFileLoading = signal(false);
  demoTypedPath = '';
  readonly demoTypedPathError = signal('');

  // --- Server example-media picker. Path is validated when submitted. ---
  readonly sfBrowseError = signal('');
  readonly sfFileSelecting = signal(false);
  sfTypedPath = '';

  // Pending crop confirmation state.
  pendingFile: File | null = null;
  pendingFileMediaType = '';

  // "Trained" tab state
  trainedView: TrainedSubView = 'picker';
  readonly labelImporters = signal<LabelImporterEntry[]>([]);
  readonly labelImportersLoading = signal(false);
  selectedLabelImporter: LabelImporterEntry | null = null;
  labelImporterValues: Record<string, string> = {};
  labelImporterFile: File | null = null;
  labelImporterFileFieldKey: string | null = null;

  /** Type_id of the active solo-mediaType streamlining, or ``null`` when
   *  off. When non-null, the mediaType form-group is hidden in the
   *  template and ``mediaType`` is locked to this value on init. */
  get effectiveSoloMediaType(): string | null {
    const v = this.settingsState.settingsSignal()?.effective_solo_media_type;
    return v ? v : null;
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    if (this.mediaTypeDropdownOpen && !target.closest('.custom-select')) {
      this.mediaTypeDropdownOpen = false;
    }
  }

  @HostListener('document:mousemove', ['$event'])
  onDocResizeMove(event: MouseEvent): void {
    this.demoCols.onResizeMove(event);
  }

  @HostListener('document:mouseup')
  onDocResizeEnd(): void {
    this.demoCols.onResizeEnd();
  }

  ngOnInit(): void {
    this.embedderCaps.ensureLoaded();
    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypeInfos.set(res.media_types || []);
        this.mediaTypes.set(this.mediaTypeInfos().map((t) => t.type_id || t.name));
      },
    });
    // Settings power the solo-mediaType lockdown; load them so the
    // template's @if guards see the resolved value on first render.
    this.settingsState.load();

    // Solo-mediaType mode forces the field to the chosen type and the
    // template hides the picker entirely (no unlock button rendered).
    const solo = this.effectiveSoloMediaType;
    if (solo) {
      this.mediaType.set(solo);
      this.mediaTypeLocked = true;
    } else if (this.defaultMediaType) {
      // Prefer the explicit default (active dataset's type) over the all-datasets guess.
      // When the active dataset dictates the type, lock the field so the user
      // can't change it without an explicit unlock click.
      this.mediaType.set(this.defaultMediaType);
      this.mediaTypeLocked = true;
    } else {
      this.datasetsRegistryApi.getRegistry().subscribe({
        next: (res) => {
          const types = new Set(
            (res.datasets || []).map((d) => d['media_type'] as string).filter(Boolean),
          );
          if (types.size === 1) {
            this.mediaType.set([...types][0]);
          }
        },
      });
    }

    if (this.seedMediaId != null) {
      this.materializeSeedFromMediaId(this.seedMediaId, this.seedCropParams() ?? undefined);
    }
  }

  /** Materialise a loaded media into example_media/ and pre-fill the
   *  example fields, so the user lands on the form with the seed already
   *  selected. */
  private materializeSeedFromMediaId(
    mediaId: number,
    cropParams?: Record<string, unknown>,
  ): void {
    this.submitting.set(true);
    this.sortingApi
      .saveServerMediaFromMediaId({ media_id: mediaId, crop_params: cropParams })
      .subscribe({
        next: (res) => {
          this.exampleType.set('media');
          this.exampleValue.set(res.filename);
          this.exampleDisplay.set(res.original_name || res.filename);
          this.exampleMediaType = this.mediaType();
          this.exampleThumbFailed.set(false);
          this.pendingText.set('');
          this.exampleTab.set('media');
          this.autoFillNameFromExample();
          this.submitting.set(false);
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(err.error?.message || 'Failed to load seed media');
        },
      });
  }

  unlockMediaType(): void {
    this.mediaTypeLocked = false;
  }

  /** Trim and collapse internal whitespace so a pasted multi-line query
   *  becomes a single-line name. */
  private sanitizeName(text: string): string {
    return text.trim().replace(/\s+/g, ' ');
  }

  /** Strip a trailing extension and the leading path so a filename like
   *  ``/foo/bar/My Sound.wav`` becomes ``My Sound``. */
  private nameFromFilename(text: string): string {
    const base = text.split(/[\\/]/).pop() || text;
    const dot = base.lastIndexOf('.');
    return dot > 0 ? base.slice(0, dot) : base;
  }

  /** Auto-derive the detector name from the picked example while the user
   *  hasn't typed into the name field. Lets the user fill the form
   *  top-down and leave Name blank. */
  private autoFillNameFromExample(): void {
    if (this.nameTouched) return;
    if (this.exampleType() === 'media' && this.exampleDisplay()) {
      this.name.set(this.sanitizeName(this.nameFromFilename(this.exampleDisplay())));
    } else if (this.pendingText()) {
      this.name.set(this.sanitizeName(this.pendingText()));
    }
  }

  onPendingTextInput(value: string): void {
    this.pendingText.set(value);
    // Text and media examples are mutually exclusive — typing a text example
    // drops any previously selected media example so the two never coexist.
    if (this.hasMediaExample) {
      this.clearMediaExample();
    }
    if (!this.nameTouched) {
      this.name.set(this.sanitizeName(value));
    }
  }

  onNameInput(value: string): void {
    this.nameTouched = true;
    this.name.set(value);
  }

  toggleMediaTypeDropdown(): void {
    if (this.mediaTypeLocked) return;
    this.mediaTypeDropdownOpen = !this.mediaTypeDropdownOpen;
  }

  get modalTitle(): string {
    if (this.view() === 'media-picker') return 'Select Media Example';
    return 'New Detector';
  }

  get hasExample(): boolean {
    return this.exampleType() === 'media' || !!this.pendingText().trim();
  }

  /**
   * True when the active dataset's embedder can't search by text and the user
   * is creating a text-hint-only detector (text entered, no media example).
   * Such a detector still works — but only after labeling enough to train it —
   * so we warn that Autopilot and Text sort won't be available up front.
   */
  get showNoTextWarning(): boolean {
    return (
      !this.embedderCaps.supportsText(this.datasetEmbedder) &&
      this.hasPendingText &&
      !this.hasMediaExample
    );
  }

  get hasMediaExample(): boolean {
    return this.exampleType() === 'media';
  }

  get hasPendingText(): boolean {
    return !!this.pendingText().trim();
  }

  /** Embedder names the active dataset has vectors for (the keys of each
   *  media's ``embeddings`` dict, surfaced as the ``embedders`` array). These
   *  are the detector's eligible primary spaces. Falls back to the single
   *  ``datasetEmbedder`` when the medias haven't loaded an array. */
  get boundEmbedderNames(): string[] {
    const medias = this.mediaState.mediasSignal() as Media[];
    const first = medias.length > 0 ? medias[0] : null;
    const names = first?.embedders ?? (first?.embedder ? [first.embedder] : []);
    if (names.length > 0) return names;
    return this.datasetEmbedder ? [this.datasetEmbedder] : [];
  }

  /** All three embedder types, in display order — the always-available options
   *  in the Advanced picker. A detector locks a type as *declared intent*, so
   *  the choice isn't constrained to what the active dataset (if any) binds. */
  get embedderTypeOptions(): EmbedderType[] {
    return EMBEDDER_TYPE_ORDER;
  }

  /** The embedder *types* the active dataset supplies, or `[]` when no dataset
   *  is loaded. Drives the default pick and the "not on this dataset" hint. */
  get datasetSuppliedTypes(): EmbedderType[] {
    return this.embedderCaps.suppliedTypes(this.boundEmbedderNames);
  }

  /** The type shown in the picker: the user's explicit pick, else the active
   *  dataset's primary supplied type, else `semantic`. Computed lazily so it
   *  reflects the dataset/registry as they load (no pre-load race). */
  get effectiveEmbedderType(): EmbedderType {
    return this.embedderType() || this.datasetSuppliedTypes[0] || 'semantic';
  }

  embedderTypeLabel(type: EmbedderType | ''): string {
    return type ? EMBEDDER_TYPE_LABELS[type] : '';
  }

  /** True when the selected type isn't one the active dataset supplies (and a
   *  dataset is loaded) — the detector is valid but won't run here until used
   *  on a dataset that binds this type. Drives an inline heads-up. */
  get embedderTypeUnavailable(): boolean {
    return this.boundEmbedderNames.length > 0 && !this.datasetSuppliedTypes.includes(this.effectiveEmbedderType);
  }

  /** License notice of the concrete embedder backing the selected type, or
   *  null. The user picks a type; the server resolves the concrete embedder,
   *  but we surface its licence up front so the warning isn't a surprise. */
  get primaryLicenseNotice(): string | null {
    const concrete = this.embedderCaps.firstOfType(this.boundEmbedderNames, this.effectiveEmbedderType);
    if (!concrete) return null;
    return (this.embedderCaps.infos() ?? []).find((e) => e.name === concrete)?.license_notice ?? null;
  }

  toggleAdvanced(): void {
    this.advancedOpen.update((open) => !open);
  }

  onEmbedderTypeChange(type: EmbedderType | ''): void {
    this.embedderType.set(type);
  }

  /** The ``embedder_type`` to send on create. An explicit pick always wins.
   *  When untouched, defer to the server's auto-resolution by sending empty
   *  (single-embedder dataset → its sole type; no dataset → empty/first-train),
   *  except on a multi-type dataset, where the server can't guess, so send the
   *  picker's defaulted (valid, supplied) type. */
  private submittedEmbedderType(): string {
    if (this.embedderType()) return this.embedderType();
    return this.datasetSuppliedTypes.length > 1 ? this.effectiveEmbedderType : '';
  }

  get canSubmitBlank(): boolean {
    return !!this.name().trim() && this.hasExample && !this.submitting();
  }

  get canSubmitTrained(): boolean {
    return (
      !!this.name().trim() &&
      !!this.selectedLabelImporter &&
      this.trainedView === 'form' &&
      !this.submitting()
    );
  }

  // --- Tab switching ---

  setTab(tab: ModalTab): void {
    if (this.submitting()) return;
    this.tab = tab;
    this.error.set('');
  }

  /** Switch between the Text and media example tabs in the blank-detector form. */
  setExampleTab(tab: 'text' | 'media'): void {
    if (this.submitting()) return;
    this.exampleTab.set(tab);
    this.error.set('');
  }

  /** Label for the media example tab: "Image", "Audio", "Video", etc. (the
   *  detector's media type), falling back to "Media". */
  get exampleMediaTabLabel(): string {
    const mediaType = this.mediaType();
    return (mediaType ? this.getMediaTypeLabel(mediaType) : '') || 'Media';
  }

  // --- Media picker (shared structure with Add Dataset) ---

  openMediaPicker(): void {
    this.view.set('media-picker');
    this.activeImporterTab = '';
    this.selectedImporter = null;
    this.resetDemoPickerState();
    this.resetServerFolderState();
    this.loadMediaImporters();
  }

  private loadMediaImporters(): void {
    this.datasetsCrudApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaImporters.set((res.importers || []).filter(
          (imp) =>
            !imp['hidden_from_picker'] &&
            NewDetectorModalComponent.SUPPORTED_PICKER_VIEWS.has(imp.picker_view || ''),
        ));
        this.declaredImporterTabs.set(res.tabs || []);
      },
    });
  }

  /** Importers ordered like the Add Dataset modal: known picker_views
   *  first, then any extras in registry order. */
  get orderedImporters(): ImporterInfo[] {
    const order = NewDetectorModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      const imp = this.mediaImporters().find((i) => i.name === name);
      if (imp) result.push(imp);
    }
    for (const imp of this.mediaImporters()) {
      if (!order.includes(imp.name)) result.push(imp);
    }
    return result;
  }

  private fallbackTabLabel(id: string): string {
    return id
      .split(/[\s_-]+/)
      .filter(Boolean)
      .map((part) => part[0].toUpperCase() + part.slice(1))
      .join(' ');
  }

  /** Visible category tabs.  Only categories that contain at least one
   *  supported importer are shown; empty categories (e.g. ``"services"``
   *  on a vanilla install) are hidden so the picker stays focused on
   *  options the user can act on. */
  get visibleImporterTabs(): ImporterPickerTab[] {
    const usedCategories = new Set(
      this.orderedImporters.map((imp) => imp.category || '').filter(Boolean),
    );
    const visible: ImporterPickerTab[] = [];
    const seen = new Set<string>();
    const declared = [...this.declaredImporterTabs()].sort(
      (a, b) => (a.order ?? 100) - (b.order ?? 100),
    );
    for (const tab of declared) {
      if (usedCategories.has(tab.id)) {
        visible.push(tab);
        seen.add(tab.id);
      }
    }
    for (const id of usedCategories) {
      if (!seen.has(id)) {
        visible.push({ id, label: this.fallbackTabLabel(id) });
      }
    }
    return visible;
  }

  get importersForActiveTab(): ImporterInfo[] {
    return this.orderedImporters.filter(
      (imp) => (imp.category || '') === this.activeImporterTab,
    );
  }

  get activeImporterTabLabel(): string {
    const tab = this.visibleImporterTabs.find((t) => t.id === this.activeImporterTab);
    return tab?.label || '';
  }

  selectImporterTab(tabId: string): void {
    this.activeImporterTab = tabId;
    this.selectedImporter = null;
    this.resetDemoPickerState();
    this.resetServerFolderState();
    // A category with a single source has no meaningful sub-tab choice, so
    // skip straight to that source's input (e.g. opening "Files" lands the
    // user on the server-path entry instead of a one-button sub-tab bar).
    const importers = this.importersForActiveTab;
    if (importers.length === 1) {
      this.selectImporter(importers[0]);
    }
  }

  selectImporter(importer: ImporterInfo): void {
    this.selectedImporter = importer;
    this.error.set('');
    const view = importer.picker_view || '';
    if (view === 'demo') {
      this.openDemoPicker();
    } else if (view === 'server_folder') {
      this.openServerFolderBrowser();
    } else {
      // local_folder / local_files: just reveal the upload input below.
      this.resetDemoPickerState();
      this.resetServerFolderState();
    }
  }

  /** Picker view of the currently selected importer, or empty when nothing
   *  is selected.  Drives which inline widget set is rendered below the
   *  inner sub-tab row. */
  get activePickerView(): string {
    return this.selectedImporter?.picker_view || '';
  }

  // --- Demo picker ---

  private resetDemoPickerState(): void {
    this.demos.set([]);
    this.demoTabs.set([]);
    this.activeDemoTab = '';
    this.demoLoading.set(false);
    this.demoFileBrowsing = false;
    this.demoFileBrowseSource = '';
    this.demoFileBrowseLabel = '';
    this.demoFileLoading.set(false);
  }

  private openDemoPicker(): void {
    this.resetDemoPickerState();
    this.demoLoading.set(true);
    this.datasetsListingsApi.getDemoList().subscribe({
      next: (res) => {
        this.demos.set(res.datasets || []);
        this.buildDemoTabs();
        this.demoLoading.set(false);
      },
      error: () => {
        this.demoLoading.set(false);
      },
    });
  }

  private buildDemoTabs(): void {
    const grouped = new Set(this.demos().map((d) => d.media_type));
    const registryOrder = this.mediaTypeInfos().map((mt) => mt.type_id);
    const tabs = registryOrder.filter((mt) => grouped.has(mt));
    for (const mt of grouped) {
      if (!tabs.includes(mt)) tabs.push(mt);
    }
    this.demoTabs.set(tabs);
  }

  selectDemoTab(tab: string): void {
    this.activeDemoTab = tab;
  }

  get filteredDemos(): DemoDatasetEntry[] {
    const items = this.demos().filter((d) => d.media_type === this.activeDemoTab);
    const statusOrder: Record<string, number> = { ready: 0, needs_embedding: 1, needs_download: 2 };
    const sortKey = this.demoCols.sortColumn;
    const asc = this.demoCols.sortAsc;
    return [...items].sort((a, b) => {
      const key = sortKey as keyof DemoDatasetEntry;
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

  /** True when the demo's files are on disk and can be browsed.  Demos
   *  that still need to be downloaded show a tooltip explaining how to
   *  fetch them via the Add Dataset modal. */
  isDemoBrowsable(demo: DemoDatasetEntry): boolean {
    return demo.status === 'ready' || demo.status === 'needs_embedding';
  }

  /** Arrow-bound predicate handed to ``<vt-source-picker>`` so its demo
   *  table can apply ``.disabled`` styling to non-browsable rows.  Kept
   *  as a class field (rather than a getter) so the function reference
   *  is stable across change-detection cycles. */
  demoRowDisabledFn = (demo: DemoDatasetEntry): boolean => !this.isDemoBrowsable(demo);

  /** Arrow-bound formatter for the ``title`` attribute on demo rows. */
  demoRowTitleFn = (demo: DemoDatasetEntry): string =>
    this.isDemoBrowsable(demo)
      ? `Browse files in ${demo.label}`
      : 'This demo has not been downloaded. Use the Add Dataset window to fetch it first.';

  /** Map ``activePickerView`` to the ``lfPickerKind`` flag understood by
   *  ``<vt-source-picker>``. */
  get lfPickerKind(): 'folder' | 'files' {
    return this.activePickerView === 'local_files' ? 'files' : 'folder';
  }

  selectDemo(demo: DemoDatasetEntry): void {
    if (!this.isDemoBrowsable(demo)) return;
    this.demoFileBrowsing = true;
    this.demoFileBrowseSource = `demo:${demo.name}`;
    this.demoFileBrowseLabel = demo.label;
    this.demoTypedPath = '';
    this.demoTypedPathError.set('');
  }

  /** Submit the typed demo-relative path. Server validates and returns
   *  the materialised filename for the example sort. */
  submitDemoTypedPath(): void {
    const raw = (this.demoTypedPath || '').trim();
    if (!raw) return;
    this.demoFileLoading.set(true);
    this.demoTypedPathError.set('');
    this.datasetsUiApi.selectBrowsedFile(this.demoFileBrowseSource, raw).subscribe({
      next: (res) => {
        this.exampleType.set('media');
        this.exampleValue.set(res.filename);
        this.exampleDisplay.set(res.original_name || raw);
        this.exampleMediaType = this.activeDemoTab || this.mediaType();
        this.exampleThumbFailed.set(false);
        this.pendingText.set('');
        this.exampleTab.set('media');
        this.autoFillNameFromExample();
        this.demoFileLoading.set(false);
        this.view.set('main');
      },
      error: (err) => {
        this.demoTypedPathError.set(err?.error?.message || 'Path not found in this demo.');
        this.demoFileLoading.set(false);
      },
    });
  }

  /** Return to the demo table from the demo example-media picker. */
  backToDemoTable(): void {
    this.demoFileBrowsing = false;
    this.demoFileBrowseSource = '';
    this.demoFileBrowseLabel = '';
    this.demoTypedPath = '';
    this.demoTypedPathError.set('');
  }

  // --- Server example-media (typed path) ---

  private resetServerFolderState(): void {
    this.sfBrowseError.set('');
    this.sfFileSelecting.set(false);
    this.sfTypedPath = '';
  }

  private openServerFolderBrowser(): void {
    this.resetServerFolderState();
  }

  /** Submit the typed absolute server path. Server validates and returns
   *  the materialised filename for the example sort. */
  submitSfTypedPath(): void {
    const raw = (this.sfTypedPath || '').trim();
    if (!raw) return;
    this.sfFileSelecting.set(true);
    this.sfBrowseError.set('');
    this.datasetsUiApi.selectBrowsedFile('server_fs', raw).subscribe({
      next: (res) => {
        this.exampleType.set('media');
        this.exampleValue.set(res.filename);
        this.exampleDisplay.set(res.original_name || raw);
        this.exampleMediaType = this.mediaType() || this.mediaTypeFromFilename(raw);
        this.exampleThumbFailed.set(false);
        this.pendingText.set('');
        this.exampleTab.set('media');
        this.autoFillNameFromExample();
        this.sfFileSelecting.set(false);
        this.view.set('main');
      },
      error: (err) => {
        this.sfBrowseError.set(err?.error?.message || 'Path not found on the server.');
        this.sfFileSelecting.set(false);
      },
    });
  }

  // --- Local file upload (single file for the example) ---

  formatSize(bytes?: number): string {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /** Local file picker handler.  Used by the drop-zone affordance in both
   *  the main form (next to "Browse Media…") and the Local Folder / Local
   *  Files cards in the media picker.  Multi-file drops (e.g. a folder)
   *  collapse to the first file since only one example is needed. */
  onLocalFileDropped(files: File[]): void {
    if (files.length === 0) return;
    const file = files[0];
    this.pendingFile = file;
    this.pendingFileMediaType = this.mediaType() || this.mediaTypeFromFile(file);
  }

  onCropConfirmed(result: MediaCropResult): void {
    const file = result.file;
    const cropParams = result.cropParams;
    const mediaType = this.pendingFileMediaType;
    this.pendingFile = null;
    this.sortingApi
      .uploadServerMediaFile(file, cropParams ? { mediaType, cropParams } : undefined)
      .subscribe({
        next: (res) => {
          this.exampleType.set('media');
          this.exampleValue.set(res.filename);
          this.exampleDisplay.set(res.original_name || res.filename);
          this.exampleMediaType = mediaType || this.mediaType();
          this.exampleThumbFailed.set(false);
          this.pendingText.set('');
          this.exampleTab.set('media');
          this.autoFillNameFromExample();
          // Close the picker if it was open so the user lands back on the form.
          if (this.view() === 'media-picker') this.view.set('main');
        },
        error: () => {
          this.error.set('Failed to upload file');
        },
      });
  }

  onCropCancelled(): void {
    this.pendingFile = null;
  }

  private mediaTypeFromFile(file: File): string {
    const m = (file.type || '').toLowerCase();
    if (m.startsWith('image/')) return 'image';
    if (m.startsWith('audio/')) return 'audio';
    if (m.startsWith('video/')) return 'video';
    return '';
  }

  private mediaTypeFromFilename(name: string): string {
    const ext = name.toLowerCase().split('.').pop() || '';
    if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'].includes(ext)) return 'image';
    if (['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac'].includes(ext)) return 'audio';
    if (['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(ext)) return 'video';
    return '';
  }

  /** URL of the example thumbnail, or null if no media example is selected
   *  or the thumbnail endpoint already failed (so we fall back to the icon). */
  get exampleThumbnailUrl(): string | null {
    if (this.exampleType() !== 'media' || !this.exampleValue() || this.exampleThumbFailed()) {
      return null;
    }
    return `/api/server-media-files/${encodeURIComponent(this.exampleValue())}/thumbnail`;
  }

  onExampleThumbError(): void {
    this.exampleThumbFailed.set(true);
  }

  backToMain(): void {
    this.view.set('main');
  }

  // --- Clear example ---

  clearExample(): void {
    this.clearMediaExample();
    this.pendingText.set('');
  }

  /** Reset only the media-example fields, leaving any pending text untouched.
   *  Used both by the "Change" button and when the user starts typing a text
   *  example (the two kinds are mutually exclusive). */
  private clearMediaExample(): void {
    this.exampleType.set(null);
    this.exampleValue.set('');
    this.exampleDisplay.set('');
    this.exampleMediaType = '';
    this.exampleThumbFailed.set(false);
  }

  // --- Trained tab: label importers ---

  private ensureLabelImportersLoaded(): void {
    if (this.labelImporters().length > 0 || this.labelImportersLoading()) return;
    this.labelImportersLoading.set(true);
    this.labelImportersApi.list().subscribe({
      next: (list) => {
        this.labelImporters.set(list.filter((imp) => !imp.hidden_from_picker));
        this.labelImportersLoading.set(false);
      },
      error: () => {
        this.labelImportersLoading.set(false);
        this.error.set('Failed to load label importers');
      },
    });
  }

  onSelectTrainedTab(): void {
    this.setTab('trained');
    this.ensureLabelImportersLoaded();
  }

  /** Typed view of the selected label importer's plugin fields for the
   *  template (the generated LabelImporterEntry types `fields` as an open
   *  dict because plugin field schemas aren't part of the OpenAPI client). */
  get selectedLabelImporterFields(): ImporterField[] {
    return (this.selectedLabelImporter?.fields ?? []) as ImporterField[];
  }

  selectLabelImporter(importer: LabelImporterEntry): void {
    this.selectedLabelImporter = importer;
    this.labelImporterValues = {};
    this.labelImporterFile = null;
    this.labelImporterFileFieldKey = null;
    this.error.set('');
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.labelImporterValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        (field.options?.length ?? 0) > 0
      ) {
        this.labelImporterValues[field.key] = field.options![0];
      }
    }
    this.trainedView = 'form';
  }

  backToImporterPicker(): void {
    this.trainedView = 'picker';
    this.selectedLabelImporter = null;
    this.error.set('');
  }

  onLabelImporterFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.labelImporterFile = input.files[0];
      this.labelImporterFileFieldKey = fieldName;
      this.labelImporterValues[fieldName] = input.files[0].name;
    }
  }

  submitTrained(): void {
    const trimmedName = this.name().trim();
    if (!trimmedName) {
      this.error.set('Name is required');
      return;
    }
    if (!this.selectedLabelImporter) {
      this.error.set('A label importer is required');
      return;
    }

    this.submitting.set(true);
    this.error.set('');

    const params: Record<string, unknown> = {
      name: trimmedName,
      embedder_type: this.submittedEmbedderType(),
      ...this.labelImporterValues,
    };

    this.detectorsRegistryApi
      .registerDetectorFromLabelset(
        this.selectedLabelImporter.name,
        params,
        this.labelImporterFile ?? undefined,
        this.labelImporterFileFieldKey ?? undefined,
      )
      .subscribe({
        next: (resp: any) => {
          const newId = resp?.detector?.id || '';
          if (!newId) {
            this.submitting.set(false);
            this.error.set('Server did not return a detector id');
            return;
          }
          this.detectorsRegistryApi.loadDetector(newId).subscribe({
            next: () => {
              this.submitting.set(false);
              this.created.emit(newId);
            },
            error: () => {
              // Model exists in registry even if load failed; still emit.
              this.submitting.set(false);
              this.created.emit(newId);
            },
          });
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Failed to create detector from labelset'));
        },
      });
  }

  // --- Submit ---

  submit(): void {
    if (this.tab === 'trained') {
      this.submitTrained();
      return;
    }

    const trimmedName = this.name().trim();
    if (!trimmedName) {
      this.error.set('Name is required');
      return;
    }

    // Accept pending text as the text example on submit
    const pendingTrimmed = this.pendingText().trim();
    if (this.exampleType() !== 'media' && pendingTrimmed) {
      this.exampleType.set('text');
      this.exampleValue.set(pendingTrimmed);
      this.exampleDisplay.set(pendingTrimmed);
    }

    if (!this.exampleType()) {
      this.error.set('An example (text or media) is required');
      return;
    }

    this.submitting.set(true);
    this.error.set('');

    const exampleType = this.exampleType();
    const exampleValue = this.exampleValue();
    const textQuery = exampleType === 'text' ? exampleValue : '';
    const mediaExample = exampleType === 'media' ? exampleValue : '';
    const examplesPayload = [{ type: exampleType!, value: exampleValue }];

    this.detectorsRegistryApi
      .registerDetector({
        name: trimmedName,
        media_type: this.mediaType(),
        text_query: textQuery,
        media_example: mediaExample,
        examples: examplesPayload,
        embedder_type: this.submittedEmbedderType(),
      })
      .subscribe({
        next: (resp: any) => {
          this.submitting.set(false);
          this.created.emit(resp?.detector?.id || '');
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Failed to create detector'));
        },
      });
  }

  getMediaTypeLabel(typeId: string): string {
    const mt = this.mediaTypeInfos().find((m) => m.type_id === typeId);
    if (mt) {
      return mt.name.trim();
    }
    return typeId;
  }

  /** "Browse Images...", "Browse Audio...", "Browse Media..." as a fallback. */
  get browseMediaLabel(): string {
    const mediaType = this.mediaType();
    const name = mediaType ? this.getMediaTypeLabel(mediaType) : '';
    if (!name) return 'Browse Media...';
    // audio and text don't take a plural -s in this context.
    const uncountable = mediaType === 'audio' || mediaType === 'text';
    return `Browse ${uncountable ? name : name + 's'}...`;
  }

  /** "Drop an image file here", "Drop a video file here", etc. */
  get dropMediaLabel(): string {
    const mediaType = this.mediaType();
    const name = mediaType ? this.getMediaTypeLabel(mediaType) : '';
    if (!name) return 'Drop a media file here';
    const lower = name.toLowerCase();
    const article = /^[aeiou]/.test(lower) ? 'an' : 'a';
    return `Drop ${article} ${lower} file here`;
  }

  getMediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypeInfos().find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  close(): void {
    this.closed.emit();
  }
}
