import { ChangeDetectionStrategy, Component, OnInit, inject, input, output, signal, viewChild } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { SourcePickerComponent } from './source-picker/source-picker.component';
import { ImportConfigComponent } from './import-config/import-config.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { GenericFormPickerComponent } from './pickers/generic-form/generic-form-picker.component';
import { ServerFolderPickerComponent } from './pickers/server-folder/server-folder-picker.component';
import { LocalFolderPickerComponent } from './pickers/local-folder/local-folder-picker.component';
import { DemoPickerComponent } from './pickers/demo/demo-picker.component';
import { ImportDefaultsService } from './pickers/shared/import-defaults.service';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { EmbedderInfo, ImporterInfo, ImporterPickerTab, MediaTypeInfo } from '../../../models/api.models';
import { DemoDatasetEntry } from '../../../generated/api-client/models/demo-dataset-entry';
import { toTypeId } from './pickers/shared/media-type.util';

/** Add Dataset modal.  Owns the importer registry, category/importer
 *  tab selection, and the shared media-type registry + Advanced-block
 *  toggles (``buildProjection`` / ``mergeNearDuplicates``) that persist
 *  across flows; delegates everything else to four self-contained
 *  "picker view" components (one per ``picker_view``: the generic form,
 *  server-folder, local-folder/files, and demo flows), each owning its
 *  own state, HTTP calls, and submit logic.
 *
 *  The four pickers stay mounted for the lifetime of the modal (rather
 *  than being created/destroyed via `@if`) for two reasons: (1) the
 *  shared `<vt-source-picker>` widget - which renders the demo table /
 *  server-folder path input / local dropzone chrome - is a single
 *  instance owned by *this* template, and needs live bindings onto
 *  whichever picker is active; keeping every picker instantiated lets
 *  template reference variables (`#demoPicker` etc.) resolve regardless
 *  of which view is showing. (2) it lets `selectImporter` reach each
 *  picker's `open()` method via `viewChild()` queries, with no
 *  create-on-demand timing to reason about. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-importer-modal',
  standalone: true,
  imports: [
    FormsModule,
    ModalComponent,
    SourcePickerComponent,
    ImportConfigComponent,
    FieldHintIconComponent,
    GenericFormPickerComponent,
    ServerFolderPickerComponent,
    LocalFolderPickerComponent,
    DemoPickerComponent,
  ],
  templateUrl: './dataset-importer-modal.component.html',
  styleUrl: './dataset-importer-modal.component.scss',
})
export class DatasetImporterModalComponent implements OnInit {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private settingsState = inject(SettingsStateService);
  private importDefaults = inject(ImportDefaultsService);

  /** Media type_id guessed from existing datasets/models (e.g. "image"). */
  readonly guessedMediaType = input('');
  /** Embedder name guessed from existing datasets/in-progress loads (e.g. "siglip"). */
  readonly guessedMediaEmbedder = input('');
  /** Picker tab id to pre-select when the modal opens (e.g. "server" from
   *  the dashboard's first-run welcome banner CTA).  Empty leaves the
   *  picker in the default "no tab selected" state. */
  readonly initialTab = input('');

  readonly closed = output<void>();
  readonly importStarted = output<void>();
  readonly demoSelected = output<DemoDatasetEntry>();

  readonly importers = signal<ImporterInfo[]>([]);
  readonly selectedImporter = signal<ImporterInfo | null>(null);

  /** "Build the 2-D Browse projection at ingest" toggle, shared across
   *  every import flow (only one is visible at a time, so a single field
   *  carries the choice and persists it when the user switches flows).
   *  Defaults off per the opt-in cost tradeoff. */
  buildProjection = false;
  /** "Merge near-duplicates" toggle, shared across every import flow like
   *  ``buildProjection``. */
  mergeNearDuplicates = false;

  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  /** Bare (media_type-agnostic) embedder list, fetched once.  Kept for
   *  HTTP-call parity with the pre-refactor init sequence; no current
   *  consumer reads the result (every picker fetches its own
   *  media-type-scoped list on demand). */
  allEmbedders: EmbedderInfo[] = [];

  // Every picker tag is unconditionally present in the template
  // (visibility is toggled with `[hidden]` / the shared
  // `<vt-source-picker>`'s own mutually-exclusive `@if` branches, never
  // `@if` on the picker tag itself), so each query always has a match by
  // the first change-detection pass.  `selectImporter` only dispatches to
  // these from the async importer-list HTTP callback and from user
  // events - both after the view (and hence these queries) has resolved -
  // so `.required()` never reads before a value is available.
  readonly genericFormPicker = viewChild.required(GenericFormPickerComponent);
  readonly serverFolderPicker = viewChild.required(ServerFolderPickerComponent);
  readonly localFolderPicker = viewChild.required(LocalFolderPickerComponent);
  readonly demoPicker = viewChild.required(DemoPickerComponent);

  get effectiveSoloMediaType(): string | null {
    return this.importDefaults.effectiveSoloMediaType;
  }

  ngOnInit(): void {
    this.datasetsCrudApi.getAllImporters().subscribe({
      next: (res) => {
        this.importers.set((res.importers || []).filter((imp) => !imp['hidden_from_picker']));
        this.declaredTabs.set(res.tabs || []);
        if (this.initialTab() && this.visibleImporterTabs.some((t) => t.id === this.initialTab())) {
          this.selectImporterTab(this.initialTab());
        } else if (this.visibleImporterTabs.length) {
          // Land on the first category that actually has importers (falling
          // back to the first tab) instead of a blank pane. The New Detector
          // modal pre-selects its first tab, and an unselected two-level tab
          // bar leaves the whole modal body empty.
          const tabs = this.visibleImporterTabs;
          const first =
            tabs.find((t) => this.orderedImporters.some((imp) => (imp.category || '') === t.id)) ?? tabs[0];
          this.selectImporterTab(first.id);
        }
      },
    });
    this.datasetsListingsApi.getEmbedders().subscribe({
      next: (embedders) => {
        this.allEmbedders = embedders || [];
      },
    });
    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes.set(res.media_types || []);
      },
    });
    // Settings carry the per-media-type "last embedder" memory + solo-mode
    // locks the pickers read via ImportDefaultsService.
    this.settingsState.load();
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
  readonly declaredTabs = signal<ImporterPickerTab[]>([]);

  /** Currently selected picker tab.  Defaults to the first visible tab as
   *  soon as the importer list arrives (``ngOnInit``); empty string only
   *  before that, so the content area is never left blank once loaded. */
  readonly activeImporterTab = signal('');

  get orderedImporters(): ImporterInfo[] {
    const order = DatasetImporterModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      const imp = this.importers().find((i) => i.name === name);
      if (imp && !imp['hidden_from_picker']) result.push(imp);
    }
    for (const imp of this.importers()) {
      if (!order.includes(imp.name) && !imp['hidden_from_picker']) {
        result.push(imp);
      }
    }
    return result.filter((imp) => this.importerAllowedUnderSolo(imp));
  }

  /** The set of media ``type_id``s an importer can produce, or ``null``
   *  when it is media-type agnostic (has no fixed ``media_type`` option
   *  list - e.g. the folder/file importers, whose type the user picks, or
   *  the demo importer, which filters its own table).  Options are
   *  normalised through the media-type registry so an importer that
   *  declares folder names (``"images"``) and one that declares type_ids
   *  (``"image"``) both resolve to the canonical id. */
  private supportedTypeIds(importer: ImporterInfo): Set<string> | null {
    const field = importer.fields?.find((f) => f.key === 'media_type');
    const options = field?.options || [];
    if (!field || options.length === 0) return null;
    const types = new Set<string>();
    for (const opt of options) types.add(toTypeId(this.mediaTypes(), opt));
    return types;
  }

  /** Whether an importer is offered under the active solo media type.
   *  With no solo lock every importer is offered; otherwise agnostic
   *  importers stay (they preselect the solo type) and type-scoped ones
   *  are hidden unless their option set includes the locked type, so the
   *  picker never presents an importer that can't produce the one type
   *  the user is streamlined to. */
  private importerAllowedUnderSolo(importer: ImporterInfo): boolean {
    const solo = this.effectiveSoloMediaType;
    if (!solo) return true;
    const supported = this.supportedTypeIds(importer);
    return supported === null || supported.has(solo);
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
   *  them - so categories like "Services" remain visible even when no
   *  extension importers are installed.  Categories used by importers but
   *  never declared get appended at the end with a title-cased label and
   *  no icon. */
  get visibleImporterTabs(): ImporterPickerTab[] {
    const visible: ImporterPickerTab[] = [];
    const seen = new Set<string>();
    const declared = [...this.declaredTabs()].sort(
      (a, b) => (a.order ?? 100) - (b.order ?? 100),
    );
    const usedCategories = new Set(
      this.orderedImporters.map((imp) => imp.category || '').filter(Boolean),
    );
    const solo = this.effectiveSoloMediaType;
    for (const tab of declared) {
      // Declared tabs normally render even when empty (so categories like
      // "Services" stay visible before any extension importer is installed).
      // Under a solo lock that would surface a category with no importer
      // able to produce the locked type, which is exactly the confusion this
      // streamlining removes - so drop empty tabs while solo is active.
      if (solo && !usedCategories.has(tab.id)) continue;
      visible.push(tab);
      seen.add(tab.id);
    }
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
      (imp) => (imp.category || '') === this.activeImporterTab(),
    );
  }

  /** Display label of the currently selected category tab, or empty when none. */
  get activeImporterTabLabel(): string {
    const tab = this.visibleImporterTabs.find((t) => t.id === this.activeImporterTab());
    return tab?.label || '';
  }

  selectImporterTab(tabId: string): void {
    this.activeImporterTab.set(tabId);
    this.selectedImporter.set(null);
    // When the tab has exactly one importer, the inner sub-tab row is
    // redundant; clicking the outer tab already declared the intent.
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
    return this.selectedImporter()?.picker_view || '';
  }

  /** Dispatch importer selection to the picker view matching its
   *  declared ``picker_view``. */
  selectImporter(importer: ImporterInfo): void {
    this.selectedImporter.set(importer);
    const pickerView = importer.picker_view || 'form';
    if (pickerView === 'local_folder' || pickerView === 'local_files') {
      this.localFolderPicker().open(importer);
    } else if (pickerView === 'server_folder') {
      this.serverFolderPicker().open(importer);
    } else if (pickerView === 'demo') {
      this.demoPicker().open(importer);
    } else {
      this.genericFormPicker().open(importer);
    }
  }

  openDemoPicker(importer?: ImporterInfo): void {
    const resolved = importer || this.importers().find((i) => i.name === 'demo') || null;
    this.selectedImporter.set(resolved);
    this.demoPicker().open(resolved);
  }

  openLocalFolderUploader(importer?: ImporterInfo): void {
    const resolved = importer || this.importers().find((i) => i.name === 'local_folder') || null;
    this.selectedImporter.set(resolved);
    this.localFolderPicker().open(resolved);
  }

  openServerFolderBrowser(importer?: ImporterInfo): void {
    const resolved = importer || this.importers().find((i) => i.name === 'server_folder') || null;
    this.selectedImporter.set(resolved);
    this.serverFolderPicker().open(resolved);
  }

  /** Forward a demo picker's committed selection to the dashboard, then
   *  close the modal. */
  onDemoDatasetSelected(demo: DemoDatasetEntry): void {
    this.demoSelected.emit(demo);
    this.close();
  }

  close(): void {
    this.closed.emit();
  }
}
