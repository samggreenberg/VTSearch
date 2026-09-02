import { ChangeDetectionStrategy, Component, HostListener, inject, input, OnInit, output, signal } from '@angular/core';

import { NgTemplateOutlet } from '@angular/common';

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
import { ProgressEventsService } from '../../../services/progress-events.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import {
  EmbedderCapabilityService,
  EMBEDDER_TYPE_LABELS,
  type EmbedderType,
} from '../../../services/embedder-capability.service';
import { MediaStateService } from '../../../services/media-state.service';
import {
  ImporterField,
  ImporterInfo,
  ImporterPickerTab,
  LoadingTask,
  Media,
  MediaTypeInfo,
} from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { formatProgressMessage, progressBarState, type ProgressBarState } from '../../../utils/format-progress';
import type { LabelImporterEntry } from '../../../generated/api-client/models/label-importer-entry';
import { DemoDatasetEntry } from '../../../generated/api-client/models/demo-dataset-entry';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../../modals/media-crop-modal/media-crop-modal.component';
import { DropZoneComponent } from '../../drop-zone/drop-zone.component';
import { SourcePickerComponent } from '../dataset-importer-modal/source-picker/source-picker.component';
import { PluginImportFormComponent } from '../../plugin-import-form/plugin-import-form.component';
import {
  DatasourceImportersApiService,
  DatasourceImportResult,
} from '../../../services/datasource-importers-api.service';
import {
  SeedImportersApiService,
  SeedImportResult,
} from '../../../services/seed-importers-api.service';
import { ColMeta, ManagedColumns } from '../../../utils/managed-columns';
import { apiErrorMessage } from '../../../utils/api-error';
import { DynamicFieldOptions } from '../../../utils/dynamic-field-options';
import { sortRowsByColumn } from '../../../utils/sort-rows';
import { demoSortValue } from '../dataset-importer-modal/pickers/shared/demo-sort';

type ModalView = 'main' | 'media-picker';
type ModalTab = 'blank' | 'trained';
type TrainedSubView = 'picker' | 'form';

/** Which example kind the Blank form is filling in: the stock text query,
 *  the stock media stack, or one registered seed importer (keyed by its
 *  plugin name). Text and media examples are mutually exclusive, and a
 *  seed importer contributes into the media stack, so this only tracks
 *  which panel is on screen. */
type ExampleTab = 'text' | 'media' | (string & {});

/** One picked media example in the blank-detector form's vertical stack. */
interface MediaExampleItem {
  /** Server-side filename in example_media/ (the persistence key). */
  value: string;
  /** Human-readable name shown next to the thumbnail. */
  display: string;
  /** Media type used when the example was picked (drives the thumbnail). */
  mediaType: string;
  /** True once the thumbnail endpoint failed; falls back to the icon. */
  thumbFailed: boolean;
  /** True for an unlabeled seed contributed by a seed importer: "close but
   *  not quite", so it steers the first sort without being submitted as a
   *  Good vote. Sent as ``labeled: false`` on the example. Absent/false for
   *  an exemplar the user picked by hand, which *is* a Good vote. */
  seed?: boolean;
  /** Durable origin dict reported by a datasource importer (URL, server
   *  path); sent with the example so the seeded media points back at its
   *  real source. Absent for uploads and other non-re-derivable picks. */
  origin?: Record<string, unknown> | null;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-new-detector-modal',
  standalone: true,
  imports: [NgTemplateOutlet, FormsModule, ModalComponent, IconComponent, MediaCropModalComponent, DropZoneComponent, SourcePickerComponent, PluginImportFormComponent, FieldHintIconComponent, ProgressBarComponent],
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
  private datasourceImportersApi = inject(DatasourceImportersApiService);
  /** Public: bound straight into ``<vt-plugin-import-form [api]>`` so a seed
   *  importer's fields render through the same generic form the datasource
   *  importers use. */
  readonly seedImportersApi = inject(SeedImportersApiService);
  private labelImportersApi = inject(LabelImportersApiService);
  private progressEvents = inject(ProgressEventsService);
  private settingsState = inject(SettingsStateService);
  private embedderCaps = inject(EmbedderCapabilityService);
  private mediaState = inject(MediaStateService);

  /** Media type of the currently active dataset, if any. */
  readonly defaultMediaType = input('');

  /** Embedder of the active dataset, if one is in context. When it can't
   *  search by text, a text-only detector won't be able to start in Autopilot
   *  or use Text sort on that dataset; the form surfaces a warning. Empty when
   *  unknown, which suppresses the warning. */
  readonly datasetEmbedder = input('');

  /** When set, the modal opens with this loaded-media id materialised into
   *  example_media/ as the seed example. The picker is bypassed and the
   *  user lands directly on the form. Cleared by callers via
   *  ``NewThingFlowsService.closeNewDetector``. */
  readonly seedMediaId = input<number | null>(null);

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
  readonly exampleTab = signal<ExampleTab>('text');
  /** Seed importers registered on this server, one extra tab apiece beside
   *  Text and the media tab. Empty on a vanilla install (the family ships no
   *  built-ins), so the tab bar looks exactly as it did before. */
  readonly seedImporters = signal<ImporterInfo[]>([]);
  readonly mediaTypes = signal<string[]>([]);
  readonly mediaTypeInfos = signal<MediaTypeInfo[]>([]);
  readonly submitting = signal(false);
  readonly error = signal('');

  /** Live snapshot of the background task that pulls the imported labels'
   *  media into the active dataset, while Create & Import waits it out.
   *  ``null`` when no ingest is running. */
  readonly ingestTask = signal<LoadingTask | null>(null);

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

  // Media examples (a vertical stack; mutually exclusive with the text
  // example). Each entry has its own Remove button; the Add button below
  // the stack reopens the picker to append another.
  readonly mediaExamples = signal<MediaExampleItem[]>([]);

  // --- Media picker state (shares structure with the Add Dataset modal) ---

  /** Dataset importers discovered from the backend, filtered to picker_views
   *  we can use to browse for a single example file (demo, local_folder,
   *  local_files). */
  readonly mediaImporters = signal<ImporterInfo[]>([]);
  /** Datasource importers (single-item fetchers: server file, URL,
   *  third-party services).  Each renders as a dynamic form built from its
   *  declared plugin fields, like the Add Dataset generic importer form. */
  readonly datasourceImporters = signal<ImporterInfo[]>([]);
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
    'server_file',
    'demo',
    'url_download',
  ];

  /** Dataset-importer picker views supported by the single-file example
   *  picker (dedicated browse/upload widgets).  Server files and every
   *  other source go through datasource importers instead, which render
   *  as dynamic forms. */
  private static readonly SUPPORTED_PICKER_VIEWS = new Set([
    'demo',
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
  /** Option lists for the Trained tab's label-importer form, mirroring
   *  label-importer-modal so plugins with dynamic_options / depends_on /
   *  allow_free_text render with full parity here too. */
  readonly labelImporterFieldOptions = new DynamicFieldOptions((key, values) =>
    this.labelImportersApi.getFieldOptions(this.selectedLabelImporter!.name, key, values),
  );

  /** Type_id of the server's solo-mediaType restriction, or ``null`` when
   *  off. When non-null, the mediaType form-group is hidden in the
   *  template and ``mediaType`` is locked to this value on init. */
  get effectiveSoloMediaType(): string | null {
    const v = this.settingsState.settingsSignal()?.solo_media_type;
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
    // Seed importers are a plugin family with no in-tree members, so this
    // usually resolves to an empty list and adds no tabs. A failure is
    // silent for the same reason: the stock Text / media tabs still work.
    this.seedImportersApi.list().subscribe({
      next: (res) => this.seedImporters.set((res.importers || []).filter((imp) => !imp['hidden_from_picker'])),
    });
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
    } else if (this.defaultMediaType()) {
      // Prefer the explicit default (active dataset's type) over the all-datasets guess.
      // When the active dataset dictates the type, lock the field so the user
      // can't change it without an explicit unlock click.
      this.mediaType.set(this.defaultMediaType());
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

    if (this.seedMediaId() != null) {
      this.materializeSeedFromMediaId(this.seedMediaId()!, this.seedCropParams() ?? undefined);
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
          this.addMediaExample(res.filename, res.original_name || res.filename, this.mediaType());
          this.submitting.set(false);
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(err.error?.message || 'Failed to load seed media');
        },
      });
  }

  /** Append a picked media example to the stack. Clears any pending text
   *  (text and media examples are mutually exclusive) and auto-fills the name
   *  from the first example.
   *
   *  ``seed`` marks an unlabeled seed contributed by a seed importer, which
   *  rides in the same stack but is submitted with ``labeled: false``.
   *
   *  A hand-picked example lands the user on the media tab, where the picker
   *  they just used lives. A seed does not (issue #3192): the seed-importer
   *  tab renders the same stack, so the batch appears where it was added
   *  instead of throwing the user onto the media tab mid-import. */
  private addMediaExample(
    value: string,
    display: string,
    mediaType: string,
    origin?: Record<string, unknown> | null,
    seed = false,
  ): void {
    this.mediaExamples.update((list) => [
      ...list,
      { value, display, mediaType, thumbFailed: false, origin: origin ?? null, seed },
    ]);
    this.pendingText.set('');
    if (!seed) this.exampleTab.set('media');
    this.autoFillNameFromExample();
  }

  /** Remove one media example from the stack. */
  removeMediaExample(index: number): void {
    this.mediaExamples.update((list) => list.filter((_, i) => i !== index));
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
   *  top-down and leave Name blank.
   *
   *  Seeds are skipped: a seed is "close but not quite" by construction, so
   *  naming the detector after one would name it after the wrong thing. A
   *  seeds-only stack leaves Name for the user to fill. */
  private autoFillNameFromExample(): void {
    if (this.nameTouched) return;
    const first = this.mediaExamples().find((ex) => !ex.seed);
    if (first?.display) {
      this.name.set(this.sanitizeName(this.nameFromFilename(first.display)));
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
      this.seedNotice.set('');
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
    return this.hasMediaExample || !!this.pendingText().trim();
  }

  /**
   * True when the active dataset's embedder can't search by text and the user
   * is creating a text-hint-only detector (text entered, no media example).
   * Such a detector still works — but only after labeling enough to train it —
   * so we warn that Autopilot and Text sort won't be available up front.
   */
  get showNoTextWarning(): boolean {
    return (
      !this.embedderCaps.supportsText(this.datasetEmbedder()) &&
      this.hasPendingText &&
      !this.hasMediaExample
    );
  }

  get hasMediaExample(): boolean {
    return this.mediaExamples().length > 0;
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
    return this.datasetEmbedder() ? [this.datasetEmbedder()] : [];
  }

  /** The embedder types offered in the Advanced picker — all three in display
   *  order normally, Semantic alone on a `semantic_only` server. A detector
   *  locks a type as *declared intent*, so the choice isn't constrained to what
   *  the active dataset (if any) binds. */
  get embedderTypeOptions(): EmbedderType[] {
    return this.embedderCaps.offeredTypes;
  }

  /** Whether to render the type picker at all. A one-option picker is a
   *  question with no answer, so a `semantic_only` server drops it (and the
   *  "not on this dataset" hint under it) rather than showing a dead select. */
  get showEmbedderTypePicker(): boolean {
    return this.embedderTypeOptions.length > 1;
  }

  /** Whether the Advanced toggle is worth showing. Normally yes (it hosts the
   *  type picker); on a `semantic_only` server only when there is a license
   *  notice left to surface, so the block never opens onto nothing. */
  get showAdvancedToggle(): boolean {
    return this.showEmbedderTypePicker || !!this.primaryLicenseNotice;
  }

  /** The embedder *types* the active dataset supplies, or `[]` when no dataset
   *  is loaded. Drives the default pick and the "not on this dataset" hint. */
  get datasetSuppliedTypes(): EmbedderType[] {
    return this.embedderCaps.suppliedTypes(this.boundEmbedderNames);
  }

  /** The type shown in the picker: the user's explicit pick, else the active
   *  dataset's primary supplied type, else `semantic`. Computed lazily so it
   *  reflects the dataset/registry as they load (no pre-load race). A
   *  `semantic_only` server pins it to `semantic`, so a dataset that still
   *  binds a prototype type can't seed a detector the server would reject. */
  get effectiveEmbedderType(): EmbedderType {
    if (this.embedderCaps.semanticOnly()) return 'semantic';
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

  /** Title for the blank-tab Create button. When enabled, describes the
   *  action; when disabled, names the specific blocker (missing example or
   *  missing name) so the user knows what still needs filling in. */
  get blankSubmitTitle(): string {
    if (this.canSubmitBlank) return 'Create the detector with the example you provided';
    if (!this.hasExample) {
      return `Provide a text or ${this.exampleMediaTabLabel.toLowerCase()} example to create the detector`;
    }
    if (!this.name().trim()) return 'Enter a detector name to create the detector';
    return 'Create the detector';
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

  /** Switch between the Text, media, and seed-importer tabs in the
   *  blank-detector form. */
  setExampleTab(tab: ExampleTab): void {
    if (this.submitting()) return;
    this.exampleTab.set(tab);
    this.error.set('');
    this.seedNotice.set('');
  }

  /** The seed importer whose tab is active, or null on the stock tabs. */
  get activeSeedImporter(): ImporterInfo | null {
    const tab = this.exampleTab();
    return this.seedImporters().find((imp) => imp.name === tab) ?? null;
  }

  /** Outcome line shown after a seed run ("Added 12 seeds"), so a run that
   *  matched nothing — or that got truncated at the plugin's cap — says so
   *  instead of looking like it silently did nothing. */
  readonly seedNotice = signal('');

  /** A seed importer returned a batch: append every saved item to the shared
   *  example stack as an unlabeled seed. The stack is mirrored under the seed
   *  importer's own form, so the batch appears in place — the user stays on
   *  this tab and can see (and prune) what arrived. */
  onSeedsImported(result: SeedImportResult): void {
    const items = result?.items ?? [];
    for (const item of items) {
      const display = item.original_name || item.filename;
      this.addMediaExample(
        item.filename,
        display,
        this.mediaType() || this.mediaTypeFromFilename(display),
        item.origin,
        true,
      );
    }
    if (items.length === 0) {
      this.seedNotice.set('That returned no seeds. Try widening the search.');
      return;
    }
    const noun = items.length === 1 ? 'seed' : 'seeds';
    this.seedNotice.set(
      result.truncated
        ? `Added ${items.length} ${noun} (the importer's limit; the rest were dropped).`
        : `Added ${items.length} ${noun}.`,
    );
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
    this.datasourceImportersApi.list().subscribe({
      next: (res) => {
        this.datasourceImporters.set(
          (res.importers || []).filter((imp) => !imp['hidden_from_picker']),
        );
      },
    });
  }

  /** True when *importer* is a datasource importer (single-item fetcher
   *  rendered as a dynamic form) rather than a dataset-importer browse
   *  view.  Identity check against the datasource list, so a name shared
   *  across the two families can't misroute. */
  isDatasourceImporter(importer: ImporterInfo | null): boolean {
    return importer != null && this.datasourceImporters().includes(importer);
  }

  /** Importers ordered like the Add Dataset modal: known names first,
   *  then any extras (e.g. third-party datasource importers) in registry
   *  order. */
  get orderedImporters(): ImporterInfo[] {
    const all = [...this.mediaImporters(), ...this.datasourceImporters()];
    const order = NewDetectorModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      const imp = all.find((i) => i.name === name);
      if (imp) result.push(imp);
    }
    for (const imp of all) {
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
    if (!this.isDatasourceImporter(importer) && (importer.picker_view || '') === 'demo') {
      this.openDemoPicker();
    } else {
      // Datasource importers render their dynamic form below the sub-tab
      // row; local_folder / local_files just reveal the upload input.
      this.resetDemoPickerState();
    }
  }

  /** Picker view of the currently selected importer, or empty when nothing
   *  is selected.  Drives which inline widget set is rendered below the
   *  inner sub-tab row.  Datasource importers report empty so the source
   *  picker renders none of its dedicated widgets; their dynamic form is
   *  rendered by this modal instead. */
  get activePickerView(): string {
    if (this.isDatasourceImporter(this.selectedImporter)) return '';
    return this.selectedImporter?.picker_view || '';
  }

  /** The selected importer when it is a datasource importer, else null.
   *  Template convenience for rendering the dynamic form view. */
  get selectedDatasourceImporter(): ImporterInfo | null {
    return this.isDatasourceImporter(this.selectedImporter) ? this.selectedImporter : null;
  }

  /** A datasource importer fetched an item into ``example_media/``: add
   *  it to the example stack (carrying its durable origin, if any) and
   *  land back on the main form. */
  onDatasourceImported(result: DatasourceImportResult): void {
    const display = result.original_name || result.filename;
    this.addMediaExample(
      result.filename,
      display,
      this.mediaType() || this.mediaTypeFromFilename(display),
      result.origin,
    );
    this.view.set('main');
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
    // Seeding a detector needs an *embeddable* example, so a convert-out half
    // type (e.g. document) has no meaningful example to pick here — filter its
    // tab out even though it has demo datasets in the Add-Dataset flow.
    const infos = this.mediaTypeInfos();
    const embeddable = new Set(infos.filter((mt) => mt.embeddable !== false).map((mt) => mt.type_id));
    const grouped = new Set(this.demos().map((d) => d.media_type).filter((mt) => embeddable.has(mt)));
    const registryOrder = infos.map((mt) => mt.type_id).filter((mt) => embeddable.has(mt));
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
    return sortRowsByColumn(items, this.demoCols.sortColumn, this.demoCols.sortAsc, demoSortValue);
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
    // The path input stays enabled during the request, so Enter can re-fire
    // this and materialise the same example twice (duplicate rows collide the
    // @for track key). The Load button's [disabled] alone isn't enough.
    if (this.demoFileLoading()) return;

    const raw = (this.demoTypedPath || '').trim();
    if (!raw) return;
    this.demoFileLoading.set(true);
    this.demoTypedPathError.set('');
    this.datasetsUiApi.selectBrowsedFile(this.demoFileBrowseSource, raw).subscribe({
      next: (res) => {
        this.addMediaExample(
          res.filename,
          res.original_name || raw,
          this.activeDemoTab || this.mediaType(),
        );
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

  // --- Local file upload (single file for the example) ---

  /** Local file picker handler.  Used by the drop-zone affordance in both
   *  the main form (next to "Browse Media…") and the Local Folder / Local
   *  Files cards in the media picker.  Multi-file drops (e.g. a folder)
   *  collapse to the first file — the crop-confirm step handles one file
   *  at a time; further examples are appended via the "+ Add" button. */
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
          this.addMediaExample(
            res.filename,
            res.original_name || res.filename,
            mediaType || this.mediaType(),
          );
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

  /** URL of one stacked example's thumbnail, or null when its thumbnail
   *  endpoint already failed (so the row falls back to the icon). */
  exampleThumbnailUrl(example: MediaExampleItem): string | null {
    if (!example.value || example.thumbFailed) return null;
    return `/api/server-media-files/${encodeURIComponent(example.value)}/thumbnail`;
  }

  onExampleThumbError(index: number): void {
    this.mediaExamples.update((list) =>
      list.map((ex, i) => (i === index ? { ...ex, thumbFailed: true } : ex)),
    );
  }

  backToMain(): void {
    this.view.set('main');
  }

  // --- Clear examples ---

  /** Reset only the media-example stack, leaving any pending text untouched.
   *  Used when the user starts typing a text example (the two kinds are
   *  mutually exclusive); individual rows are removed via their own
   *  Remove buttons. */
  private clearMediaExample(): void {
    this.mediaExamples.set([]);
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
    this.labelImporterFieldOptions.reset();
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.labelImporterValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        !field.allow_free_text &&
        (field.options?.length ?? 0) > 0
      ) {
        this.labelImporterValues[field.key] = field.options![0];
      }
    }
    this.labelImporterFieldOptions.refreshAll(fields, this.labelImporterValues);
    this.trainedView = 'form';
  }

  onLabelImporterFieldChanged(changedKey: string): void {
    if (!this.selectedLabelImporter?.fields) return;
    this.labelImporterFieldOptions.refreshDependentsOf(
      changedKey,
      this.selectedLabelImporterFields,
      this.labelImporterValues,
    );
  }

  backToImporterPicker(): void {
    this.trainedView = 'picker';
    this.selectedLabelImporter = null;
    this.error.set('');
    this.labelImporterFieldOptions.reset();
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
    if (this.submitting()) return;

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
          const ingestTaskId = String(resp?.ingest_task_id || '');
          if (!ingestTaskId) {
            this.loadAndFinish(newId);
            return;
          }
          // The imported labels' media are still being fetched and embedded in
          // the background (#2703). Loading the detector now would restore its
          // labels against media that aren't in the dataset yet, so wait the
          // ingest out — showing its bar instead of an opaque spinner. A failed
          // ingest still completes the stream: the detector exists either way,
          // and the failure surfaces as a toast from the SSE error router.
          this.progressEvents.detectorTaskUntilDone$(ingestTaskId).subscribe({
            next: (task) => this.ingestTask.set(task),
            complete: () => {
              this.ingestTask.set(null);
              this.loadAndFinish(newId);
            },
          });
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Failed to create detector from labelset'));
        },
      });
  }

  /** Load the freshly-created detector, then hand its id back to the caller.
   *  A failed load still emits: the detector is in the registry regardless. */
  private loadAndFinish(detectorId: string): void {
    this.detectorsRegistryApi.loadDetector(detectorId).subscribe({
      next: () => {
        this.submitting.set(false);
        this.created.emit(detectorId);
      },
      error: () => {
        this.submitting.set(false);
        this.created.emit(detectorId);
      },
    });
  }

  /** Bar geometry for the running media ingest. */
  get ingestBar(): ProgressBarState {
    return progressBarState(this.ingestTask());
  }

  /** One-line status for the running media ingest. */
  get ingestMessage(): string {
    return formatProgressMessage(this.ingestTask(), 'Fetching the imported labels’ media…');
  }

  // --- Submit ---

  submit(): void {
    // The footer button's [disabled] isn't the only way in: the text-example
    // and name inputs fire this on Enter while a POST is in flight, and a
    // second registerDetector would create a duplicate detector.
    if (this.submitting()) return;

    if (this.tab === 'trained') {
      this.submitTrained();
      return;
    }

    const trimmedName = this.name().trim();
    if (!trimmedName) {
      this.error.set('Name is required');
      return;
    }

    // Media examples win over pending text (the two are mutually exclusive
    // in the form; a non-empty stack means the text field was cleared).
    const mediaExamples = this.mediaExamples();
    const pendingTrimmed = this.pendingText().trim();

    if (mediaExamples.length === 0 && !pendingTrimmed) {
      this.error.set('An example (text or media) is required');
      return;
    }

    this.submitting.set(true);
    this.error.set('');

    const textQuery = mediaExamples.length === 0 ? pendingTrimmed : '';
    // The legacy scalar is the detector's headline exemplar (dashboard
    // display, Autopilot fallback), so prefer a hand-picked one over a
    // seed — a seed is deliberately not the thing being hunted.
    const mediaExample = (mediaExamples.find((ex) => !ex.seed) ?? mediaExamples[0])?.value || '';
    const examplesPayload = textQuery
      ? [{ type: 'text', value: textQuery }]
      : mediaExamples.map((ex) => ({
          type: 'media',
          value: ex.value,
          // Both extra keys are additive: an example without either keeps
          // the legacy {type, value} shape. `origin` makes the example
          // re-derivable; `labeled: false` marks an unlabeled seed, which
          // steers the first sort without becoming a Good vote.
          ...(ex.origin ? { origin: ex.origin } : {}),
          ...(ex.seed ? { labeled: false } : {}),
        }));

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

  /** "Browse Images…", "Browse Audio…", "Browse Media…" as a fallback. */
  get browseMediaLabel(): string {
    const mediaType = this.mediaType();
    const name = mediaType ? this.getMediaTypeLabel(mediaType) : '';
    if (!name) return 'Browse Media…';
    // audio and text don't take a plural -s in this context.
    const uncountable = mediaType === 'audio' || mediaType === 'text';
    return `Browse ${uncountable ? name : name + 's'}…`;
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
