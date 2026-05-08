import { Component, EventEmitter, HostListener, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import {
  DemoDataset,
  ImporterField,
  ImporterInfo,
  ImporterPickerTab,
  MediaTypeInfo,
} from '../../../models/api.models';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../../modals/media-crop-modal/media-crop-modal.component';
import { ColMeta, ManagedColumns } from '../../../utils/managed-columns';

interface BrowseEntry {
  name: string;
  path: string;
  size_bytes?: number;
  modified_at?: string;
  isDir: boolean;
}

interface LabelImporterInfo {
  name: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  hidden_from_picker?: boolean;
}

type ModalView = 'main' | 'media-picker';
type ModalTab = 'blank' | 'trained';
type TrainedSubView = 'picker' | 'form';

@Component({
  selector: 'vt-new-model-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, FileBrowserComponent, MediaCropModalComponent],
  templateUrl: './new-model-modal.component.html',
  styleUrl: './new-model-modal.component.scss',
})
export class NewModelModalComponent implements OnInit {
  /** Media type of the currently active dataset, if any. */
  @Input() defaultMediaType = '';

  @Output() closed = new EventEmitter<void>();
  @Output() created = new EventEmitter<string>();

  view: ModalView = 'main';
  tab: ModalTab = 'blank';
  name = '';
  mediaType = 'audio';
  pendingText = '';
  mediaTypes: string[] = [];
  mediaTypeInfos: MediaTypeInfo[] = [];
  submitting = false;
  error = '';
  mediaTypeDropdownOpen = false;

  // Single example (text or media, not both)
  exampleType: 'text' | 'media' | null = null;
  exampleValue = '';
  exampleDisplay = '';
  exampleMediaType = '';
  exampleThumbFailed = false;

  // --- Media picker state (shares structure with the Add Dataset modal) ---

  /** All importers discovered from the backend, filtered to picker_views we
   *  can use to select a single example file (demo, server_folder,
   *  local_folder, local_files). */
  mediaImporters: ImporterInfo[] = [];
  /** Tab declarations (categories) returned by the backend. */
  declaredImporterTabs: ImporterPickerTab[] = [];
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
   *  here — the user can still use them via the Add Dataset modal. */
  private static readonly SUPPORTED_PICKER_VIEWS = new Set([
    'demo',
    'server_folder',
    'local_folder',
    'local_files',
  ]);

  // --- Demo picker state ---
  demos: DemoDataset[] = [];
  demoTabs: string[] = [];
  activeDemoTab = '';
  demoLoading = false;

  /** Demo table column metadata + controller.  Reuses the same storage key
   *  as the Add Dataset modal so column order and sort preferences stay in
   *  sync between the two demo tables. */
  static readonly DEMO_COL_META: Record<string, ColMeta> = {
    label: { label: 'Name', title: 'Demo dataset name (click to sort)', sortable: true },
    num_files: { label: '# Media', title: 'Number of media files in the demo dataset (click to sort)', sortable: true },
    num_categories: { label: '# Cat.', title: 'Number of distinct categories or classes in the dataset (click to sort)', sortable: true },
    description: { label: 'Description', title: 'Short description of the demo dataset contents (click to sort)', sortable: true },
    status: { label: 'Readiness', title: 'Whether the dataset is pre-downloaded and ready to browse, or still needs to be fetched (click to sort)', sortable: true },
  };
  static readonly DEMO_COLUMNS_DEFAULT = ['label', 'num_files', 'num_categories', 'description', 'status'];
  private static readonly DEMO_COL_ORDER_KEY = 'vtsearch.dashboard.demoColumnOrder';

  demoCols = new ManagedColumns(
    NewModelModalComponent.DEMO_COLUMNS_DEFAULT,
    NewModelModalComponent.DEMO_COL_META,
    { initialSort: 'num_files', storageKey: NewModelModalComponent.DEMO_COL_ORDER_KEY },
  );

  // --- Demo file browser (shown after picking a demo from the table) ---
  demoFileBrowsing = false;
  demoFileBrowseSource = '';
  demoFileBrowseLabel = '';
  demoFileBrowsePath: string[] = [];
  demoFileBrowseEntries: BrowseEntry[] = [];
  demoFileLoading = false;

  // --- Server folder browser (mirrors the Add Dataset breadcrumb browser
  //     but lists files too so the user can pick one as an example). ---
  sfBrowsePath = '';
  sfBrowseRootPath = '';
  sfBrowseDirs: { name: string; path: string; modified_at?: string }[] = [];
  sfBrowseFiles: BrowseEntry[] = [];
  sfBrowseLoading = false;
  sfBrowseError = '';
  sfFileSelecting = false;

  // Pending crop confirmation state.
  pendingFile: File | null = null;
  pendingFileMediaType = '';

  // "Trained" tab state
  trainedView: TrainedSubView = 'picker';
  labelImporters: LabelImporterInfo[] = [];
  labelImportersLoading = false;
  selectedLabelImporter: LabelImporterInfo | null = null;
  labelImporterValues: Record<string, string> = {};
  labelImporterFile: File | null = null;
  labelImporterFileFieldKey: string | null = null;

  constructor(
    private modelsApi: TrainableModelsApiService,
    private datasetsApi: DatasetsApiService,
    private sortingApi: SortingApiService,
    private labelImportersApi: LabelImportersApiService,
  ) {}

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
    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypeInfos = res.media_types || [];
        this.mediaTypes = this.mediaTypeInfos.map((t) => t.type_id || t.name);
      },
    });

    // Prefer the explicit default (active dataset's type) over the all-datasets guess.
    if (this.defaultMediaType) {
      this.mediaType = this.defaultMediaType;
    } else {
      this.datasetsApi.getRegistry().subscribe({
        next: (res) => {
          const types = new Set(
            (res.datasets || []).map((d) => d['media_type'] as string).filter(Boolean),
          );
          if (types.size === 1) {
            this.mediaType = [...types][0];
          }
        },
      });
    }
  }

  get modalTitle(): string {
    if (this.view === 'media-picker') return 'Select Media Example';
    return 'New Model';
  }

  get hasExample(): boolean {
    return this.exampleType === 'media' || !!this.pendingText.trim();
  }

  get hasMediaExample(): boolean {
    return this.exampleType === 'media';
  }

  get hasPendingText(): boolean {
    return !!this.pendingText.trim();
  }

  get canSubmitBlank(): boolean {
    return !!this.name.trim() && this.hasExample && !this.submitting;
  }

  get canSubmitTrained(): boolean {
    return (
      !!this.name.trim() &&
      !!this.selectedLabelImporter &&
      this.trainedView === 'form' &&
      !this.submitting
    );
  }

  // --- Tab switching ---

  setTab(tab: ModalTab): void {
    if (this.submitting) return;
    this.tab = tab;
    this.error = '';
  }

  // --- Media picker (shared structure with Add Dataset) ---

  openMediaPicker(): void {
    this.view = 'media-picker';
    this.activeImporterTab = '';
    this.selectedImporter = null;
    this.resetDemoPickerState();
    this.resetServerFolderState();
    this.loadMediaImporters();
  }

  private loadMediaImporters(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaImporters = (res.importers || []).filter(
          (imp) =>
            !imp['hidden_from_picker'] &&
            NewModelModalComponent.SUPPORTED_PICKER_VIEWS.has(imp.picker_view || ''),
        );
        this.declaredImporterTabs = res.tabs || [];
      },
    });
  }

  /** Importers ordered like the Add Dataset modal: known picker_views
   *  first, then any extras in registry order. */
  get orderedImporters(): ImporterInfo[] {
    const order = NewModelModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      const imp = this.mediaImporters.find((i) => i.name === name);
      if (imp) result.push(imp);
    }
    for (const imp of this.mediaImporters) {
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
   *  supported importer are shown — empty categories (e.g. ``"services"``
   *  on a vanilla install) are hidden so the picker stays focused on
   *  options the user can act on. */
  get visibleImporterTabs(): ImporterPickerTab[] {
    const usedCategories = new Set(
      this.orderedImporters.map((imp) => imp.category || '').filter(Boolean),
    );
    const visible: ImporterPickerTab[] = [];
    const seen = new Set<string>();
    const declared = [...this.declaredImporterTabs].sort(
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
  }

  selectImporter(importer: ImporterInfo): void {
    this.selectedImporter = importer;
    this.error = '';
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
    this.demos = [];
    this.demoTabs = [];
    this.activeDemoTab = '';
    this.demoLoading = false;
    this.demoFileBrowsing = false;
    this.demoFileBrowseSource = '';
    this.demoFileBrowseLabel = '';
    this.demoFileBrowsePath = [];
    this.demoFileBrowseEntries = [];
    this.demoFileLoading = false;
  }

  private openDemoPicker(): void {
    this.resetDemoPickerState();
    this.demoLoading = true;
    this.datasetsApi.getDemoList().subscribe({
      next: (res) => {
        this.demos = res.datasets || [];
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
    const registryOrder = this.mediaTypeInfos.map((mt) => mt.type_id);
    this.demoTabs = registryOrder.filter((mt) => grouped.has(mt));
    for (const mt of grouped) {
      if (!this.demoTabs.includes(mt)) this.demoTabs.push(mt);
    }
  }

  selectDemoTab(tab: string): void {
    this.activeDemoTab = tab;
  }

  onDemoHeaderClick(col: string): void {
    if (this.demoCols.meta(col).sortable) this.demoCols.sortBy(col);
  }

  get filteredDemos(): DemoDataset[] {
    const items = this.demos.filter((d) => d.media_type === this.activeDemoTab);
    const statusOrder: Record<string, number> = { ready: 0, needs_embedding: 1, needs_download: 2 };
    const sortKey = this.demoCols.sortColumn;
    const asc = this.demoCols.sortAsc;
    return [...items].sort((a, b) => {
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

  /** True when the demo's files are on disk and can be browsed.  Demos
   *  that still need to be downloaded show a tooltip explaining how to
   *  fetch them via the Add Dataset modal. */
  isDemoBrowsable(demo: DemoDataset): boolean {
    return demo.status === 'ready' || demo.status === 'needs_embedding';
  }

  selectDemo(demo: DemoDataset): void {
    if (!this.isDemoBrowsable(demo)) return;
    this.demoFileBrowsing = true;
    this.demoFileBrowseSource = `demo:${demo.name}`;
    this.demoFileBrowseLabel = demo.label;
    this.demoFileBrowsePath = [demo.label];
    this.loadDemoDirectory('');
  }

  private loadDemoDirectory(relPath: string): void {
    this.demoFileLoading = true;
    this.demoFileBrowseEntries = [];
    this.datasetsApi.browseMediaFiles(this.demoFileBrowseSource, relPath).subscribe({
      next: (res) => {
        this.demoFileBrowseEntries = this.toEntries(res.directories, res.files);
        this.demoFileLoading = false;
      },
      error: () => {
        this.demoFileLoading = false;
      },
    });
  }

  enterDemoDirectory(entry: BrowseEntry): void {
    this.demoFileBrowsePath.push(entry.name);
    this.loadDemoDirectory(entry.path);
  }

  navigateDemoBreadcrumb(index: number): void {
    if (index === 0) {
      this.demoFileBrowsePath = [this.demoFileBrowseLabel];
      this.loadDemoDirectory('');
      return;
    }
    this.demoFileBrowsePath = this.demoFileBrowsePath.slice(0, index + 1);
    const relPath = this.demoFileBrowsePath.slice(1).join('/');
    this.loadDemoDirectory(relPath);
  }

  selectDemoFile(entry: BrowseEntry): void {
    this.demoFileLoading = true;
    this.datasetsApi.selectBrowsedFile(this.demoFileBrowseSource, entry.path).subscribe({
      next: (res) => {
        this.exampleType = 'media';
        this.exampleValue = res.filename;
        this.exampleDisplay = res.original_name || entry.name;
        this.exampleMediaType = this.activeDemoTab || this.mediaType;
        this.exampleThumbFailed = false;
        this.pendingText = '';
        this.demoFileLoading = false;
        this.view = 'main';
      },
      error: () => {
        this.error = 'Failed to select file';
        this.demoFileLoading = false;
      },
    });
  }

  /** Return to the demo table from the demo file browser. */
  backToDemoTable(): void {
    this.demoFileBrowsing = false;
    this.demoFileBrowseSource = '';
    this.demoFileBrowseLabel = '';
    this.demoFileBrowsePath = [];
    this.demoFileBrowseEntries = [];
  }

  // --- Server folder browser (with file selection) ---

  private resetServerFolderState(): void {
    this.sfBrowsePath = '';
    this.sfBrowseRootPath = '';
    this.sfBrowseDirs = [];
    this.sfBrowseFiles = [];
    this.sfBrowseLoading = false;
    this.sfBrowseError = '';
    this.sfFileSelecting = false;
  }

  private openServerFolderBrowser(): void {
    this.resetServerFolderState();
    this.sfLoadDirectory('');
  }

  sfLoadDirectory(path: string): void {
    this.sfBrowseLoading = true;
    this.sfBrowseError = '';
    this.datasetsApi.browseMediaFiles('folder', path).subscribe({
      next: (res) => {
        this.sfBrowseDirs = res.directories || [];
        this.sfBrowseFiles = (res.files || []).map((f) => ({
          name: f.name,
          path: f.path,
          size_bytes: f.size_bytes,
          modified_at: f.modified_at,
          isDir: false,
        }));
        this.sfBrowsePath = path;
        this.sfBrowseRootPath = res.root_path;
        this.sfBrowseLoading = false;
      },
      error: (err) => {
        this.sfBrowseError =
          err.error?.error || 'Could not browse server folders. Is saved_datasets_dir configured?';
        this.sfBrowseLoading = false;
      },
    });
  }

  sfEnterDirectory(dir: { name: string; path: string }): void {
    this.sfLoadDirectory(dir.path);
  }

  sfGoUp(): void {
    if (!this.sfBrowsePath) return;
    const parts = this.sfBrowsePath.split('/');
    parts.pop();
    this.sfLoadDirectory(parts.join('/'));
  }

  get sfBreadcrumbs(): string[] {
    if (!this.sfBrowsePath) return [];
    return this.sfBrowsePath.split('/');
  }

  sfNavigateBreadcrumb(index: number): void {
    const parts = this.sfBrowsePath.split('/');
    this.sfLoadDirectory(parts.slice(0, index + 1).join('/'));
  }

  get sfAbsolutePath(): string {
    if (!this.sfBrowseRootPath) return '';
    if (!this.sfBrowsePath) return this.sfBrowseRootPath;
    return this.sfBrowseRootPath + '/' + this.sfBrowsePath;
  }

  sfSelectFile(entry: BrowseEntry): void {
    this.sfFileSelecting = true;
    this.datasetsApi.selectBrowsedFile('folder', entry.path).subscribe({
      next: (res) => {
        this.exampleType = 'media';
        this.exampleValue = res.filename;
        this.exampleDisplay = res.original_name || entry.name;
        this.exampleMediaType = this.mediaType || this.mediaTypeFromFilename(entry.name);
        this.exampleThumbFailed = false;
        this.pendingText = '';
        this.sfFileSelecting = false;
        this.view = 'main';
      },
      error: () => {
        this.sfBrowseError = 'Failed to select file';
        this.sfFileSelecting = false;
      },
    });
  }

  // --- Local file upload (single file for the example) ---

  /** Build a unified entry list (directories first, then files). */
  private toEntries(
    directories: { name: string; path: string; modified_at?: string }[] | undefined,
    files:
      | { name: string; path: string; size_bytes: number; modified_at?: string }[]
      | undefined,
  ): BrowseEntry[] {
    const entries: BrowseEntry[] = [];
    for (const d of directories || []) {
      entries.push({ name: d.name, path: d.path, modified_at: d.modified_at, isDir: true });
    }
    for (const f of files || []) {
      entries.push({
        name: f.name,
        path: f.path,
        size_bytes: f.size_bytes,
        modified_at: f.modified_at,
        isDir: false,
      });
    }
    return entries;
  }

  formatSize(bytes?: number): string {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  /** Local file picker handler.  Used by both the inline "Upload File…"
   *  button on the main form and the Local Folder / Local Files cards in
   *  the picker.  Multi-file selections (e.g. webkitdirectory) are
   *  collapsed to the first file since only one example is needed. */
  onLocalFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    input.value = '';
    this.pendingFile = file;
    this.pendingFileMediaType = this.mediaType || this.mediaTypeFromFile(file);
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
          this.exampleType = 'media';
          this.exampleValue = res.filename;
          this.exampleDisplay = res.original_name || res.filename;
          this.exampleMediaType = mediaType || this.mediaType;
          this.exampleThumbFailed = false;
          this.pendingText = '';
          // Close the picker if it was open so the user lands back on the form.
          if (this.view === 'media-picker') this.view = 'main';
        },
        error: () => {
          this.error = 'Failed to upload file';
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
    if (this.exampleType !== 'media' || !this.exampleValue || this.exampleThumbFailed) {
      return null;
    }
    return `/api/server-media-files/${encodeURIComponent(this.exampleValue)}/thumbnail`;
  }

  onExampleThumbError(): void {
    this.exampleThumbFailed = true;
  }

  backToMain(): void {
    this.view = 'main';
  }

  // --- Media-type tab labels (mirrors Add Dataset's helpers) ---

  getDemoTabLabel(typeId: string): string {
    const mt = this.mediaTypeInfos.find((m) => m.type_id === typeId);
    if (mt) return (mt.tab_title || mt.name).trim();
    return typeId;
  }

  getDemoTabIcon(typeId: string): string {
    const mt = this.mediaTypeInfos.find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  statusBadgeClass(status: string): string {
    if (status === 'ready') return 'badge-ready';
    if (status === 'needs_embedding') return 'badge-embedding';
    return 'badge-download';
  }

  statusBadgeLabel(status: string): string {
    if (status === 'ready') return 'Ready';
    if (status === 'needs_embedding') return 'Needs Embed';
    return 'Needs Download';
  }

  // --- Clear example ---

  clearExample(): void {
    this.exampleType = null;
    this.exampleValue = '';
    this.exampleDisplay = '';
    this.exampleMediaType = '';
    this.exampleThumbFailed = false;
    this.pendingText = '';
  }

  // --- Trained tab: label importers ---

  private ensureLabelImportersLoaded(): void {
    if (this.labelImporters.length > 0 || this.labelImportersLoading) return;
    this.labelImportersLoading = true;
    this.labelImportersApi.list().subscribe({
      next: (list: any[]) => {
        this.labelImporters = (list || []).filter((imp: LabelImporterInfo) => !imp.hidden_from_picker);
        this.labelImportersLoading = false;
      },
      error: () => {
        this.labelImportersLoading = false;
        this.error = 'Failed to load label importers';
      },
    });
  }

  onSelectTrainedTab(): void {
    this.setTab('trained');
    this.ensureLabelImportersLoaded();
  }

  selectLabelImporter(importer: LabelImporterInfo): void {
    this.selectedLabelImporter = importer;
    this.labelImporterValues = {};
    this.labelImporterFile = null;
    this.labelImporterFileFieldKey = null;
    this.error = '';
    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default) this.labelImporterValues[field.key] = field.default;
      }
    }
    this.trainedView = 'form';
  }

  backToImporterPicker(): void {
    this.trainedView = 'picker';
    this.selectedLabelImporter = null;
    this.error = '';
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
    const trimmedName = this.name.trim();
    if (!trimmedName) {
      this.error = 'Name is required';
      return;
    }
    if (!this.selectedLabelImporter) {
      this.error = 'A label importer is required';
      return;
    }

    this.submitting = true;
    this.error = '';

    const params: Record<string, unknown> = {
      name: trimmedName,
      ...this.labelImporterValues,
    };

    this.modelsApi
      .registerModelFromLabelset(
        this.selectedLabelImporter.name,
        params,
        this.labelImporterFile ?? undefined,
        this.labelImporterFileFieldKey ?? undefined,
      )
      .subscribe({
        next: (resp: any) => {
          const newId = resp?.model?.id || '';
          if (!newId) {
            this.submitting = false;
            this.error = 'Server did not return a model id';
            return;
          }
          this.modelsApi.loadModel(newId).subscribe({
            next: () => {
              this.submitting = false;
              this.created.emit(newId);
            },
            error: () => {
              // Model exists in registry even if load failed; still emit.
              this.submitting = false;
              this.created.emit(newId);
            },
          });
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Failed to create model from labelset';
        },
      });
  }

  // --- Submit ---

  submit(): void {
    if (this.tab === 'trained') {
      this.submitTrained();
      return;
    }

    const trimmedName = this.name.trim();
    if (!trimmedName) {
      this.error = 'Name is required';
      return;
    }

    // Accept pending text as the text example on submit
    const pendingTrimmed = this.pendingText.trim();
    if (this.exampleType !== 'media' && pendingTrimmed) {
      this.exampleType = 'text';
      this.exampleValue = pendingTrimmed;
      this.exampleDisplay = pendingTrimmed;
    }

    if (!this.exampleType) {
      this.error = 'An example (text or media) is required';
      return;
    }

    this.submitting = true;
    this.error = '';

    const textQuery = this.exampleType === 'text' ? this.exampleValue : '';
    const mediaExample = this.exampleType === 'media' ? this.exampleValue : '';
    const examplesPayload = [{ type: this.exampleType!, value: this.exampleValue }];

    this.modelsApi
      .registerModel({
        name: trimmedName,
        media_type: this.mediaType,
        text_query: textQuery,
        media_example: mediaExample,
        examples: examplesPayload,
      })
      .subscribe({
        next: (resp: any) => {
          this.submitting = false;
          this.created.emit(resp?.model?.id || '');
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Failed to create model';
        },
      });
  }

  getMediaTypeLabel(typeId: string): string {
    const mt = this.mediaTypeInfos.find((m) => m.type_id === typeId);
    if (mt) {
      return (mt.tab_title || mt.name).trim();
    }
    return typeId;
  }

  getMediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypeInfos.find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  close(): void {
    this.closed.emit();
  }
}
