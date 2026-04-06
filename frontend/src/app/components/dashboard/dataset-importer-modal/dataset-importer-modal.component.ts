import { Component, EventEmitter, HostListener, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { IconComponent } from '../../icon/icon.component';
import { ClipperChooserComponent, ClipperSelection } from '../clipper-chooser/clipper-chooser.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { ImporterInfo, DemoDataset, MediaTypeInfo, ClipperInfo, ClipperParameter, EmbedderInfo } from '../../../models/api.models';

type ModalView = 'picker' | 'form' | 'demo' | 'server_folder';

@Component({
  selector: 'vt-dataset-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, FileBrowserComponent, IconComponent, ClipperChooserComponent],
  templateUrl: './dataset-importer-modal.component.html',
  styleUrl: './dataset-importer-modal.component.scss',
})
export class DatasetImporterModalComponent implements OnInit {
  /** Media type_id guessed from existing datasets/models (e.g. "image"). */
  @Input() guessedMediaType = '';
  /** Embedder name guessed from existing datasets/in-progress loads (e.g. "siglip"). */
  @Input() guessedMediaEmbedder = '';

  @Output() closed = new EventEmitter<void>();
  @Output() importStarted = new EventEmitter<void>();
  @Output() demoSelected = new EventEmitter<DemoDataset>();

  view: ModalView = 'picker';
  importers: ImporterInfo[] = [];
  selectedImporter: ImporterInfo | null = null;
  formValues: Record<string, any> = {};
  selectedFile: File | null = null;
  submitting = false;
  error = '';

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
  demoSortKey = 'num_files';
  demoSortAsc = true;
  demoLoading = false;
  demoEmbedders: EmbedderInfo[] = [];
  selectedDemoEmbedder = '';
  demoEmbedder = '';
  demoClippers: ClipperInfo[] = [];
  selectedDemoClipper = '';

  // Demo table column resize state
  demoColWidths: Record<string, number> = {};
  demoTableFixed = false;
  demoTableWidth = 0;
  private demoResizeInit = false;
  private demoResizeState: {
    startX: number;
    startWidth: number;
    col: string;
    dragged: boolean;
    tableEl: HTMLTableElement;
  } | null = null;

  // Server folder browser state
  sfBrowseDirs: { name: string; path: string; modified_at?: string }[] = [];
  sfBrowsePath = '';
  sfBrowseRootPath = '';
  sfBrowseLoading = false;
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

  // Clipper chooser modal state
  clipperChooserOpen = false;
  /** Which context opened the chooser: 'form' | 'demo' | 'sf' */
  clipperChooserContext: 'form' | 'demo' | 'sf' = 'form';
  clipperChooserClippers: ClipperInfo[] = [];

  constructor(private datasetsApi: DatasetsApiService) {}

  ngOnInit(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.importers = (res.importers || []).filter(
          (imp) => !imp['hidden_from_picker']
        );
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
  }

  /** Desired picker order: folder, demo placeholder, then remaining importers. */
  private static readonly PICKER_ORDER = ['folder', '_demo', 'combine_datasets'];

  get orderedImporters(): ImporterInfo[] {
    const demoPlaceholder = { name: '_demo' } as ImporterInfo;
    const order = DatasetImporterModalComponent.PICKER_ORDER;
    const result: ImporterInfo[] = [];
    for (const name of order) {
      if (name === '_demo') {
        result.push(demoPlaceholder);
      } else {
        const imp = this.importers.find((i) => i.name === name);
        if (imp) result.push(imp);
      }
    }
    // Append any importers not in the explicit order
    for (const imp of this.importers) {
      if (!order.includes(imp.name)) {
        result.push(imp);
      }
    }
    return result;
  }

  selectImporter(importer: ImporterInfo): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.error = '';
    this.selectedClipper = '';
    this.availableClippers = [];
    this.clipperParamValues = {};
    this.selectedEmbedder = '';
    this.availableEmbedders = [];

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

    this.view = 'form';
  }

  onMediaTypeChange(mediaType: string): void {
    this.formValues['media_type'] = mediaType;
    this.loadClippers(mediaType);
    this.loadEmbedders(mediaType);
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
        // Prefer guessed embedder when it's available for this media type
        const guessedMatch = this.guessedMediaEmbedder
          ? embedders.find((e) => e.name === this.guessedMediaEmbedder)
          : null;
        this.selectedEmbedder = guessedMatch ? guessedMatch.name : (embedders.length > 0 ? embedders[0].name : '');
      },
    });
  }

  openDemoPicker(): void {
    this.view = 'demo';
    this.demoLoading = true;
    this.demos = [];
    this.demoTabs = [];
    this.activeTab = '';

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
    if (this.demoTabs.length > 0 && !this.activeTab) {
      // Prefer guessed media type if it has demos
      if (this.guessedMediaType && this.demoTabs.includes(this.guessedMediaType)) {
        this.activeTab = this.guessedMediaType;
      } else {
        this.activeTab = this.demoTabs[0];
      }
      this.loadDemoEmbedders(this.activeTab);
    }
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
        // Prefer guessed embedder when it's available for this media type
        const guessedMatch = this.guessedMediaEmbedder
          ? embedders.find((e) => e.name === this.guessedMediaEmbedder)
          : null;
        this.selectedDemoEmbedder = guessedMatch ? guessedMatch.name : (embedders.length > 0 ? embedders[0].name : '');
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
    return items.sort((a, b) => {
      const key = this.demoSortKey as keyof DemoDataset;
      let va: any = a[key];
      let vb: any = b[key];
      if (key === 'status') {
        va = statusOrder[va as string] ?? 3;
        vb = statusOrder[vb as string] ?? 3;
      }
      if (typeof va === 'number' && typeof vb === 'number') {
        return this.demoSortAsc ? va - vb : vb - va;
      }
      va = String(va || '').toLowerCase();
      vb = String(vb || '').toLowerCase();
      return this.demoSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  selectDemoTab(tab: string): void {
    this.activeTab = tab;
    this.loadDemoEmbedders(tab);
  }

  sortDemoBy(key: string): void {
    if (this.demoSortKey === key) {
      this.demoSortAsc = !this.demoSortAsc;
    } else {
      this.demoSortKey = key;
      this.demoSortAsc = true;
    }
  }

  demoSortIndicator(key: string): string {
    if (this.demoSortKey !== key) return '';
    return this.demoSortAsc ? ' \u25B2' : ' \u25BC';
  }

  // --- Demo table column resize ---

  startDemoResize(event: MouseEvent, col: string): void {
    event.stopPropagation();
    event.preventDefault();

    const th = (event.target as HTMLElement).closest('th') as HTMLElement;
    const tableEl = th.closest('table') as HTMLTableElement;

    if (!this.demoResizeInit) {
      const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
      let totalWidth = 0;
      ths.forEach((t) => {
        const colKey = t.getAttribute('data-col');
        const w = t.offsetWidth;
        if (colKey) this.demoColWidths[colKey] = w;
        totalWidth += w;
      });
      this.demoTableWidth = totalWidth;
      this.demoResizeInit = true;
      this.demoTableFixed = true;
    }

    this.demoResizeState = {
      startX: event.clientX,
      startWidth: this.demoColWidths[col] ?? 100,
      col,
      dragged: false,
      tableEl,
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  @HostListener('document:mousemove', ['$event'])
  onDemoResizeMove(event: MouseEvent): void {
    if (!this.demoResizeState) return;
    const delta = event.clientX - this.demoResizeState.startX;
    if (Math.abs(delta) > 3) this.demoResizeState.dragged = true;
    if (!this.demoResizeState.dragged) return;
    const newWidth = Math.max(30, this.demoResizeState.startWidth + delta);
    const prevWidth = this.demoColWidths[this.demoResizeState.col] ?? this.demoResizeState.startWidth;
    this.demoColWidths[this.demoResizeState.col] = newWidth;
    this.demoTableWidth = Math.max(100, this.demoTableWidth + (newWidth - prevWidth));
  }

  @HostListener('document:mouseup')
  onDemoResizeEnd(): void {
    if (!this.demoResizeState) return;
    const state = this.demoResizeState;
    this.demoResizeState = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    if (!state.dragged) {
      this.autoSizeDemoColumn(state.tableEl, state.col);
    }
  }

  private autoSizeDemoColumn(tableEl: HTMLTableElement, col: string): void {
    const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
    let colIndex = -1;
    for (let i = 0; i < ths.length; i++) {
      if (ths[i].getAttribute('data-col') === col) {
        colIndex = i;
        break;
      }
    }
    if (colIndex < 0) return;

    const prevTableWidth = tableEl.style.width;
    tableEl.classList.remove('demo-table-fixed');
    tableEl.style.width = 'auto';
    ths[colIndex].style.width = '';

    let maxWidth = 0;
    const rows = tableEl.querySelectorAll('tbody tr');
    rows.forEach((row) => {
      const cell = row.children[colIndex] as HTMLElement | undefined;
      if (cell) {
        if (cell.hasAttribute('colspan')) return;
        maxWidth = Math.max(maxWidth, cell.scrollWidth);
      }
    });
    if (maxWidth === 0) {
      maxWidth = ths[colIndex].offsetWidth;
    }
    maxWidth = Math.max(30, maxWidth + 2);

    this.demoColWidths[col] = maxWidth;
    this.demoTableFixed = true;
    this.demoResizeInit = true;

    let total = 0;
    ths.forEach((t) => {
      const key = t.getAttribute('data-col');
      if (key && key !== col) {
        if (!this.demoColWidths[key]) this.demoColWidths[key] = t.offsetWidth;
        total += this.demoColWidths[key];
      } else if (key === col) {
        total += maxWidth;
      }
    });
    this.demoTableWidth = total;

    tableEl.style.width = prevTableWidth;
    tableEl.classList.add('demo-table-fixed');
  }

  /** Convert a type_id (e.g. "image") to the corresponding folder_import_name (e.g. "images"). */
  private toFolderName(typeId: string): string {
    if (!typeId) return '';
    const mt = this.mediaTypes.find((m) => m.type_id === typeId);
    return mt?.folder_import_name || typeId;
  }

  getMediaTypeOptionLabel(opt: string): string {
    const mt = this.mediaTypes.find((m) => m.folder_import_name === opt);
    if (mt) {
      return (mt.tab_title || mt.name).trim();
    }
    return opt;
  }

  getTabLabel(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    if (mt) {
      return (mt.tab_title || mt.name).trim();
    }
    return mediaType;
  }

  getTabIcon(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    return mt?.icon || '';
  }

  getTabText(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    if (mt) {
      return mt.tab_title || mt.name;
    }
    return mediaType;
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

  /** Embedders available for the currently active demo tab's media type. */
  get demoEmbeddersForTab(): EmbedderInfo[] {
    return this.allEmbedders.filter((e) => e.media_type_id === this.activeTab);
  }

  selectDemo(demo: DemoDataset): void {
    this.demoSelected.emit({ ...demo, embedder: this.selectedDemoEmbedder, clipper: this.selectedDemoClipper } as any);
    this.closed.emit();
  }

  selectDemoTabWithEmbedder(tab: string): void {
    this.selectDemoTab(tab);
    // Reset embedder selection for the new tab
    const embedders = this.allEmbedders.filter((e) => e.media_type_id === tab);
    this.demoEmbedder = embedders.length > 0 ? embedders[0].name : '';
    // Clipper is reset by loadDemoEmbedders (called from selectDemoTab)
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.error = '';
  }

  // --- Server folder browser ---

  openServerFolderBrowser(): void {
    this.view = 'server_folder';
    this.sfBrowsePath = '';
    this.sfBrowseRootPath = '';
    this.sfBrowseDirs = [];
    this.sfBrowseError = '';
    this.sfSubmitting = false;

    // Load media type options from the folder importer's fields
    const folderImporter = this.importers.find((imp) => imp.name === 'folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    this.sfMediaTypeOptions = mtField?.options || [];

    // Prefer guessed media type when available
    const guessedFolder = this.toFolderName(this.guessedMediaType);
    if (guessedFolder && this.sfMediaTypeOptions.includes(guessedFolder)) {
      this.sfMediaType = guessedFolder;
    } else {
      this.sfMediaType = mtField?.default || this.sfMediaTypeOptions[0] || 'sounds';
    }

    this.sfLoadEmbedders(this.sfMediaType);
    this.sfLoadClippers(this.sfMediaType);
    this.sfLoadDirectory('');
  }

  sfLoadDirectory(path: string): void {
    this.sfBrowseLoading = true;
    this.sfBrowseError = '';
    this.datasetsApi.browseMediaFiles('folder', path).subscribe({
      next: (res) => {
        this.sfBrowseDirs = res.directories;
        this.sfBrowsePath = path;
        this.sfBrowseRootPath = res.root_path;
        this.sfBrowseLoading = false;
      },
      error: (err) => {
        this.sfBrowseError = err.error?.error || 'Could not browse server folders. Is saved_datasets_dir configured?';
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

  sfOnMediaTypeChange(mediaType: string): void {
    this.sfMediaType = mediaType;
    this.sfLoadEmbedders(mediaType);
    this.sfLoadClippers(mediaType);
  }

  private sfLoadEmbedders(mediaType: string): void {
    if (!mediaType) {
      this.sfEmbedders = [];
      this.sfSelectedEmbedder = '';
      return;
    }
    this.datasetsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.sfEmbedders = embedders;
        // Prefer guessed embedder when it's available for this media type
        const guessedMatch = this.guessedMediaEmbedder
          ? embedders.find((e) => e.name === this.guessedMediaEmbedder)
          : null;
        this.sfSelectedEmbedder = guessedMatch ? guessedMatch.name : (embedders.length > 0 ? embedders[0].name : '');
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

  openClipperChooser(context: 'form' | 'demo' | 'sf'): void {
    this.clipperChooserContext = context;
    if (context === 'form') {
      this.clipperChooserClippers = this.availableClippers;
    } else if (context === 'demo') {
      this.clipperChooserClippers = this.demoClippers;
    } else {
      this.clipperChooserClippers = this.sfClippers;
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
    } else {
      this.sfSelectedClipper = selection.name;
      this.sfClipperParamValues = { ...selection.params };
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
    } else {
      this.sfSelectedClipper = defaultName;
      this.sfResetClipperParams();
    }
  }

  /** Display name for the currently selected clipper in a given context. */
  clipperDisplayName(context: 'form' | 'demo' | 'sf'): string {
    let clippers: ClipperInfo[];
    let selected: string;
    if (context === 'form') {
      clippers = this.availableClippers;
      selected = this.selectedClipper;
    } else if (context === 'demo') {
      clippers = this.demoClippers;
      selected = this.selectedDemoClipper;
    } else {
      clippers = this.sfClippers;
      selected = this.sfSelectedClipper;
    }
    const clipper = clippers.find((c) => c.name === selected);
    return clipper?.display_name || clipper?.name || 'Default';
  }

  sfSubmit(): void {
    this.sfSubmitting = true;
    this.sfBrowseError = '';

    const params: Record<string, unknown> = {
      path: this.sfAbsolutePath,
      media_type: this.sfMediaType,
    };
    if (this.sfSelectedEmbedder) {
      params['embedder'] = this.sfSelectedEmbedder;
    }
    if (this.sfSelectedClipper) {
      params['clipper'] = this.sfSelectedClipper;
      if (this.sfClipperParams.length > 0 && Object.keys(this.sfClipperParamValues).length > 0) {
        params['clipper_params'] = { ...this.sfClipperParamValues };
      }
    }

    this.datasetsApi.runImporter('folder', params).subscribe({
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
