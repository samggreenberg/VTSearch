import { Component, EventEmitter, HostListener, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ImporterInfo, MediaTypeInfo } from '../../../models/api.models';

interface BrowseItem {
  key: string;
  display: string;
}

interface BrowseEntry {
  name: string;
  path: string;
  size_bytes?: number;
  modified_at?: string;
  isDir: boolean;
}

type ModalView = 'main' | 'media-picker';

@Component({
  selector: 'vt-new-model-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent],
  templateUrl: './new-model-modal.component.html',
  styleUrl: './new-model-modal.component.scss',
})
export class NewModelModalComponent implements OnInit {
  /** Media type of the currently active dataset, if any. */
  @Input() defaultMediaType = '';

  @Output() closed = new EventEmitter<void>();
  @Output() created = new EventEmitter<void>();

  view: ModalView = 'main';
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

  // Media picker state
  mediaSources: ImporterInfo[] = [];
  selectedSource: ImporterInfo | null = null;
  browseItems: BrowseItem[] = [];
  browseLoading = false;

  // File browser state (for demo & folder drill-down)
  browseSource = '';
  browsePath: string[] = [];
  browseEntries: BrowseEntry[] = [];
  fileBrowsing = false;
  fileLoading = false;

  constructor(
    private modelsApi: TrainableModelsApiService,
    private datasetsApi: DatasetsApiService,
    private sortingApi: SortingApiService,
  ) {}

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    if (this.mediaTypeDropdownOpen && !target.closest('.custom-select')) {
      this.mediaTypeDropdownOpen = false;
    }
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

  get hasExample(): boolean {
    return this.exampleType === 'media' || !!this.pendingText.trim();
  }

  get hasMediaExample(): boolean {
    return this.exampleType === 'media';
  }

  get hasPendingText(): boolean {
    return !!this.pendingText.trim();
  }

  get canSubmit(): boolean {
    return !!this.name.trim() && this.hasExample && !this.submitting;
  }

  // --- Media example ---

  openMediaPicker(): void {
    this.view = 'media-picker';
    this.selectedSource = null;
    this.browseItems = [];
    this.fileBrowsing = false;
    this.browseEntries = [];
    this.browsePath = [];
    this.browseSource = '';
    this.loadMediaSources();
  }

  private loadMediaSources(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaSources = (res.importers || []).filter(
          (imp) =>
            imp.name === 'demo' ||
            imp.name === 'folder' ||
            (!imp['hidden_from_picker'] && imp.name !== 'combine_datasets'),
        );
      },
    });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseLoading = true;
    this.browseItems = [];
    this.fileBrowsing = false;

    if (source.name === 'demo') {
      this.datasetsApi.getDemoList().subscribe({
        next: (res) => {
          this.browseItems = (res.datasets || []).map((d) => ({
            key: d.name,
            display: `${d.label} (${d.media_type}, ${d.num_files} items)`,
          }));
          this.browseLoading = false;
        },
        error: () => { this.browseLoading = false; },
      });
    } else if (source.name === 'folder') {
      this.browseLoading = false;
      this.startFileBrowsing('folder', 'Saved Datasets');
    } else {
      this.sortingApi.getServerMediaFiles().subscribe({
        next: (res) => {
          this.browseItems = (res.files || []).map((f) => ({
            key: f.filename || f.name,
            display: f.name,
          }));
          this.browseLoading = false;
        },
        error: () => { this.browseLoading = false; },
      });
    }
  }

  selectBrowseItem(item: BrowseItem): void {
    if (this.selectedSource?.name === 'demo') {
      this.startFileBrowsing(`demo:${item.key}`, item.display);
      return;
    }

    // For other sources (server media files), selecting sets the example
    this.exampleType = 'media';
    this.exampleValue = item.key;
    this.exampleDisplay = item.display || item.key;
    this.pendingText = '';
    this.view = 'main';
  }

  // --- Recursive file browser ---

  private startFileBrowsing(source: string, label: string): void {
    this.fileBrowsing = true;
    this.browseSource = source;
    this.browsePath = [label];
    this.loadDirectory('');
  }

  private loadDirectory(relPath: string): void {
    this.fileLoading = true;
    this.browseEntries = [];

    this.datasetsApi.browseMediaFiles(this.browseSource, relPath).subscribe({
      next: (res) => {
        const entries: BrowseEntry[] = [];
        for (const d of res.directories || []) {
          entries.push({ name: d.name, path: d.path, modified_at: d.modified_at, isDir: true });
        }
        for (const f of res.files || []) {
          entries.push({ name: f.name, path: f.path, size_bytes: f.size_bytes, modified_at: f.modified_at, isDir: false });
        }
        this.browseEntries = entries;
        this.fileLoading = false;
      },
      error: () => {
        this.fileLoading = false;
      },
    });
  }

  enterDirectory(entry: BrowseEntry): void {
    this.browsePath.push(entry.name);
    this.loadDirectory(entry.path);
  }

  navigateBreadcrumb(index: number): void {
    if (index === 0) {
      this.fileBrowsing = false;
      this.browseEntries = [];
      this.browsePath = [];
      return;
    }
    this.browsePath = this.browsePath.slice(0, index + 1);
    const relPath = this.browsePath.slice(1).join('/');
    this.loadDirectory(relPath);
  }

  selectFile(entry: BrowseEntry): void {
    this.fileLoading = true;
    this.datasetsApi.selectBrowsedFile(this.browseSource, entry.path).subscribe({
      next: (res) => {
        this.exampleType = 'media';
        this.exampleValue = res.filename;
        this.exampleDisplay = res.original_name || entry.name;
        this.pendingText = '';
        this.fileLoading = false;
        this.view = 'main';
      },
      error: () => {
        this.error = 'Failed to select file';
        this.fileLoading = false;
      },
    });
  }

  formatSize(bytes?: number): string {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  onLocalFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    this.sortingApi.uploadServerMediaFile(file).subscribe({
      next: (res) => {
        this.exampleType = 'media';
        this.exampleValue = res.filename;
        this.exampleDisplay = res.original_name || res.filename;
        this.pendingText = '';
      },
      error: () => {
        this.error = 'Failed to upload file';
      },
    });
    input.value = '';
  }

  backToMain(): void {
    this.view = 'main';
  }

  // --- Clear example ---

  clearExample(): void {
    this.exampleType = null;
    this.exampleValue = '';
    this.exampleDisplay = '';
    this.pendingText = '';
  }

  // --- Submit ---

  submit(): void {
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
        trainable: true,
        text_query: textQuery,
        media_example: mediaExample,
        examples: examplesPayload,
      })
      .subscribe({
        next: () => {
          this.submitting = false;
          this.created.emit();
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
