import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ImporterInfo, MediaTypeInfo } from '../../../models/api.models';

interface ModelExample {
  type: 'text' | 'media';
  value: string;
  display: string;
}

interface BrowseItem {
  key: string;
  display: string;
}

interface BrowseEntry {
  name: string;
  path: string;
  size_bytes?: number;
  isDir: boolean;
}

type ModalView = 'main' | 'media-picker';

@Component({
  selector: 'vt-new-model-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './new-model-modal.component.html',
  styleUrl: './new-model-modal.component.scss',
})
export class NewModelModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() created = new EventEmitter<void>();

  view: ModalView = 'main';
  name = '';
  mediaType = 'audio';
  pendingText = '';
  mediaTypes: string[] = [];
  mediaTypeInfos: MediaTypeInfo[] = [];
  examples: ModelExample[] = [];
  submitting = false;
  error = '';

  // Media picker state
  mediaSources: ImporterInfo[] = [];
  selectedSource: ImporterInfo | null = null;
  browseItems: BrowseItem[] = [];
  browseLoading = false;

  // File browser state (for demo & folder drill-down)
  browseSource = '';          // e.g. "demo:esc50_s" or "folder"
  browsePath: string[] = [];  // breadcrumb segments
  browseEntries: BrowseEntry[] = [];
  fileBrowsing = false;
  fileLoading = false;

  constructor(
    private modelsApi: TrainableModelsApiService,
    private datasetsApi: DatasetsApiService,
    private sortingApi: SortingApiService,
  ) {}

  ngOnInit(): void {
    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypeInfos = res.media_types || [];
        this.mediaTypes = this.mediaTypeInfos.map((t) => t.type_id || t.name);
      },
    });
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

  // --- Text examples ---

  addTextExample(): void {
    const text = this.pendingText.trim();
    if (!text) return;
    this.examples.push({ type: 'text', value: text, display: text });
    this.pendingText = '';
  }

  // --- Media examples ---

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
      // List demo datasets — each one becomes a browsable source
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
      // Go straight into the file browser for saved_datasets_dir
      this.browseLoading = false;
      this.startFileBrowsing('folder', 'Saved Datasets');
    } else {
      // For other importers, list server media files (already individual files)
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
    // For demo sources, drill down into the demo's files
    if (this.selectedSource?.name === 'demo') {
      this.startFileBrowsing(`demo:${item.key}`, item.display);
      return;
    }

    // For other sources (server media files), selecting adds it directly
    this.examples.push({
      type: 'media',
      value: item.key,
      display: item.display || item.key,
    });
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
          entries.push({ name: d.name, path: d.path, isDir: true });
        }
        for (const f of res.files || []) {
          entries.push({ name: f.name, path: f.path, size_bytes: f.size_bytes, isDir: false });
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
      // Go back to source/demo list
      this.fileBrowsing = false;
      this.browseEntries = [];
      this.browsePath = [];
      return;
    }
    // Rebuild the relative path from breadcrumb segments (skip the root label)
    this.browsePath = this.browsePath.slice(0, index + 1);
    const relPath = this.browsePath.slice(1).join('/');
    this.loadDirectory(relPath);
  }

  selectFile(entry: BrowseEntry): void {
    // Copy the file to example_media via the select endpoint
    this.fileLoading = true;
    this.datasetsApi.selectBrowsedFile(this.browseSource, entry.path).subscribe({
      next: (res) => {
        this.examples.push({
          type: 'media',
          value: res.filename,
          display: res.original_name || entry.name,
        });
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
        this.examples.push({
          type: 'media',
          value: res.filename,
          display: res.original_name || res.filename,
        });
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

  // --- Examples grid ---

  removeExample(index: number): void {
    this.examples.splice(index, 1);
  }

  // --- Submit ---

  submit(): void {
    const trimmedName = this.name.trim();
    if (!trimmedName) {
      this.error = 'Name is required';
      return;
    }
    if (this.examples.length === 0) {
      this.error = 'At least one example (text or media) is required';
      return;
    }

    this.submitting = true;
    this.error = '';

    // Derive text_query and media_example from first example for autopilot
    const firstExample = this.examples[0];
    const textQuery = firstExample.type === 'text' ? firstExample.value : '';
    const mediaExample = firstExample.type === 'media' ? firstExample.value : '';

    const examplesPayload = this.examples.map((ex) => ({
      type: ex.type,
      value: ex.value,
    }));

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
      return `${mt.icon || ''} ${mt.tab_title || mt.name}`.trim();
    }
    return typeId;
  }

  close(): void {
    this.closed.emit();
  }
}
