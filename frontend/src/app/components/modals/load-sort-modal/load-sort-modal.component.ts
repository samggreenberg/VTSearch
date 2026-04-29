import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { ServerFileEntry, ModelRegistryEntry, ImporterInfo } from '../../../models/api.models';

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

type MediaPickerView = 'sources' | 'browse-items' | 'file-browser';

@Component({
  selector: 'vt-load-sort-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, IconComponent],
  templateUrl: './load-sort-modal.component.html',
  styleUrl: './load-sort-modal.component.scss',
})
export class LoadSortModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() detectorLoaded = new EventEmitter<unknown>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();

  serverDetectors: ServerFileEntry[] = [];
  serverMediaFiles: ServerFileEntry[] = [];
  registryModels: ModelRegistryEntry[] = [];
  loading = true;
  status = '';
  error = '';

  // Media source browser state
  showMediaPicker = false;
  mediaSources: ImporterInfo[] = [];
  selectedSource: ImporterInfo | null = null;
  mediaPickerView: MediaPickerView = 'sources';
  browseItems: BrowseItem[] = [];
  browseLoading = false;

  // File browser state (for demo & folder drill-down)
  browseSource = '';
  browsePath: string[] = [];
  browseEntries: BrowseEntry[] = [];
  fileLoading = false;

  constructor(
    private detectorsApi: DetectorsApiService,
    private sortingApi: SortingApiService,
    private datasetsApi: DatasetsApiService,
    private modelsApi: TrainableModelsApiService,
  ) {}

  ngOnInit(): void {
    this.detectorsApi.getServerFiles().subscribe({
      next: (res) => {
        this.serverDetectors = res.files || [];
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      },
    });
    this.sortingApi.getServerMediaFiles().subscribe({
      next: (res) => {
        this.serverMediaFiles = res.files || [];
      },
    });
    this.modelsApi.getRegistry().subscribe({
      next: (res) => {
        // Show models that have been trained (have a detector with weights)
        this.registryModels = (res.models || []).filter(
          (m) => m.detector_name && (m.num_training ?? 0) > 0,
        );
      },
    });
  }

  // --- Detector loading ---

  loadServerDetector(name: string): void {
    this.status = 'Loading server detector...';
    this.detectorsApi.getServerFile(name).subscribe({
      next: (data) => {
        this.status = '';
        this.detectorLoaded.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Failed to load detector';
      },
    });
  }

  loadRegistryModel(model: ModelRegistryEntry): void {
    if (!model.detector_name) return;
    this.status = 'Loading model detector...';
    this.detectorsApi.exportDetector(model.detector_name).subscribe({
      next: (data) => {
        this.status = '';
        const detector = data as Record<string, unknown>;
        if (!detector['name']) {
          detector['name'] = model.name;
        }
        this.detectorLoaded.emit(detector);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Failed to load model detector';
      },
    });
  }

  // --- Example media loading ---

  onMediaFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    this.status = 'Scoring with example media...';
    this.sortingApi.exampleSort(file).subscribe({
      next: (data) => {
        this.status = '';
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Example sort failed';
      },
    });
  }

  loadServerMedia(filename: string): void {
    this.status = 'Scoring with example media...';
    this.sortingApi.exampleSortServer({ filename }).subscribe({
      next: (data) => {
        this.status = '';
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Example sort failed';
      },
    });
  }

  // --- Media source browser ---

  openMediaPicker(): void {
    this.showMediaPicker = true;
    this.selectedSource = null;
    this.mediaPickerView = 'sources';
    this.browseItems = [];
    this.browseEntries = [];
    this.browsePath = [];
    this.browseSource = '';
    this.loadMediaSources();
  }

  closeMediaPicker(): void {
    this.showMediaPicker = false;
  }

  private loadMediaSources(): void {
    this.browseLoading = true;
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaSources = (res.importers || []).filter(
          (imp) =>
            imp.name === 'demo' ||
            imp.name === 'server_folder' ||
            (!imp['hidden_from_picker'] && imp.name !== 'combine_datasets'),
        );
        this.browseLoading = false;
      },
      error: () => {
        this.browseLoading = false;
      },
    });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseLoading = true;
    this.browseItems = [];

    if (source.name === 'demo') {
      this.datasetsApi.getDemoList().subscribe({
        next: (res) => {
          this.browseItems = (res.datasets || []).map((d) => ({
            key: d.name,
            display: `${d.label} (${d.media_type}, ${d.num_files} items)`,
          }));
          this.mediaPickerView = 'browse-items';
          this.browseLoading = false;
        },
        error: () => {
          this.browseLoading = false;
        },
      });
    } else if (source.name === 'server_folder') {
      this.browseLoading = false;
      this.startFileBrowsing('folder', 'Saved Datasets');
    } else {
      this.sortingApi.getServerMediaFiles().subscribe({
        next: (res) => {
          this.browseItems = (res.files || []).map((f) => ({
            key: f.filename || f.name,
            display: f.name,
          }));
          this.mediaPickerView = 'browse-items';
          this.browseLoading = false;
        },
        error: () => {
          this.browseLoading = false;
        },
      });
    }
  }

  selectBrowseItem(item: BrowseItem): void {
    if (this.selectedSource?.name === 'demo') {
      this.startFileBrowsing(`demo:${item.key}`, item.display);
      return;
    }
    // For server media files, sort directly
    this.showMediaPicker = false;
    this.loadServerMedia(item.key);
  }

  private startFileBrowsing(source: string, label: string): void {
    this.mediaPickerView = 'file-browser';
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
      this.mediaPickerView = 'browse-items';
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
        this.fileLoading = false;
        this.showMediaPicker = false;
        this.loadServerMedia(res.filename);
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

  close(): void {
    this.closed.emit();
  }
}
