import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { map } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import {
  FolderBrowserBrowseFn,
  FolderBrowserComponent,
  FolderBrowserFileEntry,
} from '../../folder-browser/folder-browser.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { DetectorRegistryEntry, ImporterInfo } from '../../../models/api.models';
import type { ServerMediaFileEntry } from '../../../generated/api-client/models/server-media-file-entry';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../media-crop-modal/media-crop-modal.component';
import { DetectorSwatchComponent } from '../../detector-swatch/detector-swatch.component';

interface BrowseItem {
  key: string;
  display: string;
}

type MediaPickerView = 'sources' | 'browse-items' | 'file-browser';

@Component({
  selector: 'vt-load-sort-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, IconComponent, FolderBrowserComponent, MediaCropModalComponent, DetectorSwatchComponent],
  templateUrl: './load-sort-modal.component.html',
  styleUrl: './load-sort-modal.component.scss',
})
export class LoadSortModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() modelSelected = new EventEmitter<string>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();

  serverMediaFiles: ServerMediaFileEntry[] = [];
  registryModels: DetectorRegistryEntry[] = [];
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
  browseSourceLabel = '';
  fileLoading = false;

  /** Browse fn for the embedded ``<vt-folder-browser>``.  Routes through
   *  ``/api/browse-media-files`` with whichever source the user picked
   *  (``demo:<name>`` or ``folder``). */
  readonly browseFn: FolderBrowserBrowseFn = (path: string) =>
    this.datasetsApi.browseMediaFiles(this.browseSource, path).pipe(
      map((res) => ({
        directories: res.directories || [],
        files: res.files || [],
        rootPath: res.root_path,
        currentPath: path,
      })),
    );

  // Pending crop confirmation state.
  pendingFile: File | null = null;
  pendingFileMediaType = '';
  pendingServerFilename = '';
  pendingOrigin: { origin: Record<string, unknown>; key: string } | null = null;

  constructor(
    private sortingApi: SortingApiService,
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
  ) {}

  ngOnInit(): void {
    this.sortingApi.getServerMediaFiles().subscribe({
      next: (res) => {
        this.serverMediaFiles = res.files || [];
      },
    });
    this.detectorsApi.getRegistry().subscribe({
      next: (res) => {
        // Show detectors that have at least one training label.
        this.registryModels = (res.detectors || []).filter(
          (m) => (m.num_training ?? 0) > 0,
        );
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      },
    });
  }

  // --- Model loading ---

  loadRegistryModel(model: DetectorRegistryEntry): void {
    this.status = 'Loading detector…';
    this.modelSelected.emit(model.id);
    this.closed.emit();
  }

  // --- Example media loading ---

  onMediaFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    input.value = '';
    this.pendingFile = file;
    this.pendingFileMediaType = this.mediaTypeFromFile(file);
  }

  onCropConfirmed(result: MediaCropResult): void {
    this.pendingFile = null;
    this.status = 'Scoring with example media...';
    this.sortingApi.exampleSort(result.file, result.cropParams).subscribe({
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

  onCropCancelled(): void {
    this.pendingFile = null;
  }

  private mediaTypeFromFile(file: File): string {
    // Map MIME type back to a vtsearch media_type so the crop modal knows
    // which overlay to render.  Falls back to "" (no cropping offered).
    const m = (file.type || '').toLowerCase();
    if (m.startsWith('image/')) return 'image';
    if (m.startsWith('audio/')) return 'audio';
    if (m.startsWith('video/')) return 'video';
    return '';
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
    this.browseSource = '';
    this.browseSourceLabel = '';
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
    this.browseSourceLabel = label;
  }

  /** Back-arrow handler for the file-browser view — returns to the
   *  source-listing step that opened it (demo list / saved-datasets
   *  list). */
  backFromFileBrowser(): void {
    this.mediaPickerView = this.selectedSource?.name === 'demo' ? 'browse-items' : 'sources';
    this.browseSource = '';
    this.browseSourceLabel = '';
  }

  /** Confirm handler for the file browser — materialises the picked
   *  file via ``/api/browse-media-files/select`` and kicks off an
   *  example sort. */
  onFileConfirmed(entry: FolderBrowserFileEntry): void {
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
