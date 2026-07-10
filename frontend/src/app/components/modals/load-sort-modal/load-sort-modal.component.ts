import { ChangeDetectionStrategy, Component, inject, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetsUiApiService } from '../../../services/datasets-ui-api.service';
import { DetectorsRegistryApiService } from '../../../services/detectors-registry-api.service';
import { ImporterInfo } from '../../../models/api.models';
import { DetectorRegistryEntry } from '../../../generated/api-client/models/detector-registry-entry';
import type { ServerMediaFileEntry } from '../../../generated/api-client/models/server-media-file-entry';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../media-crop-modal/media-crop-modal.component';

interface BrowseItem {
  key: string;
  display: string;
}

type MediaPickerView = 'sources' | 'browse-items' | 'file-browser';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-load-sort-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, MediaCropModalComponent],
  templateUrl: './load-sort-modal.component.html',
  styleUrl: './load-sort-modal.component.scss',
})
export class LoadSortModalComponent implements OnInit {
  private sortingApi = inject(SortingApiService);
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);

  readonly closed = output<void>();
  readonly modelSelected = output<string>();
  readonly exampleSortStarted = output<unknown>();

  readonly serverMediaFiles = signal<ServerMediaFileEntry[]>([]);
  readonly registryModels = signal<DetectorRegistryEntry[]>([]);
  readonly loading = signal(true);
  readonly status = signal('');
  readonly error = signal('');

  // Media source browser state
  readonly showMediaPicker = signal(false);
  readonly mediaSources = signal<ImporterInfo[]>([]);
  selectedSource: ImporterInfo | null = null;
  readonly mediaPickerView = signal<MediaPickerView>('sources');
  readonly browseItems = signal<BrowseItem[]>([]);
  readonly browseLoading = signal(false);

  // Typed-path state for the example-media picker (demo & folder drill-down).
  // The user types a path relative to the picked source; the server validates
  // it when the form is submitted.
  browseSource = '';
  browseSourceLabel = '';
  readonly fileLoading = signal(false);
  typedPath = '';
  readonly typedPathError = signal('');

  // Pending crop confirmation state.
  pendingFile: File | null = null;
  pendingFileMediaType = '';
  pendingServerFilename = '';
  pendingOrigin: { origin: Record<string, unknown>; key: string } | null = null;

  ngOnInit(): void {
    this.sortingApi.getServerMediaFiles().subscribe({
      next: (res) => {
        this.serverMediaFiles.set(res.files || []);
      },
    });
    this.detectorsRegistryApi.getRegistry().subscribe({
      next: (res) => {
        // Show detectors that have at least one training label.
        this.registryModels.set(
          (res.detectors || []).filter((m) => (m.num_training ?? 0) > 0),
        );
        this.loading.set(false);
      },
      error: () => {
        this.loading.set(false);
      },
    });
  }

  // --- Model loading ---

  loadRegistryModel(model: DetectorRegistryEntry): void {
    this.status.set('Loading detector…');
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
    this.status.set('Scoring with example media...');
    this.sortingApi.exampleSort(result.file, result.cropParams).subscribe({
      next: (data) => {
        this.status.set('');
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status.set('');
        this.error.set('Example sort failed');
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
    this.status.set('Scoring with example media...');
    this.sortingApi.exampleSortServer({ filenames: [filename] }).subscribe({
      next: (data) => {
        this.status.set('');
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status.set('');
        this.error.set('Example sort failed');
      },
    });
  }

  // --- Media source browser ---

  openMediaPicker(): void {
    this.showMediaPicker.set(true);
    this.selectedSource = null;
    this.mediaPickerView.set('sources');
    this.browseItems.set([]);
    this.browseSource = '';
    this.browseSourceLabel = '';
    this.loadMediaSources();
  }

  closeMediaPicker(): void {
    this.showMediaPicker.set(false);
  }

  private loadMediaSources(): void {
    this.browseLoading.set(true);
    this.datasetsCrudApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaSources.set(
          (res.importers || []).filter(
            (imp) =>
              imp.name === 'demo' ||
              imp.name === 'server_folder' ||
              (!imp['hidden_from_picker'] && imp.name !== 'combine_datasets'),
          ),
        );
        this.browseLoading.set(false);
      },
      error: () => {
        this.browseLoading.set(false);
      },
    });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseLoading.set(true);
    this.browseItems.set([]);

    if (source.name === 'demo') {
      this.datasetsListingsApi.getDemoList().subscribe({
        next: (res) => {
          this.browseItems.set(
            (res.datasets || []).map((d) => ({
              key: d.name,
              display: `${d.label} (${d.media_type}, ${d.num_files} items)`,
            })),
          );
          this.mediaPickerView.set('browse-items');
          this.browseLoading.set(false);
        },
        error: () => {
          this.browseLoading.set(false);
        },
      });
    } else if (source.name === 'server_folder') {
      this.browseLoading.set(false);
      this.startFileBrowsing('folder', 'Saved Datasets');
    } else {
      this.sortingApi.getServerMediaFiles().subscribe({
        next: (res) => {
          this.browseItems.set(
            (res.files || []).map((f) => ({
              key: f.filename || f.name,
              display: f.name,
            })),
          );
          this.mediaPickerView.set('browse-items');
          this.browseLoading.set(false);
        },
        error: () => {
          this.browseLoading.set(false);
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
    this.showMediaPicker.set(false);
    this.loadServerMedia(item.key);
  }

  private startFileBrowsing(source: string, label: string): void {
    this.mediaPickerView.set('file-browser');
    this.browseSource = source;
    this.browseSourceLabel = label;
    this.typedPath = '';
    this.typedPathError.set('');
  }

  /** Back-arrow handler for the typed-path view: returns to the
   *  source-listing step that opened it (demo list / saved-datasets
   *  list). */
  backFromFileBrowser(): void {
    this.mediaPickerView.set(this.selectedSource?.name === 'demo' ? 'browse-items' : 'sources');
    this.browseSource = '';
    this.browseSourceLabel = '';
    this.typedPath = '';
    this.typedPathError.set('');
  }

  /** Submit the typed path to ``/api/browse-media-files/select`` and
   *  kick off an example sort. The server validates the path. */
  submitTypedPath(): void {
    const raw = (this.typedPath || '').trim();
    if (!raw) return;
    this.typedPathError.set('');
    this.fileLoading.set(true);
    this.datasetsUiApi.selectBrowsedFile(this.browseSource, raw).subscribe({
      next: (res) => {
        this.fileLoading.set(false);
        this.showMediaPicker.set(false);
        this.loadServerMedia(res.filename);
      },
      error: (err) => {
        this.fileLoading.set(false);
        this.typedPathError.set(err?.error?.message || 'Path not found on the server.');
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
