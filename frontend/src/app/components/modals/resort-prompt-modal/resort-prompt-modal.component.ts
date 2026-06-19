import { Component, Input, inject, input, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetsUiApiService } from '../../../services/datasets-ui-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ImporterInfo } from '../../../models/api.models';

interface BrowseItem {
  key: string;
  display: string;
}

export interface ResortResult {
  action: 'new-example';
  type: 'text' | 'media';
  value: string;
}

type ModalView = 'prompt' | 'media-picker';

@Component({
  selector: 'vt-resort-prompt-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent],
  templateUrl: './resort-prompt-modal.component.html',
  styleUrl: './resort-prompt-modal.component.scss',
})
export class ResortPromptModalComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
  private sortingApi = inject(SortingApiService);

  readonly currentExampleType = input<'text' | 'media'>('text');
  readonly currentExampleDisplay = input('');
  @Input() keepLabelsCount = 0;
  readonly closed = output<void>();
  readonly keepExample = output<void>();
  readonly newExample = output<ResortResult>();

  view: ModalView = 'prompt';
  pendingText = '';
  readonly error = signal('');

  // Media picker state
  readonly mediaSources = signal<ImporterInfo[]>([]);
  selectedSource: ImporterInfo | null = null;
  readonly browseItems = signal<BrowseItem[]>([]);
  readonly browseLoading = signal(false);
  browseSource = '';
  browseSourceLabel = '';
  fileBrowsing = false;
  readonly fileLoading = signal(false);
  typedPath = '';
  readonly typedPathError = signal('');

  onKeep(): void {
    this.keepExample.emit();
  }

  // --- Text example ---

  submitText(): void {
    const text = this.pendingText.trim();
    if (!text) return;
    this.newExample.emit({ action: 'new-example', type: 'text', value: text });
  }

  // --- Media picker ---

  openMediaPicker(): void {
    this.view = 'media-picker';
    this.selectedSource = null;
    this.browseItems.set([]);
    this.fileBrowsing = false;
    this.browseSource = '';
    this.browseSourceLabel = '';
    this.typedPath = '';
    this.typedPathError.set('');
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
      },
    });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseLoading.set(true);
    this.browseItems.set([]);
    this.fileBrowsing = false;

    if (source.name === 'demo') {
      this.datasetsListingsApi.getDemoList().subscribe({
        next: (res) => {
          this.browseItems.set(
            (res.datasets || []).map((d) => ({
              key: d.name,
              display: `${d.label} (${d.media_type}, ${d.num_files} items)`,
            })),
          );
          this.browseLoading.set(false);
        },
        error: () => { this.browseLoading.set(false); },
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
          this.browseLoading.set(false);
        },
        error: () => { this.browseLoading.set(false); },
      });
    }
  }

  selectBrowseItem(item: BrowseItem): void {
    if (this.selectedSource?.name === 'demo') {
      this.startFileBrowsing(`demo:${item.key}`, item.display);
      return;
    }
    this.newExample.emit({ action: 'new-example', type: 'media', value: item.key });
  }

  private startFileBrowsing(source: string, label: string): void {
    this.fileBrowsing = true;
    this.browseSource = source;
    this.browseSourceLabel = label;
    this.typedPath = '';
    this.typedPathError.set('');
  }

  submitTypedPath(): void {
    const raw = (this.typedPath || '').trim();
    if (!raw) return;
    this.fileLoading.set(true);
    this.typedPathError.set('');
    this.datasetsUiApi.selectBrowsedFile(this.browseSource, raw).subscribe({
      next: (res) => {
        this.fileLoading.set(false);
        this.newExample.emit({ action: 'new-example', type: 'media', value: res.filename });
      },
      error: (err) => {
        this.fileLoading.set(false);
        this.typedPathError.set(err?.error?.message || 'Path not found on the server.');
      },
    });
  }

  onLocalFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    this.sortingApi.uploadServerMediaFile(file).subscribe({
      next: (res) => {
        this.newExample.emit({ action: 'new-example', type: 'media', value: res.filename });
      },
      error: () => {
        this.error.set('Failed to upload file');
      },
    });
    input.value = '';
  }

  backToPrompt(): void {
    this.view = 'prompt';
  }
}
