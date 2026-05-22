import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
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
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent],
  templateUrl: './resort-prompt-modal.component.html',
  styleUrl: './resort-prompt-modal.component.scss',
})
export class ResortPromptModalComponent {
  @Input() currentExampleType: 'text' | 'media' = 'text';
  @Input() currentExampleDisplay = '';
  @Input() keepLabelsCount = 0;
  @Output() closed = new EventEmitter<void>();
  @Output() keepExample = new EventEmitter<void>();
  @Output() newExample = new EventEmitter<ResortResult>();

  view: ModalView = 'prompt';
  pendingText = '';
  error = '';

  // Media picker state
  mediaSources: ImporterInfo[] = [];
  selectedSource: ImporterInfo | null = null;
  browseItems: BrowseItem[] = [];
  browseLoading = false;
  browseSource = '';
  browseSourceLabel = '';
  fileBrowsing = false;
  fileLoading = false;
  typedPath = '';
  typedPathError = '';

  constructor(
    private datasetsApi: DatasetsApiService,
    private sortingApi: SortingApiService,
  ) {}

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
    this.browseItems = [];
    this.fileBrowsing = false;
    this.browseSource = '';
    this.browseSourceLabel = '';
    this.typedPath = '';
    this.typedPathError = '';
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaSources = (res.importers || []).filter(
          (imp) =>
            imp.name === 'demo' ||
            imp.name === 'server_folder' ||
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
    this.newExample.emit({ action: 'new-example', type: 'media', value: item.key });
  }

  private startFileBrowsing(source: string, label: string): void {
    this.fileBrowsing = true;
    this.browseSource = source;
    this.browseSourceLabel = label;
    this.typedPath = '';
    this.typedPathError = '';
  }

  submitTypedPath(): void {
    const raw = (this.typedPath || '').trim();
    if (!raw) return;
    this.fileLoading = true;
    this.typedPathError = '';
    this.datasetsApi.selectBrowsedFile(this.browseSource, raw).subscribe({
      next: (res) => {
        this.fileLoading = false;
        this.newExample.emit({ action: 'new-example', type: 'media', value: res.filename });
      },
      error: (err) => {
        this.fileLoading = false;
        this.typedPathError = err?.error?.message || 'Path not found on the server.';
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
        this.error = 'Failed to upload file';
      },
    });
    input.value = '';
  }

  backToPrompt(): void {
    this.view = 'prompt';
  }
}
