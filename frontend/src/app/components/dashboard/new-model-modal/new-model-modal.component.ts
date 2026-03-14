import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ImporterInfo } from '../../../models/api.models';

interface ModelExample {
  type: 'text' | 'media';
  value: string;
  display: string;
}

interface BrowseItem {
  key: string;
  display: string;
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
  examples: ModelExample[] = [];
  submitting = false;
  error = '';

  // Media picker state
  mediaSources: ImporterInfo[] = [];
  selectedSource: ImporterInfo | null = null;
  browseItems: BrowseItem[] = [];
  browseLoading = false;

  constructor(
    private modelsApi: TrainableModelsApiService,
    private datasetsApi: DatasetsApiService,
    private sortingApi: SortingApiService,
  ) {}

  ngOnInit(): void {
    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes = (res.media_types || []).map((t) => t.type_id || t.name);
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
    this.loadMediaSources();
  }

  private loadMediaSources(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.mediaSources = (res.importers || []).filter(
          (imp) => !imp['hidden_from_picker'],
        );
      },
    });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseLoading = true;
    this.browseItems = [];

    // For demo importer, list demo datasets
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
      // List available files on the server
      this.datasetsApi.getAvailableFiles().subscribe({
        next: (res) => {
          this.browseItems = (res.files || []).map((f) => ({
            key: f.path || f.name,
            display: `${f.name} (${f.size_mb?.toFixed(1) || '?'} MB)`,
          }));
          this.browseLoading = false;
        },
        error: () => { this.browseLoading = false; },
      });
    } else {
      // For other importers, list server media files as a fallback
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
    this.examples.push({
      type: 'media',
      value: item.key,
      display: item.display || item.key,
    });
    this.view = 'main';
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

  close(): void {
    this.closed.emit();
  }
}
