import { ChangeDetectionStrategy, Component, inject, input, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasourceImportFormComponent } from '../../datasource-import-form/datasource-import-form.component';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetsUiApiService } from '../../../services/datasets-ui-api.service';
import {
  DatasourceImportersApiService,
  DatasourceImportResult,
} from '../../../services/datasource-importers-api.service';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-resort-prompt-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, DatasourceImportFormComponent],
  templateUrl: './resort-prompt-modal.component.html',
  styleUrl: './resort-prompt-modal.component.scss',
})
export class ResortPromptModalComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
  private datasourceImportersApi = inject(DatasourceImportersApiService);
  private sortingApi = inject(SortingApiService);

  readonly currentExampleType = input<'text' | 'media'>('text');
  readonly currentExampleDisplay = input('');
  readonly keepLabelsCount = input(0);
  readonly closed = output<void>();
  readonly keepExample = output<void>();
  readonly newExample = output<ResortResult>();

  view: ModalView = 'prompt';
  pendingText = '';
  readonly error = signal('');

  // Media picker state
  readonly mediaSources = signal<ImporterInfo[]>([]);
  /** Single-item fetchers (server file, URL download, third-party plugins)
   *  rendered as a dynamic form instead of a browse list. */
  readonly datasourceImporters = signal<ImporterInfo[]>([]);
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
    this.clearFileBrowsing();
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
    this.datasourceImportersApi.list().subscribe({
      next: (res) => {
        this.datasourceImporters.set(
          (res.importers || []).filter((imp) => !imp['hidden_from_picker']),
        );
      },
    });
  }

  /** Every option the picker offers: dataset-importer browse views first,
   *  then the single-item datasource importers. */
  get allSources(): ImporterInfo[] {
    return [...this.mediaSources(), ...this.datasourceImporters()];
  }

  /** True when *source* is a datasource importer (a single-item fetcher
   *  rendered as a dynamic form) rather than a browse view.  Identity
   *  check against the datasource list, so a name shared across the two
   *  families can't misroute. */
  isDatasourceImporter(source: ImporterInfo | null): boolean {
    return source != null && this.datasourceImporters().includes(source);
  }

  /** The selected source when it is a datasource importer, else null.
   *  Template convenience for rendering the dynamic form view. */
  get selectedDatasourceImporter(): ImporterInfo | null {
    return this.isDatasourceImporter(this.selectedSource) ? this.selectedSource : null;
  }

  /** A datasource importer fetched an item into ``example_media/``: use it
   *  as the new sort example. */
  onDatasourceImported(result: DatasourceImportResult): void {
    this.newExample.emit({ action: 'new-example', type: 'media', value: result.filename });
  }

  selectSource(source: ImporterInfo): void {
    this.selectedSource = source;
    this.browseItems.set([]);
    this.fileBrowsing = false;

    if (this.isDatasourceImporter(source)) {
      // Rendered as a dynamic form; nothing to browse.
      this.browseLoading.set(false);
      return;
    }
    this.browseLoading.set(true);

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
    this.clearFileBrowsing();
    this.fileBrowsing = true;
    this.browseSource = source;
    this.browseSourceLabel = label;
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

  /** Back one view: out of a demo's file entry to the demo list, out of a
   *  selected source (browse list or import form) to the source list, and
   *  out of the picker itself to the prompt. */
  back(): void {
    if (this.fileBrowsing && this.selectedSource?.name === 'demo') {
      this.clearFileBrowsing();
      return;
    }
    if (this.selectedSource) {
      this.selectedSource = null;
      this.browseItems.set([]);
      this.browseLoading.set(false);
      this.clearFileBrowsing();
      return;
    }
    this.view = 'prompt';
  }

  /** Label for the back button, naming the view it returns to. */
  get backLabel(): string {
    if (this.fileBrowsing && this.selectedSource?.name === 'demo') return 'Back to demo list';
    if (this.selectedSource) return 'Back to sources';
    return 'Back';
  }

  private clearFileBrowsing(): void {
    this.fileBrowsing = false;
    this.browseSource = '';
    this.browseSourceLabel = '';
    this.typedPath = '';
    this.typedPathError.set('');
  }
}
