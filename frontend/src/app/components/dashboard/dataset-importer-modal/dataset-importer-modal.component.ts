import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { ImporterInfo, DemoDataset, MediaTypeInfo } from '../../../models/api.models';

type ModalView = 'picker' | 'form' | 'demo';

@Component({
  selector: 'vt-dataset-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './dataset-importer-modal.component.html',
  styleUrl: './dataset-importer-modal.component.scss',
})
export class DatasetImporterModalComponent implements OnInit {
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

  // Demo picker state
  demos: DemoDataset[] = [];
  mediaTypes: MediaTypeInfo[] = [];
  demoTabs: string[] = [];
  activeTab = '';
  demoSortKey = 'num_files';
  demoSortAsc = true;
  demoLoading = false;

  constructor(private datasetsApi: DatasetsApiService) {}

  ngOnInit(): void {
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.importers = (res.importers || []).filter(
          (imp) => !imp.hidden_from_picker
        );
      },
    });
  }

  selectImporter(importer: ImporterInfo): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.error = '';

    // Pre-populate defaults
    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default !== undefined) {
          this.formValues[field.key] = field.default;
        }
      }
    }

    this.view = 'form';
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

  private fetchDemos(): void {
    this.datasetsApi.getDemoList().subscribe({
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
      this.activeTab = this.demoTabs[0];
    }
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

  getTabLabel(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    if (mt) {
      return `${mt.icon || ''} ${mt.tab_title || mt.name}`.trim();
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

  selectDemo(demo: DemoDataset): void {
    this.demoSelected.emit(demo);
    this.closed.emit();
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.error = '';
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
      this.datasetsApi.runImporter(this.selectedImporter.name, this.formValues).subscribe({
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
