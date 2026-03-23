import { Component, ElementRef, EventEmitter, OnInit, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { IconComponent } from '../../icon/icon.component';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { VoteStateService } from '../../../services/vote-state.service';
import { ImporterField } from '../../../models/api.models';

interface LabelImporterInfo {
  name: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  ui_mode?: string;
  hidden_from_picker?: boolean;
}

type ModalView = 'picker' | 'form' | 'missing';

@Component({
  selector: 'vt-label-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, FileBrowserComponent, IconComponent],
  templateUrl: './label-importer-modal.component.html',
  styleUrl: './label-importer-modal.component.scss',
})
export class LabelImporterModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() imported = new EventEmitter<void>();

  @ViewChild('addGoodInput') addGoodInput!: ElementRef<HTMLInputElement>;
  @ViewChild('addBadInput') addBadInput!: ElementRef<HTMLInputElement>;

  view: ModalView = 'picker';
  importers: LabelImporterInfo[] = [];
  loading = true;
  selectedImporter: LabelImporterInfo | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  selectedFileFieldKey: string | null = null;
  submitting = false;
  error = '';
  successMessage = '';
  addingGood = false;
  addingBad = false;
  missingEntries: unknown[] = [];
  ingesting = false;

  constructor(
    private labelImportersApi: LabelImportersApiService,
    private mediasApi: MediasApiService,
    private voteState: VoteStateService,
  ) {}

  get modalTitle(): string {
    if (this.view === 'missing') {
      return 'Missing Media';
    }
    if (this.view === 'form' && this.selectedImporter) {
      return this.selectedImporter.display_name || this.selectedImporter.name;
    }
    return 'Import Labels';
  }

  ngOnInit(): void {
    this.labelImportersApi.list().subscribe({
      next: (list: any[]) => {
        this.importers = list.filter((imp) => !imp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load label importers';
      },
    });
  }

  selectImporter(importer: LabelImporterInfo): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.selectedFile = null;
    this.selectedFileFieldKey = null;
    this.error = '';
    this.successMessage = '';
    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default) this.formValues[field.key] = field.default;
      }
    }
    this.view = 'form';
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.error = '';
    this.successMessage = '';
  }

  onFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.selectedFileFieldKey = fieldName;
      this.formValues[fieldName] = input.files[0].name;
    }
  }

  submit(): void {
    if (!this.selectedImporter) return;
    this.submitting = true;
    this.error = '';
    this.successMessage = '';

    this.labelImportersApi.runImport(
      this.selectedImporter.name,
      this.formValues,
      this.selectedFile ?? undefined,
      this.selectedFileFieldKey ?? undefined,
    ).subscribe({
      next: (res: any) => {
        this.submitting = false;
        this.successMessage = res.message || `Applied ${res.applied ?? 0} labels`;
        this.imported.emit();

        if (res.missing_count > 0 && res.missing?.length) {
          this.missingEntries = res.missing;
          this.view = 'missing';
        } else {
          setTimeout(() => this.close(), 1500);
        }
      },
      error: (err) => {
        this.submitting = false;
        this.error = err.error?.error || 'Import failed';
      },
    });
  }

  ingestMissing(): void {
    if (!this.missingEntries.length) return;
    this.ingesting = true;
    this.error = '';

    this.labelImportersApi.ingestMissing(this.missingEntries).subscribe({
      next: (res: any) => {
        this.ingesting = false;
        this.successMessage = res.message || `Ingested ${res.ingested ?? 0} media(s), applied ${res.applied ?? 0} label(s).`;
        this.missingEntries = [];
        this.imported.emit();
        setTimeout(() => this.close(), 1500);
      },
      error: (err) => {
        this.ingesting = false;
        this.error = err.error?.error || 'Failed to ingest missing media';
      },
    });
  }

  skipMissing(): void {
    this.missingEntries = [];
    this.close();
  }

  triggerAddGood(): void {
    this.addGoodInput.nativeElement.click();
  }

  triggerAddBad(): void {
    this.addBadInput.nativeElement.click();
  }

  onAddToPile(event: Event, label: 'good' | 'bad'): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    input.value = '';

    if (label === 'good') {
      this.addingGood = true;
    } else {
      this.addingBad = true;
    }
    this.error = '';
    this.successMessage = '';

    this.mediasApi.addToPile(file, label).subscribe({
      next: (result) => {
        const action = result.is_new ? 'Added new media' : 'Matched existing media';
        this.successMessage = `${action} to ${label} pile.`;
        this.voteState.loadVotes();
        this.imported.emit();
        if (label === 'good') {
          this.addingGood = false;
        } else {
          this.addingBad = false;
        }
      },
      error: (err) => {
        this.error = err.error?.error || `Failed to add media to ${label} pile`;
        if (label === 'good') {
          this.addingGood = false;
        } else {
          this.addingBad = false;
        }
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
