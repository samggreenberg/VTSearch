import { Component, ElementRef, EventEmitter, Input, OnInit, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { IconComponent } from '../../icon/icon.component';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { VoteStateService } from '../../../services/vote-state.service';
import { ImporterField } from '../../../models/api.models';
import type { LabelImporterEntry } from '../../../generated/api-client/models/label-importer-entry';

type ModalView = 'picker' | 'form';

@Component({
  selector: 'vt-label-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, FileBrowserComponent, IconComponent],
  templateUrl: './label-importer-modal.component.html',
  styleUrl: './label-importer-modal.component.scss',
})
export class LabelImporterModalComponent implements OnInit {
  /** When set, labels are imported directly into this trainable model
   *  instead of into the active dataset's vote state. */
  @Input() targetModelName: string | null = null;

  @Output() closed = new EventEmitter<void>();
  @Output() imported = new EventEmitter<void>();

  @ViewChild('addGoodInput') addGoodInput!: ElementRef<HTMLInputElement>;
  @ViewChild('addBadInput') addBadInput!: ElementRef<HTMLInputElement>;

  view: ModalView = 'picker';
  importers: LabelImporterEntry[] = [];
  loading = true;
  selectedImporter: LabelImporterEntry | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  selectedFileFieldKey: string | null = null;
  submitting = false;
  error = '';
  successMessage = '';
  addingGood = false;
  addingBad = false;

  constructor(
    private labelImportersApi: LabelImportersApiService,
    private mediasApi: MediasApiService,
    private voteState: VoteStateService,
  ) {}

  get modalTitle(): string {
    if (this.view === 'form' && this.selectedImporter) {
      return this.selectedImporter.display_name || this.selectedImporter.name;
    }
    return this.targetModelName ? 'Add Labels to Detector' : 'Import Labels';
  }

  /** Typed view of the selected importer's plugin fields for the template
   *  (the generated LabelImporterEntry types `fields` as an open dict
   *  because plugin field schemas aren't part of the OpenAPI client). */
  get selectedImporterFields(): ImporterField[] {
    return (this.selectedImporter?.fields ?? []) as ImporterField[];
  }

  ngOnInit(): void {
    this.labelImportersApi.list().subscribe({
      next: (list) => {
        this.importers = list.filter((imp) => !imp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load label importers';
      },
    });
  }

  selectImporter(importer: LabelImporterEntry): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.selectedFile = null;
    this.selectedFileFieldKey = null;
    this.error = '';
    this.successMessage = '';
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) this.formValues[field.key] = field.default;
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

    const request$ = this.targetModelName
      ? this.labelImportersApi.runModelImport(
          this.targetModelName,
          this.selectedImporter.name,
          this.formValues,
          this.selectedFile ?? undefined,
          this.selectedFileFieldKey ?? undefined,
        )
      : this.labelImportersApi.runImport(
          this.selectedImporter.name,
          this.formValues,
          this.selectedFile ?? undefined,
          this.selectedFileFieldKey ?? undefined,
        );

    request$.subscribe({
      next: (res: any) => {
        this.submitting = false;
        this.successMessage = res.message || `Applied ${res.applied ?? 0} labels`;
        this.imported.emit();

        if (res.missing_count > 0 && res.missing?.length) {
          // Show unresolved elements as a warning but don't prompt — the
          // backend already tried to auto-resolve them.
          this.error = `${res.missing_count} element(s) could not be resolved from their original sources.`;
        }
        setTimeout(() => this.close(), res.missing_count > 0 ? 3000 : 1500);
      },
      error: (err) => {
        this.submitting = false;
        this.error = err.error?.error || 'Import failed';
      },
    });
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
