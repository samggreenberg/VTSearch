import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { SettingsIoApiService } from '../../../services/settings-io-api.service';
import { ImporterField } from '../../../models/api.models';
import type { SettingsImporterEntry } from '../../../generated/api-client/models/settings-importer-entry';

type ModalView = 'picker' | 'form';

@Component({
  selector: 'vt-settings-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './settings-importer-modal.component.html',
  styleUrl: './settings-importer-modal.component.scss',
})
export class SettingsImporterModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() imported = new EventEmitter<void>();

  view: ModalView = 'picker';
  importers: SettingsImporterEntry[] = [];
  loading = true;
  selectedImporter: SettingsImporterEntry | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  selectedFileFieldKey: string | null = null;
  submitting = false;
  error = '';
  successMessage = '';

  constructor(private settingsIoApi: SettingsIoApiService) {}

  get modalTitle(): string {
    if (this.view === 'form' && this.selectedImporter) {
      return this.selectedImporter.display_name || this.selectedImporter.name;
    }
    return 'Import Settings';
  }

  /** Typed view of the selected importer's plugin fields for the template
   *  (the generated SettingsImporterEntry types `fields` as an open dict
   *  because plugin field schemas aren't part of the OpenAPI client). */
  get selectedImporterFields(): ImporterField[] {
    return (this.selectedImporter?.fields ?? []) as ImporterField[];
  }

  ngOnInit(): void {
    this.settingsIoApi.listImporters().subscribe({
      next: (list) => {
        this.importers = list.filter((imp) => !imp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load settings importers';
      },
    });
  }

  selectImporter(importer: SettingsImporterEntry): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.selectedFile = null;
    this.selectedFileFieldKey = null;
    this.error = '';
    this.successMessage = '';
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.formValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        (field.options?.length ?? 0) > 0
      ) {
        this.formValues[field.key] = field.options![0];
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

    this.settingsIoApi
      .runImport(
        this.selectedImporter.name,
        this.formValues,
        this.selectedFile ?? undefined,
        this.selectedFileFieldKey ?? undefined,
      )
      .subscribe({
        next: (res) => {
          this.submitting = false;
          this.successMessage = res.message || 'Settings imported successfully';
          this.imported.emit();
          setTimeout(() => this.close(), 1500);
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Import failed';
        },
      });
  }

  close(): void {
    this.closed.emit();
  }
}
