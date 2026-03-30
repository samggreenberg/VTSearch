import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { IconComponent } from '../../icon/icon.component';
import { SettingsIoApiService } from '../../../services/settings-io-api.service';
import { ImporterField } from '../../../models/api.models';

interface SettingsImporterInfo {
  name: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  ui_mode?: string;
  hidden_from_picker?: boolean;
}

type ModalView = 'picker' | 'form';

@Component({
  selector: 'vt-settings-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, FileBrowserComponent, IconComponent],
  templateUrl: './settings-importer-modal.component.html',
  styleUrl: './settings-importer-modal.component.scss',
})
export class SettingsImporterModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() imported = new EventEmitter<void>();

  view: ModalView = 'picker';
  importers: SettingsImporterInfo[] = [];
  loading = true;
  selectedImporter: SettingsImporterInfo | null = null;
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

  ngOnInit(): void {
    this.settingsIoApi.listImporters().subscribe({
      next: (list: SettingsImporterInfo[]) => {
        this.importers = list.filter((imp) => !imp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load settings importers';
      },
    });
  }

  selectImporter(importer: SettingsImporterInfo): void {
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
