import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { SettingsIoApiService } from '../../../services/settings-io-api.service';
import { ImporterField } from '../../../models/api.models';
import type { SettingsExporterEntry } from '../../../generated/api-client/models/settings-exporter-entry';
import type { RunSettingsExportResponse } from '../../../generated/api-client/models/run-settings-export-response';

type ModalView = 'picker' | 'form';

@Component({
  selector: 'vt-settings-exporter-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './settings-exporter-modal.component.html',
  styleUrl: './settings-exporter-modal.component.scss',
})
export class SettingsExporterModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() exported = new EventEmitter<void>();

  view: ModalView = 'picker';
  exporters: SettingsExporterEntry[] = [];
  loading = true;
  selectedExporter: SettingsExporterEntry | null = null;
  formValues: Record<string, string> = {};
  submitting = false;
  error = '';
  successMessage = '';

  constructor(private settingsIoApi: SettingsIoApiService) {}

  get modalTitle(): string {
    if (this.view === 'form' && this.selectedExporter) {
      return this.selectedExporter.display_name || this.selectedExporter.name;
    }
    return 'Export Settings';
  }

  /** Typed view of the selected exporter's plugin fields for the template
   *  (the generated SettingsExporterEntry types `fields` as an open dict
   *  because plugin field schemas aren't part of the OpenAPI client). */
  get selectedExporterFields(): ImporterField[] {
    return (this.selectedExporter?.fields ?? []) as ImporterField[];
  }

  ngOnInit(): void {
    this.settingsIoApi.listExporters().subscribe({
      next: (list) => {
        this.exporters = list.filter((exp) => !exp.hidden_from_picker);
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load settings exporters';
      },
    });
  }

  selectExporter(exporter: SettingsExporterEntry): void {
    this.selectedExporter = exporter;
    this.formValues = {};
    this.error = '';
    this.successMessage = '';
    const fields = (exporter.fields ?? []) as ImporterField[];
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
    // If the exporter has no fields, submit immediately
    if (fields.length === 0) {
      this.view = 'form';
      this.submit();
    } else {
      this.view = 'form';
    }
  }

  back(): void {
    this.view = 'picker';
    this.selectedExporter = null;
    this.error = '';
    this.successMessage = '';
  }

  submit(): void {
    if (!this.selectedExporter) return;
    this.submitting = true;
    this.error = '';
    this.successMessage = '';

    this.settingsIoApi.runExport(this.selectedExporter.name, this.formValues).subscribe({
      next: (res: RunSettingsExportResponse) => {
        this.submitting = false;

        // Handle browser download response
        if (res.download && res.data) {
          const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = res.filename || 'settings.json';
          a.click();
          URL.revokeObjectURL(url);
        }

        this.successMessage = res.message || 'Settings exported successfully';
        this.exported.emit();
        setTimeout(() => this.close(), 1500);
      },
      error: (err) => {
        this.submitting = false;
        this.error = err.error?.error || 'Export failed';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
