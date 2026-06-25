import { ChangeDetectionStrategy, Component, computed, inject, OnDestroy, output, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';

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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-settings-exporter-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './settings-exporter-modal.component.html',
  styleUrl: './settings-exporter-modal.component.scss',
})
export class SettingsExporterModalComponent implements OnDestroy {
  readonly closed = output<void>();
  readonly exported = output<void>();

  private readonly settingsIoApi = inject(SettingsIoApiService);

  view: ModalView = 'picker';

  // Eager `rxResource`: loads the settings-exporter list once on creation,
  // wrapping the generated-client read so the interceptor chain still applies.
  private readonly exportersResource = rxResource({
    stream: () => this.settingsIoApi.listExporters(),
  });
  readonly exporters = computed<SettingsExporterEntry[]>(() =>
    (this.exportersResource.value() ?? []).filter((exp) => !exp.hidden_from_picker),
  );
  readonly loading = computed(() => this.exportersResource.isLoading());

  selectedExporter: SettingsExporterEntry | null = null;
  formValues: Record<string, string> = {};
  // Signalized: the submit subscribe writes these from an unpatched callback.
  readonly submitting = signal(false);

  /** Error from a failed export action; the list-load failure is merged in. */
  private readonly exportError = signal('');
  readonly error = computed(
    () =>
      this.exportError() ||
      (this.exportersResource.error() ? 'Failed to load settings exporters' : ''),
  );
  readonly successMessage = signal('');
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

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

  selectExporter(exporter: SettingsExporterEntry): void {
    this.selectedExporter = exporter;
    this.formValues = {};
    this.exportError.set('');
    this.successMessage.set('');
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
    this.exportError.set('');
    this.successMessage.set('');
  }

  submit(): void {
    if (!this.selectedExporter) return;
    this.submitting.set(true);
    this.exportError.set('');
    this.successMessage.set('');

    this.settingsIoApi.runExport(this.selectedExporter.name, this.formValues).subscribe({
      next: (res: RunSettingsExportResponse) => {
        this.submitting.set(false);

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

        this.successMessage.set(res.message || 'Settings exported successfully');
        this.exported.emit();
        this.closeTimer = setTimeout(() => this.close(), 1500);
      },
      error: (err) => {
        this.submitting.set(false);
        this.exportError.set(err.error?.error || 'Export failed');
      },
    });
  }

  close(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
    this.closed.emit();
  }

  ngOnDestroy(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
  }
}
