import { ChangeDetectionStrategy, Component, computed, inject, OnDestroy, output, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { SettingsIoApiService } from '../../../services/settings-io-api.service';
import { ImporterField } from '../../../models/api.models';
import type { SettingsImporterEntry } from '../../../generated/api-client/models/settings-importer-entry';
import { apiErrorMessage } from '../../../utils/api-error';

type ModalView = 'picker' | 'form';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-settings-importer-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './settings-importer-modal.component.html',
  styleUrl: './settings-importer-modal.component.scss',
})
export class SettingsImporterModalComponent implements OnDestroy {
  readonly closed = output<void>();
  readonly imported = output<void>();

  private readonly settingsIoApi = inject(SettingsIoApiService);

  view: ModalView = 'picker';

  // Eager `rxResource`: loads the settings-importer list once on creation,
  // wrapping the generated-client read so the interceptor chain still applies.
  private readonly importersResource = rxResource({
    stream: () => this.settingsIoApi.listImporters(),
  });
  readonly importers = computed<SettingsImporterEntry[]>(() =>
    (this.importersResource.value() ?? []).filter((imp) => !imp.hidden_from_picker),
  );
  readonly loading = computed(() => this.importersResource.isLoading());

  selectedImporter: SettingsImporterEntry | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  selectedFileFieldKey: string | null = null;
  // Signalized: the submit subscribe writes these from an unpatched callback.
  readonly submitting = signal(false);

  /** Error from a failed import action; the list-load failure is merged in. */
  private readonly importError = signal('');
  readonly error = computed(
    () =>
      this.importError() ||
      (this.importersResource.error() ? 'Failed to load settings importers' : ''),
  );
  readonly successMessage = signal('');
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

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

  selectImporter(importer: SettingsImporterEntry): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.selectedFile = null;
    this.selectedFileFieldKey = null;
    this.importError.set('');
    this.successMessage.set('');
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.formValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        !field.allow_free_text &&
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
    this.importError.set('');
    this.successMessage.set('');
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
    this.submitting.set(true);
    this.importError.set('');
    this.successMessage.set('');

    this.settingsIoApi
      .runImport(
        this.selectedImporter.name,
        this.formValues,
        this.selectedFile ?? undefined,
        this.selectedFileFieldKey ?? undefined,
      )
      .subscribe({
        next: (res) => {
          this.submitting.set(false);
          this.successMessage.set(res.message || 'Settings imported successfully');
          this.imported.emit();
          this.closeTimer = setTimeout(() => this.close(), 1500);
        },
        error: (err) => {
          this.submitting.set(false);
          this.importError.set(apiErrorMessage(err, 'Import failed'));
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
