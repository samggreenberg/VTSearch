import { ChangeDetectionStrategy, Component, effect, inject, input, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';

import { ImporterField, ImporterInfo } from '../../models/api.models';
import {
  DatasourceImportersApiService,
  DatasourceImportResult,
} from '../../services/datasource-importers-api.service';
import { apiErrorMessage } from '../../utils/api-error';
import { DynamicFieldOptions } from '../../utils/dynamic-field-options';
import { FieldHintIconComponent } from '../field-hint-icon/field-hint-icon.component';
import { FileBrowserComponent } from '../file-browser/file-browser.component';

/** Dynamic form for one datasource importer, rendered from its declared
 *  plugin fields (the single-item sibling of the Add Dataset modal's
 *  generic importer form).  Submitting runs the importer server-side; the
 *  fetched item lands in ``example_media/`` and is emitted as
 *  ``{filename, original_name}`` for the caller to use as a media
 *  example.
 *
 *  Shared by every flow that picks a single example media item: the New
 *  Detector modal's example picker and the re-sort prompt modal's
 *  swap-the-exemplar picker. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-datasource-import-form',
  standalone: true,
  imports: [FormsModule, FieldHintIconComponent, FileBrowserComponent],
  templateUrl: './datasource-import-form.component.html',
  styleUrl: './datasource-import-form.component.scss',
})
export class DatasourceImportFormComponent {
  private api = inject(DatasourceImportersApiService);

  /** The datasource importer whose fields to render. */
  readonly importer = input.required<ImporterInfo>();

  /** Emits the fetched item once the importer run succeeds. */
  readonly imported = output<DatasourceImportResult>();

  values: Record<string, string> = {};
  selectedFile: File | null = null;
  fileFieldKey: string | null = null;
  readonly submitting = signal(false);
  readonly error = signal('');
  /** Option lists for the importer's ``dynamic_options`` fields. */
  readonly fieldOptions = new DynamicFieldOptions((key, values) =>
    this.api.getFieldOptions(this.importer().name, key, values),
  );

  constructor() {
    // Re-seed the form whenever the parent selects a different importer.
    effect(() => {
      this.resetFor(this.importer());
    });
  }

  /** Typed view of the importer's plugin fields for the template. */
  get importerFields(): ImporterField[] {
    return (this.importer().fields ?? []) as ImporterField[];
  }

  private resetFor(importer: ImporterInfo): void {
    this.values = {};
    this.selectedFile = null;
    this.fileFieldKey = null;
    this.submitting.set(false);
    this.error.set('');
    this.fieldOptions.reset();
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.values[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        !field.allow_free_text &&
        (field.options?.length ?? 0) > 0
      ) {
        this.values[field.key] = field.options![0];
      }
    }
    this.fieldOptions.refreshAll(fields, this.values);
  }

  onFieldChanged(changedKey: string): void {
    this.fieldOptions.refreshDependentsOf(changedKey, this.importerFields, this.values);
  }

  onFileSelected(event: Event, fieldKey: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.fileFieldKey = fieldKey;
      this.values[fieldKey] = input.files[0].name;
    }
  }

  onServerPathSelected(fieldKey: string, path: string): void {
    this.values[fieldKey] = path;
  }

  get canSubmit(): boolean {
    if (this.submitting()) return false;
    for (const field of this.importerFields) {
      if (!field.required) continue;
      if (field.field_type === 'file') {
        if (!this.selectedFile) return false;
      } else if (!(this.values[field.key] || '').trim()) {
        return false;
      }
    }
    return true;
  }

  submit(): void {
    if (!this.canSubmit) return;
    this.submitting.set(true);
    this.error.set('');
    this.api
      .run(
        this.importer().name,
        { ...this.values },
        this.selectedFile ?? undefined,
        this.fileFieldKey ?? undefined,
      )
      .subscribe({
        next: (res) => {
          this.submitting.set(false);
          this.imported.emit(res);
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Failed to fetch the media item'));
        },
      });
  }
}
