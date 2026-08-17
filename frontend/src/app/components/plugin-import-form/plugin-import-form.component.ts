import { ChangeDetectionStrategy, Component, effect, inject, input, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { Observable } from 'rxjs';

import { FieldOptions, ImporterField, ImporterInfo } from '../../models/api.models';
import { DatasourceImportersApiService } from '../../services/datasource-importers-api.service';
import { apiErrorMessage } from '../../utils/api-error';
import { DynamicFieldOptions } from '../../utils/dynamic-field-options';
import { FieldHintIconComponent } from '../field-hint-icon/field-hint-icon.component';
import { FileBrowserComponent } from '../file-browser/file-browser.component';

/** The two calls this form makes against whichever plugin family backs it.
 *
 *  Both {@link DatasourceImportersApiService} and
 *  {@link SeedImportersApiService} satisfy this structurally, so switching
 *  families is a matter of binding a different service — the field
 *  rendering, dynamic-option refresh, validation, and error handling are
 *  identical across families because they all come from the same
 *  {@link ImporterField} declarations. */
export interface PluginImportApi {
  run(
    pluginName: string,
    values: Record<string, string>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown>;
  getFieldOptions(
    pluginName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }>;
}

/** Dynamic form for one media-fetching plugin, rendered from its declared
 *  plugin fields (the single-item sibling of the Add Dataset modal's
 *  generic importer form).  Submitting runs the plugin server-side; what it
 *  fetched lands in ``example_media/`` and is emitted for the caller to use.
 *
 *  Family-neutral: {@link api} decides which endpoints the form talks to.
 *  It defaults to the datasource importers (fetch a *single* exemplar), and
 *  the New Detector modal's Blank flow rebinds it to the seed importers
 *  (fetch a *batch* of unlabeled seeds).  The emitted payload's shape is
 *  therefore the bound family's, so callers narrow it themselves.
 *
 *  Used by the New Detector modal's example picker and seed tabs, and by
 *  the re-sort prompt modal's swap-the-exemplar picker. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-plugin-import-form',
  standalone: true,
  imports: [FormsModule, FieldHintIconComponent, FileBrowserComponent],
  templateUrl: './plugin-import-form.component.html',
  styleUrl: './plugin-import-form.component.scss',
})
export class PluginImportFormComponent {
  private datasourceApi = inject(DatasourceImportersApiService);

  /** The plugin whose fields to render. */
  readonly importer = input.required<ImporterInfo>();

  /** Which plugin family's endpoints to call.  Defaults to datasource
   *  importers so the single-exemplar callers need no binding. */
  readonly api = input<PluginImportApi>(this.datasourceApi);

  /** Label of the submit button in its resting state. */
  readonly submitLabel = input('Load');

  /** Tooltip on the submit button. */
  readonly submitTitle = input('Fetch this media item and use it as the example');

  /** Message shown when the run fails without a server-supplied one. */
  readonly errorFallback = input('Failed to fetch the media item');

  /** Emits the bound family's run-response once the plugin run succeeds. */
  readonly imported = output<unknown>();

  values: Record<string, string> = {};
  selectedFile: File | null = null;
  fileFieldKey: string | null = null;
  readonly submitting = signal(false);
  readonly error = signal('');
  /** Option lists for the plugin's ``dynamic_options`` fields. */
  readonly fieldOptions = new DynamicFieldOptions((key, values) =>
    this.api().getFieldOptions(this.importer().name, key, values),
  );

  constructor() {
    // Re-seed the form whenever the parent selects a different plugin.
    effect(() => {
      this.resetFor(this.importer());
    });
  }

  /** Typed view of the plugin's fields for the template. */
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
    this.api()
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
          this.error.set(apiErrorMessage(err, this.errorFallback()));
        },
      });
  }
}
