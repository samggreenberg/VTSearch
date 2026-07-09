import { ChangeDetectionStrategy, ChangeDetectorRef, Component, EventEmitter, Input, Output, inject, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ClipperChooserComponent, ClipperSelection } from '../../../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from '../../import-advanced/import-advanced.component';
import { ImportConfigComponent } from '../../import-config/import-config.component';
import { FieldHintIconComponent } from '../../../../field-hint-icon/field-hint-icon.component';
import { FileBrowserComponent } from '../../../../file-browser/file-browser.component';
import { DatasetsCrudApiService } from '../../../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../../../services/datasets-listings-api.service';
import { apiErrorMessage } from '../../../../../utils/api-error';
import {
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  ImporterField,
  ImporterInfo,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../../models/api.models';
import { ImportDefaultsService } from '../shared/import-defaults.service';
import { availableConvertersFor, composeEmbedders, mediaTypeLabels, mediaTypeOptionIcons, mediaTypeOptionLabels, toFolderName, toTypeId } from '../shared/media-type.util';

/** Generic importer form: renders whatever fields an importer declares
 *  (``server_files``, ``pickle``, or any extension importer whose
 *  ``picker_view`` is the catch-all ``"form"``).  Handles dynamic
 *  (server-fetched) select options, the shared embedder/clipper/source-
 *  specs Advanced block, and submission via ``runImporter`` /
 *  ``loadFile``.
 *
 *  One of the four self-contained "picker view" components extracted
 *  from ``DatasetImporterModalComponent``; see that component for the
 *  shared chrome (importer category tabs) all four sit behind. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-generic-form-picker',
  standalone: true,
  imports: [FormsModule, ImportAdvancedComponent, ImportConfigComponent, ClipperChooserComponent, FieldHintIconComponent, FileBrowserComponent],
  templateUrl: './generic-form-picker.component.html',
  styleUrl: './generic-form-picker.component.scss',
})
export class GenericFormPickerComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private importDefaults = inject(ImportDefaultsService);
  private cdr = inject(ChangeDetectorRef);

  /** Every registered importer (used to resolve the active importer's
   *  ``available_converters_by_media_type`` for the source-specs picker). */
  @Input() importers: ImporterInfo[] = [];
  @Input() mediaTypes: MediaTypeInfo[] = [];
  @Input() guessedMediaType = '';
  @Input() guessedMediaEmbedder = '';

  /** "Build the 2-D Browse projection at ingest" toggle, shared across
   *  every import flow in the parent modal. */
  @Input() buildProjection = false;
  @Output() buildProjectionChange = new EventEmitter<boolean>();
  @Input() mergeNearDuplicates = false;
  @Output() mergeNearDuplicatesChange = new EventEmitter<boolean>();

  @Output() importStarted = new EventEmitter<void>();

  readonly selectedImporter = signal<ImporterInfo | null>(null);
  formValues: Record<string, any> = {};
  selectedFile: File | null = null;
  readonly submitting = signal(false);
  readonly error = signal('');
  /** Whether the user has manually edited the generic-form dataset_name
   *  input (so we stop auto-deriving it from path/url/file fields). */
  private datasetNameDirty = false;

  readonly availableClippers = signal<ClipperInfo[]>([]);
  readonly selectedClipper = signal('');
  readonly clipperParamValues = signal<Record<string, number | string>>({});

  readonly availableEmbedders = signal<EmbedderInfo[]>([]);
  readonly selectedEmbedder = signal('');
  readonly selectedPatchEmbedder = signal('');
  readonly selectedStructuralEmbedder = signal('');

  sourceSpecs: SourceSpec[] = [];

  readonly dynamicFieldOptions = signal<Record<string, string[]>>({});
  readonly dynamicFieldLoading = signal<Record<string, boolean>>({});
  readonly dynamicFieldError = signal<Record<string, string>>({});

  clipperChooserOpen = false;
  clipperChooserClippers: ClipperInfo[] = [];

  get effectiveSoloMediaType(): string | null {
    return this.importDefaults.effectiveSoloMediaType;
  }

  get effectiveSoloFolderName(): string {
    return this.importDefaults.effectiveSoloFolderName(this.mediaTypes);
  }

  get mediaTypeOptionLabels(): Record<string, string> {
    return mediaTypeOptionLabels(this.mediaTypes);
  }

  get mediaTypeLabels(): Record<string, string> {
    return mediaTypeLabels(this.mediaTypes);
  }

  get mediaTypeOptionIcons(): Record<string, string> {
    return mediaTypeOptionIcons(this.mediaTypes);
  }

  lockedEmbedderFor(mediaTypeFolderOrTypeId: string, embedders: EmbedderInfo[]): string {
    return this.importDefaults.lockedEmbedderFor(mediaTypeFolderOrTypeId, this.mediaTypes, embedders);
  }

  /** Open this picker for *importer* (an importer whose ``picker_view``
   *  is the generic ``"form"``, or unset). */
  open(importer: ImporterInfo): void {
    this.selectedImporter.set(importer);
    this.formValues = {};
    this.error.set('');
    this.selectedClipper.set('');
    this.availableClippers.set([]);
    this.clipperParamValues.set({});
    this.selectedEmbedder.set('');
    this.availableEmbedders.set([]);
    this.dynamicFieldOptions.set({});
    this.dynamicFieldLoading.set({});
    this.dynamicFieldError.set({});
    this.datasetNameDirty = false;

    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default !== undefined && field.default !== '') {
          this.formValues[field.key] = field.default;
        } else if (
          field.field_type === 'select' &&
          !field.dynamic_options &&
          (field.options?.length ?? 0) > 0
        ) {
          this.formValues[field.key] = field.options![0];
        }
      }
    }

    const mediaTypeField = importer.fields?.find((f) => f.key === 'media_type');
    if (mediaTypeField && this.guessedMediaType) {
      const folderName = toFolderName(this.mediaTypes, this.guessedMediaType);
      if (folderName && mediaTypeField.options?.includes(folderName)) {
        this.formValues['media_type'] = folderName;
      }
    }

    if (mediaTypeField && this.effectiveSoloFolderName) {
      if (mediaTypeField.options?.includes(this.effectiveSoloFolderName)) {
        this.formValues['media_type'] = this.effectiveSoloFolderName;
      }
    }

    if (mediaTypeField) {
      const defaultType = this.formValues['media_type'] || mediaTypeField.default || '';
      this.loadClippers(defaultType);
      this.loadEmbedders(defaultType);
    }

    this.resetSourceSpecs();

    for (const field of importer.fields || []) {
      if (field.dynamic_options) {
        this.refreshDynamicFieldOptions(field);
      }
    }

    // `open()` is invoked imperatively from the parent's importer-selection
    // handler (a listener bound on a sibling `<vt-source-picker>`), so this
    // component's own OnPush view is not on the ancestor-marked dirty path
    // that call produces. Explicitly notify the scheduler so the
    // just-populated form actually paints under zoneless.
    this.cdr.markForCheck();
  }

  onMediaTypeChange(mediaType: string): void {
    this.formValues['media_type'] = mediaType;
    this.loadClippers(mediaType);
    this.loadEmbedders(mediaType);
    this.resetSourceSpecs();
    this.onFieldChanged('media_type');
  }

  /** Called whenever a form field value changes.  Refreshes options for
   *  every dynamic-options field whose ``depends_on`` includes
   *  *changedKey*, clears the dependent field's now-stale value, and
   *  re-derives ``dataset_name`` unless the user has typed one. */
  onFieldChanged(changedKey: string): void {
    const importer = this.selectedImporter();
    if (!importer?.fields) return;
    for (const field of importer.fields) {
      if (!field.dynamic_options) continue;
      if (!(field.depends_on || []).includes(changedKey)) continue;
      this.formValues[field.key] = '';
      this.refreshDynamicFieldOptions(field);
    }
    if (changedKey !== 'dataset_name') {
      this.maybeApplyDerivedDatasetName();
    }
  }

  onDatasetNameInput(value: string): void {
    this.formValues['dataset_name'] = value;
    this.datasetNameDirty = true;
  }

  onServerPathSelected(key: string, path: string): void {
    this.formValues[key] = path;
    this.onFieldChanged(key);
  }

  private maybeApplyDerivedDatasetName(): void {
    if (this.datasetNameDirty) return;
    const derived = this.derivedDatasetName();
    if (derived) {
      this.formValues['dataset_name'] = derived;
    }
  }

  private derivedDatasetName(): string {
    const fields = this.selectedImporter()?.fields || [];
    for (const f of fields) {
      if (f.key === 'dataset_name') continue;
      const raw = this.formValues[f.key];
      if (typeof raw !== 'string' || !raw) continue;
      if (f.field_type === 'url') {
        const cleaned = raw.split('?')[0].replace(/\/+$/, '');
        const tail = cleaned.split('/').pop() || '';
        if (!tail) continue;
        const stripped = tail.replace(/\.(?:tar\.gz|tar\.bz2|tar\.xz|tar|zip|rar)$/i, '');
        return stripped || tail;
      }
      if (f.field_type === 'server_path' || f.field_type === 'file') {
        const basename = raw.split(/[\\/]/).pop() || '';
        if (!basename) continue;
        const dot = basename.lastIndexOf('.');
        return dot > 0 ? basename.slice(0, dot) : basename;
      }
      if (f.key === 'path') {
        const parts = raw.split(/[\\/]/).filter(Boolean);
        if (parts.length > 0) return parts[parts.length - 1];
      }
    }
    return '';
  }

  private refreshDynamicFieldOptions(field: ImporterField): void {
    const importer = this.selectedImporter();
    if (!importer) return;
    const key = field.key;
    this.dynamicFieldLoading.update((m) => ({ ...m, [key]: true }));
    this.dynamicFieldError.update((m) => ({ ...m, [key]: '' }));
    this.datasetsCrudApi.getImporterFieldOptions(importer.name, key, { ...this.formValues }).subscribe({
      next: (res) => {
        this.dynamicFieldOptions.update((m) => ({ ...m, [key]: res.options || [] }));
        this.dynamicFieldLoading.update((m) => ({ ...m, [key]: false }));
        const current = this.formValues[key];
        if (current && !this.dynamicFieldOptions()[key].includes(String(current))) {
          this.formValues[key] = '';
        }
        if (!this.formValues[key] && field.required && this.dynamicFieldOptions()[key].length > 0) {
          this.formValues[key] = this.dynamicFieldOptions()[key][0];
        }
      },
      error: (err) => {
        this.dynamicFieldLoading.update((m) => ({ ...m, [key]: false }));
        this.dynamicFieldError.update((m) => ({ ...m, [key]: apiErrorMessage(err, 'Could not load options') }));
        this.dynamicFieldOptions.update((m) => ({ ...m, [key]: [] }));
      },
    });
  }

  optionsFor(field: ImporterField): string[] {
    if (field.dynamic_options) {
      return this.dynamicFieldOptions()[field.key] || [];
    }
    return field.options || [];
  }

  private loadClippers(mediaType: string): void {
    if (!mediaType) {
      this.availableClippers.set([]);
      this.selectedClipper.set('');
      return;
    }
    this.datasetsListingsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.availableClippers.set(clippers);
        const chosen = this.importDefaults.chooseClipperForType(clippers, mediaType, this.mediaTypes);
        this.selectedClipper.set(chosen.name);
        if (chosen.params !== null) {
          this.clipperParamValues.set(chosen.params);
        } else {
          this.resetClipperParams();
        }
      },
    });
  }

  onClipperChange(clipperName: string): void {
    this.selectedClipper.set(clipperName);
    this.resetClipperParams();
  }

  get selectedClipperParams(): ClipperParameter[] {
    const clipper = this.availableClippers().find((c) => c.name === this.selectedClipper());
    return clipper?.parameters || [];
  }

  private resetClipperParams(): void {
    const next: Record<string, number | string> = {};
    for (const param of this.selectedClipperParams) {
      next[param.key] = param.default;
    }
    this.clipperParamValues.set(next);
  }

  private loadEmbedders(mediaType: string): void {
    this.selectedPatchEmbedder.set('');
    this.selectedStructuralEmbedder.set('');
    if (!mediaType) {
      this.availableEmbedders.set([]);
      this.selectedEmbedder.set('');
      return;
    }
    this.datasetsListingsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.availableEmbedders.set(embedders);
        this.selectedEmbedder.set(
          this.importDefaults.chooseEmbedderForType(embedders, mediaType, this.mediaTypes, this.guessedMediaEmbedder),
        );
      },
    });
  }

  onFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.formValues[fieldName] = input.files[0].name;
      this.maybeApplyDerivedDatasetName();
    }
  }

  get outputTypeId(): string {
    return toTypeId(this.mediaTypes, String(this.formValues['media_type'] || ''));
  }

  get availableConverters(): ConverterInfo[] {
    if (!this.selectedImporter()) return [];
    return availableConvertersFor(this.importers, this.selectedImporter()!.name, this.outputTypeId);
  }

  resetSourceSpecs(): void {
    this.sourceSpecs = this.importDefaults.specsListWithDefaultsFor(this.mediaTypes, this.outputTypeId, this.availableConverters);
  }

  onSpecsChange(specs: SourceSpec[]): void {
    this.sourceSpecs = specs;
  }

  /** True when every required field on the active importer has a value. */
  get canSubmit(): boolean {
    const fields = this.selectedImporter()?.fields ?? [];
    for (const f of fields) {
      if (!f.required) continue;
      if (f.field_type === 'file') {
        if (!this.selectedFile) return false;
      } else {
        const v = this.formValues[f.key];
        if (v === undefined || v === null || String(v).trim() === '') return false;
      }
    }
    return true;
  }

  openClipperChooser(): void {
    this.clipperChooserClippers = this.availableClippers();
    this.clipperChooserOpen = true;
  }

  onClipperChooserSelected(selection: ClipperSelection): void {
    this.clipperChooserOpen = false;
    this.selectedClipper.set(selection.name);
    this.clipperParamValues.set({ ...selection.params });
  }

  onClipperChooserCancelled(): void {
    this.clipperChooserOpen = false;
    const clippers = this.clipperChooserClippers;
    const defaultClipper = clippers.find((c) => c.name.endsWith('_default')) || clippers[0];
    this.selectedClipper.set(defaultClipper?.name || '');
    this.resetClipperParams();
  }

  submit(): void {
    const importer = this.selectedImporter();
    if (!importer) return;
    this.submitting.set(true);
    this.error.set('');

    const submitValues = { ...this.formValues };
    if (this.selectedClipper()) {
      submitValues['clipper'] = this.selectedClipper();
      if (this.selectedClipperParams.length > 0 && Object.keys(this.clipperParamValues()).length > 0) {
        submitValues['clipper_params'] = { ...this.clipperParamValues() };
      }
    }
    if (this.selectedEmbedder()) {
      submitValues['embedder'] = this.selectedEmbedder();
    }
    const embedders = composeEmbedders(this.selectedEmbedder(), this.selectedPatchEmbedder(), this.selectedStructuralEmbedder());
    if (embedders) {
      submitValues['embedders'] = embedders;
    }
    if (this.sourceSpecs.length > 0) {
      submitValues['source_specs'] = this.sourceSpecs;
    }
    submitValues['build_projection'] = this.buildProjection ? 'true' : 'false';
    submitValues['merge_near_duplicates'] = this.mergeNearDuplicates ? 'true' : 'false';

    const fileField = importer.fields?.find((f) => f.field_type === 'file');
    if (fileField && this.selectedFile) {
      this.datasetsCrudApi.loadFile(this.selectedFile, this.buildProjection).subscribe({
        next: () => {
          this.submitting.set(false);
          this.offerSaveImportDefaults();
          this.importStarted.emit();
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Import failed'));
        },
      });
    } else {
      this.datasetsCrudApi.runImporter(importer.name, submitValues).subscribe({
        next: () => {
          this.submitting.set(false);
          this.offerSaveImportDefaults();
          this.importStarted.emit();
        },
        error: (err) => {
          this.submitting.set(false);
          this.error.set(apiErrorMessage(err, 'Import failed'));
        },
      });
    }
  }

  private offerSaveImportDefaults(): void {
    const typeId = this.outputTypeId;
    const cfg = this.importDefaults.snapshotImportConfig(
      typeId,
      this.selectedEmbedder(),
      this.selectedClipper(),
      this.clipperParamValues(),
      this.sourceSpecs,
      this.availableEmbedders(),
      this.availableClippers(),
    );
    this.importDefaults.maybeOfferSaveImportDefaults(typeId, cfg, this.mediaTypes);
  }
}
