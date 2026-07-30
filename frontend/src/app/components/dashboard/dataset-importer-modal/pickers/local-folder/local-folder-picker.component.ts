import { ChangeDetectionStrategy, ChangeDetectorRef, Component, inject, signal, input, output } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ClipperChooserComponent, ClipperSelection } from '../../../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from '../../import-advanced/import-advanced.component';
import { DatasetsCrudApiService } from '../../../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../../../services/datasets-listings-api.service';
import { apiErrorMessage } from '../../../../../utils/api-error';
import type { DatasetLoadStartedResponse } from '../../../../../generated/api-client/models/dataset-load-started-response';
import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  ImporterInfo,
  MediaTypeDetectionResponse,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../../models/api.models';
import { ImportDefaultsService } from '../shared/import-defaults.service';
import {
  autofillFromDetection,
  availableConvertersFor,
  composeEmbedders,
  detectFromFiles,
  detectionHint,
  mediaTypeLabels,
  mediaTypeOptionIcons,
  mediaTypeOptionLabels,
  readRecursiveDefault,
  toFolderName,
  toTypeId,
} from '../shared/media-type.util';
import { Observable } from 'rxjs';

/** Local-folder / local-files picker view: the user drops (or browses
 *  to) a folder or a single paths file from the machine running the
 *  browser.  Both importers ("Folder" and "Files") share this one
 *  component; :prop:`pickerKind` (derived from the selected importer's
 *  name in :meth:`open`) picks which sub-flow is active.
 *
 *  Sits behind the shared ``<vt-source-picker>`` chrome, which owns the
 *  dropzone widget itself; this component owns everything around it
 *  (media-type select, recursive checkbox, dataset name, Advanced
 *  block) via the ``lfInfo`` / ``lfBefore`` / ``lfAfter`` projection
 *  slots, plus the upload/submit logic. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-local-folder-picker',
  standalone: true,
  imports: [FormsModule, ImportAdvancedComponent, ClipperChooserComponent],
  templateUrl: './local-folder-picker.component.html',
  styleUrl: './local-folder-picker.component.scss',
})
export class LocalFolderPickerComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private importDefaults = inject(ImportDefaultsService);
  private cdr = inject(ChangeDetectorRef);

  readonly importers = input<ImporterInfo[]>([]);
  readonly mediaTypes = input<MediaTypeInfo[]>([]);
  readonly guessedMediaType = input('');
  readonly guessedMediaEmbedder = input('');

  readonly buildProjection = input(false);
  readonly buildProjectionChange = output<boolean>();
  readonly mergeNearDuplicates = input(false);
  readonly mergeNearDuplicatesChange = output<boolean>();

  readonly importStarted = output<void>();

  readonly selectedImporter = signal<ImporterInfo | null>(null);

  /** ``"folder"`` opens a directory picker (Local Folder card), ``"files"``
   *  opens a multi-file picker (Local Files card).  Bound (via the
   *  parent) onto the sibling ``<vt-source-picker>``, hence a signal. */
  readonly pickerKind = signal<'folder' | 'files'>('folder');
  /** Picked files.  Same cross-component binding requirement as
   *  :prop:`pickerKind`. */
  readonly files = signal<File[]>([]);
  /** Set from :meth:`onFilesDropped`, invoked via a listener bound on
   *  the sibling `<vt-source-picker>` in the parent's template - a
   *  signal so this component's own `detectionHint()` read notifies
   *  correctly regardless of ancestor-marking. */
  readonly detection = signal<MediaTypeDetectionResponse | null>(null);
  readonly submitting = signal(false);
  readonly error = signal('');
  recursive = true;
  datasetName = '';
  private datasetNameDirty = false;

  mediaType = '';
  mediaTypeOptions: string[] = [];
  readonly embedders = signal<EmbedderInfo[]>([]);
  readonly selectedEmbedder = signal('');
  readonly selectedPatchEmbedder = signal('');
  readonly selectedStructuralEmbedder = signal('');
  readonly clippers = signal<ClipperInfo[]>([]);
  /** Cleanup gates available for the current media type, and the subset the
   *  user has enabled (seeded from each cleaner's ``default_enabled``). */
  readonly cleaners = signal<CleanerInfo[]>([]);
  readonly selectedCleaners = signal<CleanerSelection[]>([]);
  readonly selectedClipper = signal('');
  clipperParams: ClipperParameter[] = [];
  readonly clipperParamValues = signal<Record<string, number | string>>({});

  sourceSpecs: SourceSpec[] = [];

  clipperChooserOpen = false;
  clipperChooserClippers: ClipperInfo[] = [];

  get effectiveSoloMediaType(): string | null {
    return this.importDefaults.effectiveSoloMediaType;
  }

  get effectiveSoloFolderName(): string {
    return this.importDefaults.effectiveSoloFolderName(this.mediaTypes());
  }

  get mediaTypeOptionLabels(): Record<string, string> {
    return mediaTypeOptionLabels(this.mediaTypes());
  }

  get mediaTypeOptionIcons(): Record<string, string> {
    return mediaTypeOptionIcons(this.mediaTypes());
  }

  get mediaTypeLabels(): Record<string, string> {
    return mediaTypeLabels(this.mediaTypes());
  }

  lockedEmbedderFor(mediaTypeFolderOrTypeId: string, embedders: EmbedderInfo[]): string {
    return this.importDefaults.lockedEmbedderFor(mediaTypeFolderOrTypeId, this.mediaTypes(), embedders);
  }

  detectionHint(): string {
    return detectionHint(this.mediaTypes(), this.detection());
  }

  /** First selected file's webkitRelativePath top-level segment, for display. */
  get folderName(): string {
    const files = this.files();
    if (files.length === 0) return '';
    const rel = (files[0] as any).webkitRelativePath as string | undefined;
    if (!rel) return '';
    const idx = rel.indexOf('/');
    return idx >= 0 ? rel.slice(0, idx) : rel;
  }

  open(importer: ImporterInfo | null): void {
    this.selectedImporter.set(importer);
    this.pickerKind.set(importer?.name === 'local_files' ? 'files' : 'folder');
    this.files.set([]);
    this.detection.set(null);
    this.error.set('');
    this.submitting.set(false);
    this.recursive = readRecursiveDefault(importer);
    this.datasetName = '';
    this.datasetNameDirty = false;

    const folderImporter = this.importers().find((imp) => imp.name === 'server_folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    this.mediaTypeOptions = mtField?.options || [];

    const guessedFolder = toFolderName(this.mediaTypes(), this.guessedMediaType());
    if (guessedFolder && this.mediaTypeOptions.includes(guessedFolder)) {
      this.mediaType = guessedFolder;
    } else {
      this.mediaType = mtField?.default || this.mediaTypeOptions[0] || 'audio';
    }

    if (this.effectiveSoloFolderName && this.mediaTypeOptions.includes(this.effectiveSoloFolderName)) {
      this.mediaType = this.effectiveSoloFolderName;
    }

    this.loadEmbedders(this.mediaType);
    this.loadClippers(this.mediaType);
    this.loadCleaners(this.mediaType);
    this.resetSourceSpecs();

    // `open()` is invoked imperatively from the parent's importer-selection
    // handler (a listener bound on a sibling `<vt-source-picker>`), so this
    // component's own OnPush view is not on the ancestor-marked dirty path
    // that call produces; force a check so the reset form actually paints.
    this.cdr.markForCheck();
  }

  /** Invoked via a listener bound on the sibling `<vt-source-picker>`
   *  in the parent's template (the drop-zone lives in its rendered
   *  chrome), so this component isn't on the ancestor-marked dirty
   *  path; `markForCheck()` covers the plain `datasetName` field
   *  alongside the signal writes below. */
  onFilesDropped(files: File[]): void {
    this.acceptFiles(files);
    this.cdr.markForCheck();
  }

  private acceptFiles(files: File[]): void {
    if (files.length === 0) {
      this.files.set([]);
      this.detection.set(null);
      return;
    }
    if (this.pickerKind() === 'files') {
      this.files.set([files[0]]);
      this.detection.set(null);
      this.error.set('');
      if (!this.datasetNameDirty) {
        this.datasetName = this.derivedDatasetName();
      }
      return;
    }
    this.files.set(files);
    this.error.set('');
    if (!this.datasetNameDirty) {
      this.datasetName = this.derivedDatasetName();
    }
    this.detection.set(detectFromFiles(this.mediaTypes(), this.files(), this.recursive));
    this.applyDetection();
  }

  onRecursiveChange(recursive: boolean): void {
    this.recursive = recursive;
    if (this.files().length > 0) {
      this.detection.set(detectFromFiles(this.mediaTypes(), this.files(), this.recursive));
      this.applyDetection();
    }
  }

  private applyDetection(): void {
    const detection = this.detection();
    if (!detection) return;
    const { mediaType, sourceSpecs } = autofillFromDetection(this.mediaTypes(), detection, this.mediaTypeOptions, (typeId) =>
      availableConvertersFor(this.importers(), 'server_folder', typeId),
    );
    if (mediaType && mediaType !== this.mediaType) {
      this.mediaType = mediaType;
      this.loadEmbedders(this.mediaType);
      this.loadClippers(this.mediaType);
      this.loadCleaners(this.mediaType);
    }
    if (sourceSpecs) {
      this.sourceSpecs = sourceSpecs;
    }
  }

  private derivedDatasetName(): string {
    if (this.pickerKind() === 'folder' && this.folderName) {
      return this.folderName;
    }
    const files = this.files();
    if (files.length === 1) {
      const name = files[0].name || '';
      const dot = name.lastIndexOf('.');
      return dot > 0 ? name.slice(0, dot) : name;
    }
    return '';
  }

  onDatasetNameInput(value: string): void {
    this.datasetName = value;
    this.datasetNameDirty = true;
  }

  onMediaTypeChange(mediaType: string): void {
    this.mediaType = mediaType;
    this.loadEmbedders(mediaType);
    this.loadClippers(mediaType);
    this.loadCleaners(mediaType);
    this.resetSourceSpecs();
  }

  private loadEmbedders(mediaType: string): void {
    this.selectedPatchEmbedder.set('');
    this.selectedStructuralEmbedder.set('');
    if (!mediaType) {
      this.embedders.set([]);
      this.selectedEmbedder.set('');
      return;
    }
    this.datasetsListingsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        this.embedders.set(embedders);
        this.selectedEmbedder.set(
          this.importDefaults.chooseEmbedderForType(embedders, mediaType, this.mediaTypes(), this.guessedMediaEmbedder()),
        );
      },
    });
  }

  /** Fetch the cleanup gates registered for *mediaType* and seed the
   *  selection from each cleaner's ``default_enabled`` flag. */
  private loadCleaners(mediaType: string): void {
    if (!mediaType) {
      this.cleaners.set([]);
      this.selectedCleaners.set([]);
      return;
    }
    this.datasetsListingsApi.getCleaners(mediaType).subscribe({
      next: (cleaners) => {
        this.cleaners.set(cleaners);
        this.selectedCleaners.set(this.importDefaults.defaultCleanerSelection(cleaners));
      },
    });
  }

  private loadClippers(mediaType: string): void {
    if (!mediaType) {
      this.clippers.set([]);
      this.selectedClipper.set('');
      return;
    }
    this.datasetsListingsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        this.clippers.set(clippers);
        const chosen = this.importDefaults.chooseClipperForType(clippers, mediaType, this.mediaTypes());
        this.selectedClipper.set(chosen.name);
        if (chosen.params !== null) {
          this.clipperParams = clippers.find((c) => c.name === chosen.name)?.parameters || [];
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

  private resetClipperParams(): void {
    const clipper = this.clippers().find((c) => c.name === this.selectedClipper());
    this.clipperParams = clipper?.parameters || [];
    const next: Record<string, number | string> = {};
    for (const param of this.clipperParams) {
      next[param.key] = param.default;
    }
    this.clipperParamValues.set(next);
  }

  get outputTypeId(): string {
    return toTypeId(this.mediaTypes(), this.mediaType);
  }

  get availableConverters(): ConverterInfo[] {
    return availableConvertersFor(this.importers(), 'server_folder', this.outputTypeId);
  }

  private resetSourceSpecs(): void {
    this.sourceSpecs = this.importDefaults.specsListWithDefaultsFor(this.mediaTypes(), this.outputTypeId, this.availableConverters);
  }

  onSpecsChange(specs: SourceSpec[]): void {
    this.sourceSpecs = specs;
  }

  openClipperChooser(): void {
    this.clipperChooserClippers = this.clippers();
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
    if (this.files().length === 0) {
      this.error.set(this.pickerKind() === 'files' ? 'Please select a paths file to upload.' : 'Please select a folder to upload.');
      return;
    }
    if (this.pickerKind() === 'files') {
      this.submitFiles();
      return;
    }
    this.submitFolder();
  }

  private submitFolder(): void {
    let filesToUpload = this.files();
    if (!this.recursive) {
      filesToUpload = filesToUpload.filter((file) => {
        const rel = ((file as any).webkitRelativePath as string | undefined) || '';
        return rel.split('/').length <= 2;
      });
      if (filesToUpload.length === 0) {
        this.error.set('No files at the top level of the selected folder. Enable "Include subfolders" to import nested files.');
        return;
      }
    }

    this.submitting.set(true);
    this.error.set('');

    const formData = new FormData();
    formData.append('media_type', this.mediaType);
    formData.append('recursive', this.recursive ? 'true' : 'false');
    this.appendCommonFormFields(formData);
    for (const file of filesToUpload) {
      const rel = (file as any).webkitRelativePath as string | undefined;
      formData.append('files', file, rel && rel.length > 0 ? rel : file.name);
    }
    if (this.sourceSpecs.length > 0) {
      formData.append('source_specs', JSON.stringify(this.sourceSpecs));
    }

    this.runUpload(this.datasetsCrudApi.importLocalFolder(formData));
  }

  private submitFiles(): void {
    this.submitting.set(true);
    this.error.set('');

    const pathsFile = this.files()[0];
    const formData = new FormData();
    formData.append('media_type', this.mediaType);
    formData.append('paths_file', pathsFile, pathsFile.name);
    this.appendCommonFormFields(formData);
    if (this.sourceSpecs.length > 0) {
      formData.append('source_specs', JSON.stringify(this.sourceSpecs));
    }

    this.runUpload(this.datasetsCrudApi.importLocalFiles(formData));
  }

  private runUpload(request: Observable<DatasetLoadStartedResponse>): void {
    request.subscribe({
      next: () => {
        this.submitting.set(false);
        this.offerSaveImportDefaults();
        this.importStarted.emit();
      },
      error: (err) => {
        this.submitting.set(false);
        this.error.set(apiErrorMessage(err, 'Upload failed'));
      },
    });
  }

  private appendCommonFormFields(formData: FormData): void {
    const name = (this.datasetName || '').trim();
    if (name) {
      formData.append('dataset_name', name);
    }
    if (this.selectedEmbedder()) {
      formData.append('embedder', this.selectedEmbedder());
    }
    const embedders = composeEmbedders(this.selectedEmbedder(), this.selectedPatchEmbedder(), this.selectedStructuralEmbedder());
    if (embedders) {
      formData.append('embedders', JSON.stringify(embedders));
    }
    if (this.selectedClipper()) {
      formData.append('clipper', this.selectedClipper());
      if (this.clipperParams.length > 0 && Object.keys(this.clipperParamValues()).length > 0) {
        formData.append('clipper_params', JSON.stringify(this.clipperParamValues()));
      }
    }
    if (this.selectedCleaners().length > 0) {
      formData.append('cleaners', JSON.stringify(this.selectedCleaners()));
    }
    formData.append('build_projection', this.buildProjection() ? 'true' : 'false');
    formData.append('merge_near_duplicates', this.mergeNearDuplicates() ? 'true' : 'false');
  }

  private offerSaveImportDefaults(): void {
    const typeId = this.outputTypeId;
    const cfg = this.importDefaults.snapshotImportConfig(
      typeId,
      this.selectedEmbedder(),
      this.selectedClipper(),
      this.clipperParamValues(),
      this.sourceSpecs,
      this.embedders(),
      this.clippers(),
    );
    this.importDefaults.maybeOfferSaveImportDefaults(typeId, cfg, this.mediaTypes());
  }
}
