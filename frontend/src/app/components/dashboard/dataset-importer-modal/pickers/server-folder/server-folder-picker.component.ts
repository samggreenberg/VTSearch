import { ChangeDetectionStrategy, ChangeDetectorRef, Component, inject, signal, input, output } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { map } from 'rxjs/operators';
import { ClipperChooserComponent, ClipperSelection } from '../../../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from '../../import-advanced/import-advanced.component';
import { FolderBrowserComponent, FolderBrowserBrowseFn } from '../../../../folder-browser/folder-browser.component';
import { DatasetsCrudApiService } from '../../../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../../../services/datasets-listings-api.service';
import { DatasetsUiApiService } from '../../../../../services/datasets-ui-api.service';
import { apiErrorMessage } from '../../../../../utils/api-error';
import {
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
  detectionHint,
  mediaTypeLabels,
  mediaTypeOptionIcons,
  mediaTypeOptionLabels,
  readRecursiveDefault,
  toFolderName,
  toTypeId,
} from '../shared/media-type.util';

/** Server-folder picker view: user types (or browses to) an absolute
 *  path on the server filesystem; the backend auto-detects the media
 *  type and the picker offers converter rows for any secondary types
 *  found in the sample.
 *
 *  Sits behind the shared ``<vt-source-picker>`` chrome (rendered by
 *  the parent modal) which owns the typed-path input widget itself;
 *  this component owns everything below/around it (media-type select,
 *  inline folder browser, checkboxes, dataset name, Advanced block) via
 *  the ``sfBefore`` / ``sfAfter`` projection slots, plus the submit
 *  logic. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-server-folder-picker',
  standalone: true,
  imports: [FormsModule, ImportAdvancedComponent, ClipperChooserComponent, FolderBrowserComponent],
  templateUrl: './server-folder-picker.component.html',
  styleUrl: './server-folder-picker.component.scss',
})
export class ServerFolderPickerComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
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

  /** Current value of the editable absolute-path input.  Two-way bound
   *  (via the parent) to ``<vt-source-picker>``'s ``sfPathInputValue``
   *  input/output, so this must be a signal: the widget that renders it
   *  lives in a sibling component. */
  readonly pathInputValue = signal('');
  /** Committed absolute server path (trailing slash trimmed); read by
   *  the parent's footer submit button, hence also a signal. */
  readonly folderPath = signal('');
  readonly browseError = signal('');
  readonly mediaType = signal('');
  mediaTypeOptions: string[] = [];
  readonly embedders = signal<EmbedderInfo[]>([]);
  readonly selectedEmbedder = signal('');
  readonly selectedPatchEmbedder = signal('');
  readonly selectedStructuralEmbedder = signal('');
  readonly clippers = signal<ClipperInfo[]>([]);
  readonly selectedClipper = signal('');
  clipperParams: ClipperParameter[] = [];
  readonly clipperParamValues = signal<Record<string, number | string>>({});
  readonly submitting = signal(false);
  /** Whether subdirectories of the picked server folder are scanned. */
  recursive = true;
  /** Whether archives inside the picked folder are extracted and imported. */
  digArchives = false;
  /** Whether the dataset references the original files in place. */
  referenceFiles = false;
  datasetName = '';
  private datasetNameDirty = false;
  readonly sourceSpecs = signal<SourceSpec[]>([]);
  readonly detection = signal<MediaTypeDetectionResponse | null>(null);

  /** Whether the inline server-filesystem folder browser is expanded. */
  browserOpen = false;

  clipperChooserOpen = false;
  clipperChooserClippers: ClipperInfo[] = [];

  /** Token guarding overlapping detection responses.  Each
   *  :meth:`runDetection` invocation bumps this and stamps it onto its
   *  closure; the response handler bails when the token has moved on. */
  private detectionToken = 0;

  readonly folderBrowseFn: FolderBrowserBrowseFn = (path: string) =>
    this.datasetsUiApi.browseMediaFiles('server_fs', path).pipe(
      map((res) => ({
        directories: res.directories,
        files: res.files,
        rootPath: res.root_path,
      })),
    );

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

  open(importer: ImporterInfo | null): void {
    this.selectedImporter.set(importer);
    this.folderPath.set('');
    this.pathInputValue.set('');
    this.browseError.set('');
    this.detection.set(null);
    this.submitting.set(false);
    this.recursive = readRecursiveDefault(importer);
    this.digArchives = false;
    this.referenceFiles = false;
    this.datasetName = '';
    this.datasetNameDirty = false;

    const folderImporter = this.importers().find((imp) => imp.name === 'server_folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    this.mediaTypeOptions = mtField?.options || [];

    const guessedFolder = toFolderName(this.mediaTypes(), this.guessedMediaType());
    if (guessedFolder && this.mediaTypeOptions.includes(guessedFolder)) {
      this.mediaType.set(guessedFolder);
    } else {
      this.mediaType.set(mtField?.default || this.mediaTypeOptions[0] || 'audio');
    }

    if (this.effectiveSoloFolderName && this.mediaTypeOptions.includes(this.effectiveSoloFolderName)) {
      this.mediaType.set(this.effectiveSoloFolderName);
    }

    this.loadEmbedders(this.mediaType());
    this.loadClippers(this.mediaType());
    this.resetSourceSpecs();

    // `open()` is invoked imperatively from the parent's importer-selection
    // handler (a listener bound on a sibling `<vt-source-picker>`), so this
    // component's own OnPush view is not on the ancestor-marked dirty path
    // that call produces; force a check so the reset form actually paints.
    this.cdr.markForCheck();
  }

  onFolderPicked(evt: { path: string; rootPath: string }): void {
    const { path, rootPath } = evt;
    let absolute: string;
    if (!rootPath) {
      absolute = path;
    } else if (!path) {
      absolute = rootPath;
    } else if (rootPath === '/') {
      absolute = '/' + path;
    } else {
      absolute = rootPath + '/' + path;
    }
    this.pathInputValue.set(absolute);
    this.applyPathInput();
  }

  /** Invoked via a listener bound on the sibling `<vt-source-picker>` in
   *  the parent's template (typing in the shared typed-path input), so
   *  this component isn't on the ancestor-marked dirty path that
   *  produces; every exit writes at least one signal (``pathInputValue``
   *  / ``folderPath`` / ``browseError`` / ``detection``) which notifies
   *  the scheduler on its own, but ``datasetName`` is a plain field -
   *  ``markForCheck()`` covers it too. */
  onPathInput(value: string): void {
    this.pathInputValue.set(value);
    const raw = (value || '').trim();
    if (!raw) {
      this.folderPath.set('');
      this.browseError.set('');
      this.detection.set(null);
      if (!this.datasetNameDirty) {
        this.datasetName = '';
      }
      this.cdr.markForCheck();
      return;
    }
    this.folderPath.set(raw.replace(/\/+$/, '') || '/');
    if (!this.datasetNameDirty) {
      this.datasetName = this.derivedDatasetName();
    }
    this.cdr.markForCheck();
  }

  applyPathInput(): void {
    const raw = (this.pathInputValue() || '').trim();
    if (!raw) {
      this.folderPath.set('');
      this.browseError.set('');
      this.detection.set(null);
      if (!this.datasetNameDirty) {
        this.datasetName = '';
      }
      this.cdr.markForCheck();
      return;
    }
    this.folderPath.set(raw.replace(/\/+$/, '') || '/');
    this.browseError.set('');
    if (!this.datasetNameDirty) {
      this.datasetName = this.derivedDatasetName();
    }
    this.runDetection();
    this.cdr.markForCheck();
  }

  private runDetection(): void {
    const token = ++this.detectionToken;
    this.datasetsCrudApi.detectMediaType('server_fs', this.folderPath(), this.recursive).subscribe({
      next: (res) => {
        if (token !== this.detectionToken) return;
        this.detection.set(res);
        this.applyDetection();
      },
      error: () => {
        if (token !== this.detectionToken) return;
        this.detection.set(null);
      },
    });
  }

  private applyDetection(): void {
    const detection = this.detection();
    if (!detection) return;
    const { mediaType, sourceSpecs } = autofillFromDetection(this.mediaTypes(), detection, this.mediaTypeOptions, (typeId) =>
      availableConvertersFor(this.importers(), 'server_folder', typeId),
    );
    if (mediaType && mediaType !== this.mediaType()) {
      this.mediaType.set(mediaType);
      this.loadEmbedders(this.mediaType());
      this.loadClippers(this.mediaType());
    }
    if (sourceSpecs) {
      this.sourceSpecs.set(sourceSpecs);
    }
  }

  private derivedDatasetName(): string {
    const path = this.folderPath();
    if (!path) return '';
    const parts = path.split('/').filter(Boolean);
    return parts.length > 0 ? parts[parts.length - 1] : '';
  }

  onDatasetNameInput(value: string): void {
    this.datasetName = value;
    this.datasetNameDirty = true;
  }

  onRecursiveChange(recursive: boolean): void {
    this.recursive = recursive;
    if (this.folderPath()) {
      this.runDetection();
    }
  }

  onMediaTypeChange(mediaType: string): void {
    this.mediaType.set(mediaType);
    this.loadEmbedders(mediaType);
    this.loadClippers(mediaType);
    this.resetSourceSpecs();
  }

  get outputTypeId(): string {
    return toTypeId(this.mediaTypes(), this.mediaType());
  }

  get availableConverters(): ConverterInfo[] {
    return availableConvertersFor(this.importers(), 'server_folder', this.outputTypeId);
  }

  private resetSourceSpecs(): void {
    this.sourceSpecs.set(this.importDefaults.specsListWithDefaultsFor(this.mediaTypes(), this.outputTypeId, this.availableConverters));
  }

  onSpecsChange(specs: SourceSpec[]): void {
    this.sourceSpecs.set(specs);
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
    this.submitting.set(true);
    this.browseError.set('');

    const params: Record<string, unknown> = {
      path: this.folderPath(),
      media_type: this.mediaType(),
      recursive: this.recursive,
      dig_archives: this.digArchives,
      reference_files: this.referenceFiles,
    };
    const name = (this.datasetName || '').trim();
    if (name) {
      params['dataset_name'] = name;
    }
    if (this.selectedEmbedder()) {
      params['embedder'] = this.selectedEmbedder();
    }
    const embedders = composeEmbedders(this.selectedEmbedder(), this.selectedPatchEmbedder(), this.selectedStructuralEmbedder());
    if (embedders) {
      params['embedders'] = embedders;
    }
    if (this.selectedClipper()) {
      params['clipper'] = this.selectedClipper();
      if (this.clipperParams.length > 0 && Object.keys(this.clipperParamValues()).length > 0) {
        params['clipper_params'] = { ...this.clipperParamValues() };
      }
    }
    if (this.sourceSpecs().length > 0) {
      params['source_specs'] = this.sourceSpecs();
    }
    params['build_projection'] = this.buildProjection() ? 'true' : 'false';
    params['merge_near_duplicates'] = this.mergeNearDuplicates() ? 'true' : 'false';

    this.datasetsCrudApi.runImporter('server_folder', params).subscribe({
      next: () => {
        this.submitting.set(false);
        this.offerSaveImportDefaults();
        this.importStarted.emit();
      },
      error: (err) => {
        this.submitting.set(false);
        this.browseError.set(apiErrorMessage(err, 'Import failed'));
      },
    });
  }

  private offerSaveImportDefaults(): void {
    const typeId = this.outputTypeId;
    const cfg = this.importDefaults.snapshotImportConfig(
      typeId,
      this.selectedEmbedder(),
      this.selectedClipper(),
      this.clipperParamValues(),
      this.sourceSpecs(),
      this.embedders(),
      this.clippers(),
    );
    this.importDefaults.maybeOfferSaveImportDefaults(typeId, cfg, this.mediaTypes());
  }
}
