import { ChangeDetectionStrategy, Component, computed, inject, signal, input, output, viewChild } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { map } from 'rxjs/operators';
import { ClipperChooserComponent, ClipperSelection } from '../../../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from '../../import-advanced/import-advanced.component';
import {
  FolderBrowserComponent,
  FolderBrowserBrowseFn,
  FolderBrowserPathChange,
} from '../../../../folder-browser/folder-browser.component';
import { DatasetsCrudApiService } from '../../../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../../../services/datasets-listings-api.service';
import { DatasetsUiApiService } from '../../../../../services/datasets-ui-api.service';
import { apiErrorMessage } from '../../../../../utils/api-error';
import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  ImporterInfo,
  DetectMediaTypeResponse,
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
  singleDatasetMediaTypes,
  toFolderName,
  toTypeId,
} from '../shared/media-type.util';
import {
  OutputDraft,
  buildOutputDrafts,
  checkedKindsSummary,
  convertersByTypeOf,
  isMultiOutputImporter,
  outputsFromDrafts,
  tickCategory,
  tickDetectedCategories,
} from '../shared/multi-output.util';
import { MultiOutputConfigComponent } from '../../multi-output-config/multi-output-config.component';

/** Server-folder picker view: user types (or browses to) an absolute
 *  path on the server filesystem; the backend auto-detects the media
 *  type and the picker offers converter rows for any secondary types
 *  found in the sample.
 *
 *  Sits behind the shared ``<vt-source-picker>`` chrome (rendered by
 *  the parent modal), projected into its ``sfAfter`` slot.  This
 *  component owns the whole form: the "Folder to import" field (typed
 *  path + Browse button + inline folder browser), the checkboxes, the
 *  dataset name, the Advanced block, and the submit logic.  The
 *  media-type select above it is projected into ``sfBefore`` by the
 *  parent. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-server-folder-picker',
  standalone: true,
  imports: [FormsModule, ImportAdvancedComponent, MultiOutputConfigComponent, ClipperChooserComponent, FolderBrowserComponent],
  templateUrl: './server-folder-picker.component.html',
  styleUrl: './server-folder-picker.component.scss',
})
export class ServerFolderPickerComponent {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private datasetsUiApi = inject(DatasetsUiApiService);
  private importDefaults = inject(ImportDefaultsService);

  /** This view's "Advanced" block. The Add Dataset modal reads it to render
   *  the block's toggle in its footer row (#4305). */
  readonly importAdvanced = viewChild(ImportAdvancedComponent);
  /** The Multi-Dataset editor, projected into the Advanced block in place of
   *  the single dataset's fields while :prop:`multiDataset` is on (#4703,
   *  #5230). */
  readonly multiOutputConfig = viewChild(MultiOutputConfigComponent);

  readonly importers = input<ImporterInfo[]>([]);
  readonly mediaTypes = input<MediaTypeInfo[]>([]);
  readonly guessedMediaType = input('');
  readonly guessedMediaEmbedder = input('');

  readonly buildProjection = input(false);
  readonly buildProjectionChange = output<boolean>();
  readonly mergeNearDuplicates = input(false);
  readonly mergeNearDuplicatesChange = output<boolean>();
  /** The Add Dataset dialog's "Run AutoFind" choice for this import, sent as
   *  ``autofind``; ``null`` (the box is hidden) sends nothing, so the user's
   *  remembered ``autofind_on_import`` setting decides server-side. */
  readonly autofind = input<boolean | null>(null);

  readonly importStarted = output<void>();
  /** Fired with the dataset type whose Import Defaults the picker has just
   *  applied, so the modal, which owns the two toggles above, can apply that
   *  type's saved ones. */
  readonly typeDefaultsApplied = output<string>();

  readonly selectedImporter = signal<ImporterInfo | null>(null);

  /** Current value of the editable absolute-path input rendered by this
   *  component's own "Folder to import" field. */
  readonly pathInputValue = signal('');
  /** Committed absolute server path (trailing slash trimmed); read by
   *  the parent's footer submit button, hence also a signal. */
  readonly folderPath = signal('');
  readonly browseError = signal('');
  readonly mediaType = signal('');
  readonly mediaTypeOptions = signal<string[]>([]);
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
  /** The selected clipper's parameters, whichever way it was picked (the
   *  type's default, or the clipper chooser). */
  readonly clipperParams = computed<ClipperParameter[]>(
    () => this.clippers().find((c) => c.name === this.selectedClipper())?.parameters || [],
  );
  readonly clipperParamValues = signal<Record<string, number | string>>({});
  readonly submitting = signal(false);
  /** Whether subdirectories of the picked server folder are scanned. */
  readonly recursive = signal(true);
  /** Whether archives inside the picked folder are extracted and imported. */
  readonly digArchives = signal(false);
  /** Whether the dataset references the original files in place. */
  readonly referenceFiles = signal(false);
  /** Follows the picked folder's name until the user types a name of their
   *  own (``datasetNameDirty``). */
  readonly datasetName = signal('');
  private datasetNameDirty = false;
  readonly sourceSpecs = signal<SourceSpec[]>([]);
  readonly detection = signal<DetectMediaTypeResponse | null>(null);

  /** Multi-Dataset mode (#4703): one dataset per ticked category of
   *  :prop:`outputDrafts`, all from one walk of the folder.  Read by the
   *  parent modal (the footer toggle and the Import button), hence signals. */
  readonly multiDataset = signal(false);
  readonly outputDrafts = signal<OutputDraft[]>([]);
  /** The category whose row opened the clipper chooser, or ``null`` when the
   *  single-dataset Advanced block did. */
  private clipperChooserCategory: string | null = null;

  /** Whether the inline server-filesystem folder browser is expanded. */
  readonly browserOpen = signal(false);

  readonly clipperChooserOpen = signal(false);
  readonly clipperChooserClippers = signal<ClipperInfo[]>([]);

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
    this.recursive.set(readRecursiveDefault(importer));
    this.digArchives.set(false);
    this.referenceFiles.set(false);
    this.datasetName.set('');
    this.datasetNameDirty = false;
    this.multiDataset.set(false);
    this.outputDrafts.set([]);

    const folderImporter = this.importers().find((imp) => imp.name === 'server_folder');
    const mtField = folderImporter?.fields?.find((f) => f.key === 'media_type');
    const options = singleDatasetMediaTypes(this.mediaTypes(), mtField?.options || []);
    this.mediaTypeOptions.set(options);

    const guessedFolder = toFolderName(this.mediaTypes(), this.guessedMediaType());
    if (guessedFolder && options.includes(guessedFolder)) {
      this.mediaType.set(guessedFolder);
    } else {
      this.mediaType.set(mtField?.default || options[0] || 'audio');
    }

    if (this.effectiveSoloFolderName && options.includes(this.effectiveSoloFolderName)) {
      this.mediaType.set(this.effectiveSoloFolderName);
    }

    this.loadEmbedders(this.mediaType());
    this.loadClippers(this.mediaType());
    this.loadCleaners(this.mediaType());
    this.resetSourceSpecs();
    this.applyTypeDefaults();
  }

  onFolderPicked(evt: FolderBrowserPathChange): void {
    // The browser opens at the typed path (via `startPath`), or at the root
    // when that path can't be listed. Either way its opening listing is not a
    // pick, so it must not overwrite what the user typed or re-run detection.
    // With nothing typed there is nothing to lose, and the field takes the root.
    if (evt.initial && this.folderPath()) return;
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

  /** Typing in the "Folder to import" input. */
  onPathInput(value: string): void {
    this.pathInputValue.set(value);
    const raw = (value || '').trim();
    if (!raw) {
      this.folderPath.set('');
      this.browseError.set('');
      this.detection.set(null);
      if (!this.datasetNameDirty) {
        this.datasetName.set('');
      }
      return;
    }
    this.folderPath.set(raw.replace(/\/+$/, '') || '/');
    if (!this.datasetNameDirty) {
      this.datasetName.set(this.derivedDatasetName());
    }
  }

  applyPathInput(): void {
    const raw = (this.pathInputValue() || '').trim();
    if (!raw) {
      this.folderPath.set('');
      this.browseError.set('');
      this.detection.set(null);
      if (!this.datasetNameDirty) {
        this.datasetName.set('');
      }
      return;
    }
    this.folderPath.set(raw.replace(/\/+$/, '') || '/');
    this.browseError.set('');
    if (!this.datasetNameDirty) {
      this.datasetName.set(this.derivedDatasetName());
    }
    this.runDetection();
  }

  private runDetection(): void {
    const token = ++this.detectionToken;
    this.datasetsCrudApi.detectMediaType('server_fs', this.folderPath(), this.recursive()).subscribe({
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
    if (this.multiDataset()) {
      this.outputDrafts.set(tickDetectedCategories(this.outputDrafts(), detection));
    }
    const { mediaType, sourceSpecs } = autofillFromDetection(this.mediaTypes(), detection, this.mediaTypeOptions(), (typeId) =>
      availableConvertersFor(this.importers(), 'server_folder', typeId),
    );
    if (mediaType && mediaType !== this.mediaType()) {
      this.mediaType.set(mediaType);
      this.loadEmbedders(this.mediaType());
      this.loadClippers(this.mediaType());
      this.loadCleaners(this.mediaType());
      this.applyTypeDefaults();
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
    this.datasetName.set(value);
    this.datasetNameDirty = true;
  }

  onRecursiveChange(recursive: boolean): void {
    this.recursive.set(recursive);
    if (this.folderPath()) {
      this.runDetection();
    }
  }

  onMediaTypeChange(mediaType: string): void {
    this.mediaType.set(mediaType);
    this.loadEmbedders(mediaType);
    this.loadClippers(mediaType);
    this.loadCleaners(mediaType);
    this.resetSourceSpecs();
    this.applyTypeDefaults();
  }

  /** Apply what the current type's Import Defaults say outside the single
   *  dataset's fields (those load with their option lists): open the
   *  Multi-Dataset editor on the saved kinds, plus whatever the folder scan
   *  has found, and hand the type to the modal for the two whole-import
   *  toggles. */
  private applyTypeDefaults(): void {
    const typeId = this.outputTypeId;
    if (!this.multiDataset() && this.supportsMultiOutput && !this.effectiveSoloMediaType) {
      const fresh = buildOutputDrafts(this.mediaTypes(), this.convertersByType);
      const drafts = this.importDefaults.savedMultiDatasetDrafts(typeId, this.mediaTypes(), fresh);
      if (drafts) {
        this.multiDataset.set(true);
        this.outputDrafts.set(tickDetectedCategories(drafts, this.detection()));
      }
    }
    this.typeDefaultsApplied.emit(typeId);
  }

  get outputTypeId(): string {
    return toTypeId(this.mediaTypes(), this.mediaType());
  }

  get availableConverters(): ConverterInfo[] {
    return availableConvertersFor(this.importers(), 'server_folder', this.outputTypeId);
  }

  /** Whether the Multi-Dataset box is offered: the folder importer says so. */
  get supportsMultiOutput(): boolean {
    return isMultiOutputImporter(this.importers().find((imp) => imp.name === 'server_folder'));
  }

  get convertersByType(): Record<string, ConverterInfo[]> {
    return convertersByTypeOf(this.importers().find((imp) => imp.name === 'server_folder'));
  }

  /** Switch between the single-dataset form and the Multi-Dataset editor.
   *  Entering multi mode seeds one row per category, ticks the type the
   *  dropdown had picked, and ticks whatever the folder scan has found. */
  setMultiDataset(on: boolean): void {
    this.multiDataset.set(on);
    if (on && this.outputDrafts().length === 0) {
      const seeded = tickCategory(buildOutputDrafts(this.mediaTypes(), this.convertersByType), this.outputTypeId);
      this.outputDrafts.set(tickDetectedCategories(seeded, this.detection()));
    }
  }

  onOutputDraftsChange(drafts: OutputDraft[]): void {
    this.outputDrafts.set(drafts);
  }

  /** Whether the import has something to make: always in single mode, at
   *  least one ticked category in multi mode. */
  get hasOutputs(): boolean {
    return !this.multiDataset() || this.outputDrafts().some((d) => d.checked);
  }

  /** The ticked rows' kinds (``"Image, Document"``) for the line that stands
   *  in for the media-type dropdown in multi mode, the list itself being
   *  under Advanced (#5230). */
  get multiDatasetSummary(): string {
    return checkedKindsSummary(this.outputDrafts(), this.mediaTypes());
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

  /** Fetch the cleanup gates registered for *mediaType* and seed the
   *  selection from the saved import default, else each cleaner's
   *  ``default_enabled`` flag. */
  private loadCleaners(mediaType: string): void {
    if (!mediaType) {
      this.cleaners.set([]);
      this.selectedCleaners.set([]);
      return;
    }
    this.datasetsListingsApi.getCleaners(mediaType).subscribe({
      next: (cleaners) => {
        this.cleaners.set(cleaners);
        this.selectedCleaners.set(this.importDefaults.defaultCleanerSelection(cleaners, mediaType, this.mediaTypes()));
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
    const next: Record<string, number | string> = {};
    for (const param of this.clipperParams()) {
      next[param.key] = param.default;
    }
    this.clipperParamValues.set(next);
  }

  openClipperChooser(): void {
    this.clipperChooserCategory = null;
    this.clipperChooserClippers.set(this.clippers());
    this.clipperChooserOpen.set(true);
  }

  /** A Multi-Dataset row asked for the chooser: open it on that row's clippers. */
  openClipperChooserFor(request: { category: string; clippers: ClipperInfo[] }): void {
    this.clipperChooserCategory = request.category;
    this.clipperChooserClippers.set(request.clippers);
    this.clipperChooserOpen.set(true);
  }

  onClipperChooserSelected(selection: ClipperSelection): void {
    this.clipperChooserOpen.set(false);
    if (this.clipperChooserCategory !== null) {
      this.multiOutputConfig()?.setClipper(this.clipperChooserCategory, selection.name, selection.params);
      this.clipperChooserCategory = null;
      return;
    }
    this.selectedClipper.set(selection.name);
    this.clipperParamValues.set({ ...selection.params });
  }

  onClipperChooserCancelled(): void {
    this.clipperChooserOpen.set(false);
    if (this.clipperChooserCategory !== null) {
      // The row keeps the clipper it had.
      this.clipperChooserCategory = null;
      return;
    }
    const clippers = this.clipperChooserClippers();
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
      recursive: this.recursive(),
      dig_archives: this.digArchives(),
      reference_files: this.referenceFiles(),
    };
    const name = this.datasetName().trim();
    if (name) {
      params['dataset_name'] = name;
    }
    if (this.multiDataset()) {
      // One dataset per ticked category, each row with its own settings; the
      // single-dataset embedder / clipper / cleanup / include rows stay home.
      const outputs = outputsFromDrafts(this.outputDrafts());
      params['outputs'] = outputs;
      params['media_type'] = toFolderName(this.mediaTypes(), outputs[0]?.media_type || '') || this.mediaType();
    } else {
      if (this.selectedEmbedder()) {
        params['embedder'] = this.selectedEmbedder();
      }
      const embedders = composeEmbedders(this.selectedEmbedder(), this.selectedPatchEmbedder(), this.selectedStructuralEmbedder());
      if (embedders) {
        params['embedders'] = embedders;
      }
      if (this.selectedClipper()) {
        params['clipper'] = this.selectedClipper();
        if (this.clipperParams().length > 0 && Object.keys(this.clipperParamValues()).length > 0) {
          params['clipper_params'] = { ...this.clipperParamValues() };
        }
      }
      if (this.selectedCleaners().length > 0) {
        params['cleaners'] = this.selectedCleaners();
      }
      if (this.sourceSpecs().length > 0) {
        params['source_specs'] = this.sourceSpecs();
      }
    }
    params['build_projection'] = this.buildProjection() ? 'true' : 'false';
    params['merge_near_duplicates'] = this.mergeNearDuplicates() ? 'true' : 'false';
    const autofind = this.autofind();
    if (autofind !== null) params['autofind'] = autofind ? 'true' : 'false';

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
    const cfg = this.multiDataset()
      ? this.importDefaults.snapshotMultiDatasetConfig(
          typeId,
          this.mediaTypes(),
          this.outputDrafts(),
          this.buildProjection(),
          this.mergeNearDuplicates(),
        )
      : {
          ...this.importDefaults.snapshotImportConfig(
            typeId,
            this.selectedEmbedder(),
            this.selectedClipper(),
            this.clipperParamValues(),
            this.sourceSpecs(),
            this.embedders(),
            this.clippers(),
            this.selectedCleaners(),
            this.cleaners(),
          ),
          ...this.importDefaults.snapshotIngestToggles(this.buildProjection(), this.mergeNearDuplicates()),
        };
    this.importDefaults.maybeOfferSaveImportDefaults(typeId, cfg, this.mediaTypes());
  }
}
