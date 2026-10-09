import { ChangeDetectionStrategy, Component, effect, inject, input, output, signal, untracked } from '@angular/core';

import { FormsModule } from '@angular/forms';

import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ConverterInfo,
  DetectMediaTypeResponse,
  EmbedderInfo,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../models/api.models';
import { DatasetsListingsApiService } from '../../../../services/datasets-listings-api.service';
import { IconComponent } from '../../../icon/icon.component';
import { ImportAdvancedComponent } from '../import-advanced/import-advanced.component';
import { ImportDefaultsService } from '../pickers/shared/import-defaults.service';
import { mediaTypeLabels } from '../pickers/shared/media-type.util';
import { OutputDraft, detectedCount, sourceRowsFor } from '../pickers/shared/multi-output.util';

/** What the per-row Advanced block needs to know about one dataset type:
 *  the embedders, clippers and cleanup gates registered for it. */
interface TypeOptions {
  embedders: EmbedderInfo[];
  clippers: ClipperInfo[];
  cleaners: CleanerInfo[];
}

/** The Add Dataset dialog's **Multi-Dataset** editor (#4703): one row per
 *  ingestion category, each ticked row a dataset of its own with its own
 *  Advanced block (include-media rows, embedders, clipper, cleanup).
 *
 *  The draft list is the parent picker's state (two-way bound, like the
 *  source-specs picker's `specs`); this component edits it and loads, per
 *  dataset type, the option lists the rows' Advanced blocks offer.  A row's
 *  defaults (embedder, clipper, cleanup, saved include rows) are seeded from
 *  the user's Import Defaults the first time its type's options arrive, the
 *  same way the single-dataset pickers seed theirs.
 *
 *  Each row hosts its own `<vt-import-advanced>`, opened by the row's
 *  "Details ▾" link rather than the modal's footer toggle (which the modal
 *  hides in multi mode: there is no single block for it to open).  The two
 *  ingest toggles that are shared across every dataset of the import render
 *  once, at the foot of the list, instead of inside every row. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-multi-output-config',
  standalone: true,
  imports: [FormsModule, IconComponent, ImportAdvancedComponent],
  templateUrl: './multi-output-config.component.html',
  styleUrl: './multi-output-config.component.scss',
})
export class MultiOutputConfigComponent {
  private listingsApi = inject(DatasetsListingsApiService);
  private importDefaults = inject(ImportDefaultsService);

  readonly mediaTypes = input<MediaTypeInfo[]>([]);
  /** The active importer's `available_converters_by_media_type`. */
  readonly convertersByType = input<Record<string, ConverterInfo[]>>({});
  /** Two-way bound draft list, one per category (see `buildOutputDrafts`). */
  readonly drafts = input<OutputDraft[]>([]);
  readonly draftsChange = output<OutputDraft[]>();
  /** The folder scan's result, for the per-row file counts. */
  readonly detection = input<DetectMediaTypeResponse | null>(null);
  readonly guessedMediaEmbedder = input('');
  /** Prefix for the rows' element ids, unique per flow so several flows can
   *  be mounted at once without duplicate ids. */
  readonly idPrefix = input('mo');

  /** The two ingest toggles shared across every dataset of the import. */
  readonly buildProjection = input(false);
  readonly buildProjectionChange = output<boolean>();
  readonly mergeNearDuplicates = input(false);
  readonly mergeNearDuplicatesChange = output<boolean>();

  /** A row's clipper "Details" was clicked: the parent opens its clipper
   *  chooser for that category, with the clippers of the row's dataset type. */
  readonly clipperChooserRequested = output<{ category: string; clippers: ClipperInfo[] }>();

  /** Option lists per dataset `type_id`, loaded on demand. */
  readonly typeOptions = signal<Record<string, TypeOptions>>({});
  private loading = new Set<string>();

  constructor() {
    // A row ticked from outside (the folder scan pre-ticks what it found)
    // needs its type's options too; load them the moment a ticked row's type
    // is new.
    effect(() => {
      const drafts = this.drafts();
      untracked(() => {
        for (const d of drafts) if (d.checked) this.ensureLoaded(d.mediaType);
      });
    });
  }

  get typeLabels(): Record<string, string> {
    return mediaTypeLabels(this.mediaTypes());
  }

  category(draft: OutputDraft): MediaTypeInfo | undefined {
    return this.mediaTypes().find((m) => m.type_id === draft.category);
  }

  labelFor(draft: OutputDraft): string {
    return this.category(draft)?.name.trim() || draft.category;
  }

  iconFor(draft: OutputDraft): string {
    return this.category(draft)?.icon || '';
  }

  rowId(draft: OutputDraft): string {
    return `${this.idPrefix()}-output-${draft.category}`;
  }

  /** "12 files" when the scan found some; empty otherwise. */
  countHint(draft: OutputDraft): string {
    const n = detectedCount(this.detection(), draft.category);
    if (n <= 0) return '';
    return `${n} ${n === 1 ? 'file' : 'files'}`;
  }

  /** What the row's checkbox means, for its tooltip. */
  rowTitle(draft: OutputDraft): string {
    const mt = this.category(draft);
    const label = this.labelFor(draft);
    if (mt?.embeddable === false) {
      return `Make a dataset of the ${label.toLowerCase()} files, converted to ${this.typeLabels[draft.mediaType] || draft.mediaType} so they can be searched.`;
    }
    if (mt?.importable === false) {
      return `Make a ${label} dataset from what the converters below find in the other files (nothing of this type is read from disk directly).`;
    }
    return `Make a dataset of the ${label.toLowerCase()} files.`;
  }

  /** Whether the row is a convert-out category: a "Convert to" choice, no include rows. */
  isConvertOut(draft: OutputDraft): boolean {
    return this.category(draft)?.embeddable === false;
  }

  convertTargets(draft: OutputDraft): string[] {
    return this.isConvertOut(draft) ? this.category(draft)?.converts_to || [] : [];
  }

  nativeImportable(draft: OutputDraft): boolean {
    return this.category(draft)?.importable !== false;
  }

  /** Converters feeding the row's dataset type, for its include-media rows. */
  convertersFor(draft: OutputDraft): ConverterInfo[] {
    if (this.isConvertOut(draft)) return [];
    return this.convertersByType()[draft.mediaType] || [];
  }

  options(draft: OutputDraft): TypeOptions {
    return this.typeOptions()[draft.mediaType] || { embedders: [], clippers: [], cleaners: [] };
  }

  lockedEmbedder(draft: OutputDraft): string {
    return this.importDefaults.lockedEmbedderFor(draft.mediaType, this.mediaTypes(), this.options(draft).embedders);
  }

  get anyChecked(): boolean {
    return this.drafts().some((d) => d.checked);
  }

  toggle(draft: OutputDraft, checked: boolean): void {
    this.patch(draft.category, { checked });
    if (checked) this.ensureLoaded(draft.mediaType);
  }

  onConvertTargetChange(draft: OutputDraft, target: string): void {
    const mt = this.category(draft);
    if (!mt || target === draft.convertTarget) return;
    const rows = sourceRowsFor(mt, this.convertersByType(), target);
    // A new dataset type: its embedder / clipper / cleanup are chosen afresh
    // once that type's options arrive.
    this.patch(draft.category, {
      ...rows,
      embedder: '',
      patchEmbedder: '',
      structuralEmbedder: '',
      clipper: '',
      clipperParams: {},
      cleaners: [],
    });
    this.ensureLoaded(rows.mediaType);
  }

  onSourceSpecsChange(draft: OutputDraft, specs: SourceSpec[]): void {
    this.patch(draft.category, { sourceSpecs: specs });
  }

  onEmbedderChange(draft: OutputDraft, embedder: string): void {
    this.patch(draft.category, { embedder });
  }

  onPatchEmbedderChange(draft: OutputDraft, patchEmbedder: string): void {
    this.patch(draft.category, { patchEmbedder });
  }

  onStructuralEmbedderChange(draft: OutputDraft, structuralEmbedder: string): void {
    this.patch(draft.category, { structuralEmbedder });
  }

  onCleanersChange(draft: OutputDraft, cleaners: CleanerSelection[]): void {
    this.patch(draft.category, { cleaners });
  }

  onClipperRequested(draft: OutputDraft): void {
    this.clipperChooserRequested.emit({ category: draft.category, clippers: this.options(draft).clippers });
  }

  /** Apply the parent's clipper-chooser result to the row it was opened for. */
  setClipper(category: string, name: string, params: Record<string, number | string>): void {
    this.patch(category, { clipper: name, clipperParams: { ...params } });
  }

  private patch(category: string, changes: Partial<OutputDraft>): void {
    this.draftsChange.emit(this.drafts().map((d) => (d.category === category ? { ...d, ...changes } : d)));
  }

  /** Fetch *mediaType*'s embedders / clippers / cleaners once, then seed the
   *  defaults of every row of that type still untouched. */
  private ensureLoaded(mediaType: string): void {
    if (!mediaType || this.typeOptions()[mediaType] || this.loading.has(mediaType)) return;
    this.loading.add(mediaType);
    const loaded: Partial<TypeOptions> = {};
    const settle = () => {
      if (!loaded.embedders || !loaded.clippers || !loaded.cleaners) return;
      const options = loaded as TypeOptions;
      this.loading.delete(mediaType);
      this.typeOptions.update((all) => ({ ...all, [mediaType]: options }));
      this.seedDefaults(mediaType, options);
    };
    this.listingsApi.getEmbedders(mediaType).subscribe({
      next: (embedders) => {
        loaded.embedders = embedders || [];
        settle();
      },
      error: () => {
        loaded.embedders = [];
        settle();
      },
    });
    this.listingsApi.getClippers(mediaType).subscribe({
      next: (clippers) => {
        loaded.clippers = clippers || [];
        settle();
      },
      error: () => {
        loaded.clippers = [];
        settle();
      },
    });
    this.listingsApi.getCleaners(mediaType).subscribe({
      next: (cleaners) => {
        loaded.cleaners = cleaners || [];
        settle();
      },
      error: () => {
        loaded.cleaners = [];
        settle();
      },
    });
  }

  /** Seed the Import Defaults into every row of *mediaType* that has no
   *  embedder yet - the marker for "the user has not touched this row". */
  private seedDefaults(mediaType: string, options: TypeOptions): void {
    const mediaTypes = this.mediaTypes();
    const next = this.drafts().map((d) => {
      if (d.mediaType !== mediaType || d.embedder) return d;
      const clipper = this.importDefaults.chooseClipperForType(options.clippers, mediaType, mediaTypes);
      const clipperParams = clipper.params ?? this.defaultClipperParams(options.clippers, clipper.name);
      const sourceSpecs =
        this.isConvertOut(d) || !this.nativeImportable(d)
          ? d.sourceSpecs
          : this.importDefaults.specsListWithDefaultsFor(mediaTypes, mediaType, this.convertersFor(d));
      return {
        ...d,
        embedder: this.importDefaults.chooseEmbedderForType(options.embedders, mediaType, mediaTypes, this.guessedMediaEmbedder()),
        clipper: clipper.name,
        clipperParams,
        cleaners: this.importDefaults.defaultCleanerSelection(options.cleaners),
        sourceSpecs,
      };
    });
    if (next.some((d, i) => d !== this.drafts()[i])) this.draftsChange.emit(next);
  }

  private defaultClipperParams(clippers: ClipperInfo[], name: string): Record<string, number | string> {
    const out: Record<string, number | string> = {};
    for (const p of clippers.find((c) => c.name === name)?.parameters || []) out[p.key] = p.default;
    return out;
  }
}
