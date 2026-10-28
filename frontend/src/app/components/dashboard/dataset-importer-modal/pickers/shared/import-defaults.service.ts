import { Injectable, inject } from '@angular/core';

import { SettingsStateService } from '../../../../../services/settings-state.service';
import { ToastService } from '../../../../../services/toast.service';
import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsForMediaType,
  MediaTypeInfo,
  SavedCleanerChoice,
  SourceSpec,
} from '../../../../../models/api.models';
import { defaultSpecListFor, getTabLabel, toTypeId } from './media-type.util';
import { OutputDraft, categoryLabel } from './multi-output.util';

/** Shallow-equality check for clipper-param dicts - both are flat
 *  ``{string: string|number}`` maps. */
function shallowParamsEqual(
  a: Record<string, string | number> | undefined,
  b: Record<string, string | number> | undefined,
): boolean {
  const ka = Object.keys(a || {});
  const kb = Object.keys(b || {});
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if ((a as Record<string, unknown>)[k] !== (b as Record<string, unknown>)[k]) {
      return false;
    }
  }
  return true;
}

/** Order-insensitive deep-ish equality for source-spec lists - compares
 *  each row by ``source_type`` + ``converter`` + params. */
function shallowSpecsEqual(a: SourceSpec[] | undefined, b: SourceSpec[] | undefined): boolean {
  const la = a || [];
  const lb = b || [];
  if (la.length !== lb.length) return false;
  const key = (s: SourceSpec) => `${s.source_type}|${s.converter ?? ''}`;
  const sortedA = [...la].sort((x, y) => key(x).localeCompare(key(y)));
  const sortedB = [...lb].sort((x, y) => key(x).localeCompare(key(y)));
  for (let i = 0; i < sortedA.length; i++) {
    if (sortedA[i].source_type !== sortedB[i].source_type) return false;
    if ((sortedA[i].converter ?? null) !== (sortedB[i].converter ?? null)) return false;
    if (
      !shallowParamsEqual(
        sortedA[i].params as Record<string, string | number>,
        sortedB[i].params as Record<string, string | number>,
      )
    ) {
      return false;
    }
  }
  return true;
}

/** Equality for saved cleanup choices: same cleaner names, same enabled
 *  flags, and - for an enabled cleaner - the same parameter overrides. */
function cleanerChoicesEqual(
  a: Record<string, SavedCleanerChoice> | undefined,
  b: Record<string, SavedCleanerChoice> | undefined,
): boolean {
  const ka = Object.keys(a || {});
  if (ka.length !== Object.keys(b || {}).length) return false;
  for (const name of ka) {
    const x = (a as Record<string, SavedCleanerChoice>)[name];
    const y = b?.[name];
    if (!y || !!x.enabled !== !!y.enabled) return false;
    if (x.enabled && !shallowParamsEqual(x.params, y.params)) return false;
  }
  return true;
}

/** Order-insensitive equality for saved Multi-Dataset kinds. */
function kindsEqual(a: string[] | undefined, b: string[] | undefined): boolean {
  const sa = new Set(a || []);
  const sb = new Set(b || []);
  return sa.size === sb.size && [...sa].every((k) => sb.has(k));
}

/** Join a list as prose: ``"a"``, ``"a and b"``, ``"a, b and c"``. */
function proseList(items: string[]): string {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/** Per-user "sticky" import config: solo mediaType/embedder locks, saved
 *  embedder/clipper/converter/cleanup/ingest-toggle/Multi-Dataset defaults per
 *  media type, and the post-import "save these as default?" toast offer.
 *
 *  Shared by every picker view (generic form / server-folder /
 *  local-folder / demo) of the Add Dataset modal - each view used to
 *  carry its own copy of this logic under a ``form*`` / ``sf*`` / ``lf*``
 *  prefix. */
@Injectable({ providedIn: 'root' })
export class ImportDefaultsService {
  private settingsState = inject(SettingsStateService);
  private toastService = inject(ToastService);

  /** Type_id (e.g. ``"image"``) of the server's solo-mediaType
   *  restriction, or ``null`` when not active. */
  get effectiveSoloMediaType(): string | null {
    const v = this.settingsState.settingsSignal()?.solo_media_type;
    return v ? v : null;
  }

  /** Folder name (e.g. ``"images"``) of the solo mediaType, or ``""``
   *  if not active. */
  effectiveSoloFolderName(mediaTypes: MediaTypeInfo[]): string {
    const tid = this.effectiveSoloMediaType;
    if (!tid) return '';
    const mt = mediaTypes.find((m) => m.type_id === tid);
    return mt?.folder_import_name || tid;
  }

  /** Resolve the Solo mediaEmbedder lock for a mediaType. Returns the
   *  embedder name when a lock is set AND that embedder is currently
   *  registered for the type. */
  lockedEmbedderFor(mediaTypeFolderOrTypeId: string, mediaTypes: MediaTypeInfo[], embedders: EmbedderInfo[]): string {
    if (!mediaTypeFolderOrTypeId) return '';
    const typeId = toTypeId(mediaTypes, mediaTypeFolderOrTypeId) || mediaTypeFolderOrTypeId;
    const effectiveMap = this.settingsState.settingsSignal()?.effective_solo_embedder_per_media_type || {};
    const locked = effectiveMap[typeId];
    if (!locked) return '';
    return embedders.find((e) => e.name === locked) ? locked : '';
  }

  /** Pick the initial embedder for a picker view.  Priority:
   *  1. Solo mediaEmbedder lock for this mediaType.
   *  2. ``guessedMediaEmbedder`` (computed from currently loaded datasets)
   *  3. the user's last pick for this media type (per-user setting)
   *  4. first option, or empty when the list is empty. */
  pickInitialEmbedder(
    embedders: EmbedderInfo[],
    mediaTypeFolderOrTypeId: string,
    mediaTypes: MediaTypeInfo[],
    guessedMediaEmbedder: string,
  ): string {
    if (embedders.length === 0) return '';
    const locked = this.lockedEmbedderFor(mediaTypeFolderOrTypeId, mediaTypes, embedders);
    if (locked) return locked;
    const guessedMatch = guessedMediaEmbedder ? embedders.find((e) => e.name === guessedMediaEmbedder) : null;
    if (guessedMatch) return guessedMatch.name;
    const typeId = toTypeId(mediaTypes, mediaTypeFolderOrTypeId) || mediaTypeFolderOrTypeId;
    const savedMap = this.settingsState.settingsSignal()?.last_embedder_per_media_type || {};
    const saved = savedMap[typeId];
    if (saved) {
      const savedMatch = embedders.find((e) => e.name === saved);
      if (savedMatch) return savedMatch.name;
    }
    return embedders[0].name;
  }

  private importDefaultsForFolderOrTypeId(
    folderOrTypeId: string,
    mediaTypes: MediaTypeInfo[],
  ): ImportDefaultsForMediaType | null {
    if (!folderOrTypeId) return null;
    const typeId = toTypeId(mediaTypes, folderOrTypeId) || folderOrTypeId;
    const map = ((this.settingsState.settingsSignal() as Record<string, unknown> | undefined)?.[
      'import_defaults_by_media_type'
    ] || {}) as Record<string, ImportDefaultsForMediaType>;
    return map[typeId] || null;
  }

  /** Apply saved embedder default for *mediaType* if one is set and is
   *  still in the *embedders* list; otherwise fall back to the usual
   *  pickInitialEmbedder priority. Returns the chosen name. */
  chooseEmbedderForType(
    embedders: EmbedderInfo[],
    mediaType: string,
    mediaTypes: MediaTypeInfo[],
    guessedMediaEmbedder: string,
  ): string {
    const saved = this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes)?.embedder;
    if (saved && embedders.some((e) => e.name === saved)) return saved;
    return this.pickInitialEmbedder(embedders, mediaType, mediaTypes, guessedMediaEmbedder);
  }

  /** Apply saved clipper default for *mediaType*. Returns
   *  ``{ name, params }`` so the caller can update both selection and
   *  parameter values together. Falls back to the registry's default
   *  (first entry) when no saved default exists or it's no longer in
   *  the clipper list. */
  chooseClipperForType(
    clippers: ClipperInfo[],
    mediaType: string,
    mediaTypes: MediaTypeInfo[],
  ): { name: string; params: Record<string, number | string> | null } {
    const saved = this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes);
    if (saved?.clipper && clippers.some((c) => c.name === saved.clipper)) {
      return { name: saved.clipper, params: { ...(saved.clipper_params || {}) } };
    }
    return {
      name: clippers.length > 0 ? clippers[0].name : '',
      params: null, // signals "use the clipper's own param defaults"
    };
  }

  /** Cleanup selection for a freshly-fetched cleaner list, in registry
   *  order.  A cleaner the user's saved default for *mediaType* names takes
   *  the saved choice (enabled flag plus params, dropping any param the
   *  cleaner no longer declares); any other cleaner - all of them when
   *  nothing is saved, or one registered since the save - follows its
   *  ``default_enabled``.  Saved names no longer registered are ignored. */
  defaultCleanerSelection(cleaners: CleanerInfo[], mediaType: string, mediaTypes: MediaTypeInfo[]): CleanerSelection[] {
    const saved = this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes)?.cleaners || {};
    const out: CleanerSelection[] = [];
    for (const c of cleaners) {
      const choice = saved[c.name];
      if (!(choice ? choice.enabled : c.default_enabled)) continue;
      const declared = new Set((c.parameters || []).map((p) => p.key));
      const params = Object.fromEntries(Object.entries(choice?.params || {}).filter(([k]) => declared.has(k)));
      out.push({ name: c.name, params });
    }
    return out;
  }

  /** Build a source-specs list for a freshly-opened picker, preferring
   *  the user's saved defaults (filtered to converters that the active
   *  importer actually supports) and falling back to the importer's
   *  bare "include directly" row. */
  specsListWithDefaultsFor(
    mediaTypes: MediaTypeInfo[],
    outputTypeId: string,
    availableConverters: ConverterInfo[],
  ): SourceSpec[] {
    const fallback = defaultSpecListFor(outputTypeId);
    if (!outputTypeId) return fallback;
    const saved = this.importDefaultsForFolderOrTypeId(outputTypeId, mediaTypes)?.source_specs;
    if (!saved || saved.length === 0) return fallback;
    const validConverterNames = new Set(availableConverters.map((c) => c.name));
    const filtered = saved.filter((s) => s.converter === null || validConverterNames.has(s.converter));
    if (filtered.length === 0) return fallback;
    const hasNative = filtered.some((s) => s.source_type === outputTypeId && s.converter === null);
    if (!hasNative) {
      return [{ source_type: outputTypeId, converter: null, params: {} }, ...filtered];
    }
    return filtered;
  }

  /** The Include-media rows saved for *mediaType*, verbatim.  For a flow
   *  with no Include-media picker of its own (the demo) to carry into its
   *  snapshot, so that saving from it leaves them as they were. */
  savedSourceSpecs(mediaType: string, mediaTypes: MediaTypeInfo[]): SourceSpec[] | undefined {
    return this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes)?.source_specs;
  }

  /** The two whole-import toggles ("Build Browse map now", "Merge
   *  near-duplicates") saved for *mediaType*; both off when nothing is saved. */
  ingestTogglesFor(
    mediaType: string,
    mediaTypes: MediaTypeInfo[],
  ): { buildProjection: boolean; mergeNearDuplicates: boolean } {
    const saved = this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes);
    return { buildProjection: !!saved?.build_projection, mergeNearDuplicates: !!saved?.merge_near_duplicates };
  }

  /** *drafts* (a picker's fresh Multi-Dataset rows) with exactly the kinds
   *  saved for *mediaType* ticked, or ``null`` when there is no saved
   *  Multi-Dataset default, or none of its kinds is a row here (a server
   *  without the face models, say), so the picker stays on the single form. */
  savedMultiDatasetDrafts(mediaType: string, mediaTypes: MediaTypeInfo[], drafts: OutputDraft[]): OutputDraft[] | null {
    const kinds = new Set(this.importDefaultsForFolderOrTypeId(mediaType, mediaTypes)?.multi_dataset || []);
    if (!drafts.some((d) => kinds.has(d.category))) return null;
    return drafts.map((d) => ({ ...d, checked: kinds.has(d.category) }));
  }

  /** The two whole-import toggles as saved defaults: a key only for a box
   *  that is ticked, so an untouched pair adds nothing. */
  snapshotIngestToggles(buildProjection: boolean, mergeNearDuplicates: boolean): ImportDefaultsForMediaType {
    const out: ImportDefaultsForMediaType = {};
    if (buildProjection) out.build_projection = true;
    if (mergeNearDuplicates) out.merge_near_duplicates = true;
    return out;
  }

  /** Snapshot of a Multi-Dataset import as *typeId*'s default: the kinds it
   *  ticked and the two toggles.  Each row's embedder / clipper / cleanup is
   *  not part of it (rows seed those from their own type's defaults), and the
   *  type's saved single-dataset settings ride along unchanged, for when the
   *  user turns Multi-Dataset back off. */
  snapshotMultiDatasetConfig(
    typeId: string,
    mediaTypes: MediaTypeInfo[],
    drafts: OutputDraft[],
    buildProjection: boolean,
    mergeNearDuplicates: boolean,
  ): ImportDefaultsForMediaType {
    const single: ImportDefaultsForMediaType = { ...(this.importDefaultsForFolderOrTypeId(typeId, mediaTypes) || {}) };
    delete single.build_projection;
    delete single.merge_near_duplicates;
    delete single.multi_dataset;
    return {
      ...single,
      multi_dataset: drafts.filter((d) => d.checked).map((d) => d.category),
      ...this.snapshotIngestToggles(buildProjection, mergeNearDuplicates),
    };
  }

  /** Snapshot of the per-view import-config (embedder/clipper/specs/cleanup)
   *  the user submitted, normalised to the same shape as the persisted
   *  defaults.  The two ingest toggles are added by the caller, through
   *  :meth:`snapshotIngestToggles`. */
  snapshotImportConfig(
    typeId: string,
    embedder: string,
    clipper: string,
    clipperParams: Record<string, number | string>,
    sourceSpecs: SourceSpec[],
    availableEmbedders: EmbedderInfo[],
    availableClippers: ClipperInfo[],
    selectedCleaners: CleanerSelection[],
    availableCleaners: CleanerInfo[],
  ): ImportDefaultsForMediaType {
    const out: ImportDefaultsForMediaType = {};
    const isDefaultEmbedder = !embedder || !!availableEmbedders.find((e) => e.name === embedder)?.is_default;
    if (embedder && !isDefaultEmbedder) {
      out.embedder = embedder;
    }
    const isDefaultClipper =
      !clipper || clipper.endsWith('_default') || (availableClippers.length > 0 && availableClippers[0].name === clipper);
    if (clipper && !isDefaultClipper) {
      out.clipper = clipper;
      if (clipperParams && Object.keys(clipperParams).length > 0) {
        out.clipper_params = { ...clipperParams };
      }
    }
    const nonNative = (sourceSpecs || []).filter((s) => !(s.source_type === typeId && s.converter === null));
    if (nonNative.length > 0) {
      out.source_specs = nonNative.map((s) => ({ ...s, params: { ...s.params } }));
    }
    const cleaners = this.cleanerChoicesIfNonDefault(selectedCleaners, availableCleaners);
    if (cleaners) out.cleaners = cleaners;
    return out;
  }

  /** The cleanup selection as saved choices for every listed cleaner, or
   *  ``null`` when it is the registry's own (each ``default_enabled``
   *  cleaner on with no param overrides, every other one off). */
  private cleanerChoicesIfNonDefault(
    selected: CleanerSelection[],
    available: CleanerInfo[],
  ): Record<string, SavedCleanerChoice> | null {
    const byName = new Map((selected || []).map((s) => [s.name, s]));
    const hasParams = (s: CleanerSelection | undefined) => !!s && Object.keys(s.params || {}).length > 0;
    const isRegistryDefault = available.every((c) => {
      const sel = byName.get(c.name);
      return c.default_enabled ? !!sel && !hasParams(sel) : !sel;
    });
    if (isRegistryDefault) return null;
    const out: Record<string, SavedCleanerChoice> = {};
    for (const c of available) {
      const sel = byName.get(c.name);
      out[c.name] = sel && hasParams(sel) ? { enabled: true, params: { ...sel.params } } : { enabled: !!sel };
    }
    return out;
  }

  private hasMeaningfulOverrides(cfg: ImportDefaultsForMediaType): boolean {
    return !!(
      cfg.embedder ||
      cfg.clipper ||
      (cfg.source_specs && cfg.source_specs.length > 0) ||
      (cfg.cleaners && Object.keys(cfg.cleaners).length > 0) ||
      cfg.build_projection ||
      cfg.merge_near_duplicates ||
      (cfg.multi_dataset && cfg.multi_dataset.length > 0)
    );
  }

  private importDefaultsEqual(a: ImportDefaultsForMediaType, b: ImportDefaultsForMediaType): boolean {
    if ((a.embedder || '') !== (b.embedder || '')) return false;
    if ((a.clipper || '') !== (b.clipper || '')) return false;
    if (!shallowParamsEqual(a.clipper_params, b.clipper_params)) return false;
    if (!shallowSpecsEqual(a.source_specs, b.source_specs)) return false;
    if (!cleanerChoicesEqual(a.cleaners, b.cleaners)) return false;
    if (!!a.build_projection !== !!b.build_projection) return false;
    if (!!a.merge_near_duplicates !== !!b.merge_near_duplicates) return false;
    if (!kindsEqual(a.multi_dataset, b.multi_dataset)) return false;
    return true;
  }

  /** What saving *cfg* would make the next import start with, for the
   *  offer's detail line.  A Multi-Dataset config names its kinds but not the
   *  single-dataset settings it carries over unchanged; *dropsMulti* says the
   *  saved default it replaces opened in Multi-Dataset mode. */
  private describeImportDefaults(cfg: ImportDefaultsForMediaType, dropsMulti: boolean, mediaTypes: MediaTypeInfo[]): string {
    const parts: string[] = [];
    if (cfg.multi_dataset && cfg.multi_dataset.length > 0) {
      parts.push(`Multi-Dataset (${cfg.multi_dataset.map((k) => categoryLabel(mediaTypes, k)).join(', ')})`);
    } else {
      if (dropsMulti) parts.push('one dataset, not Multi-Dataset');
      const picks: string[] = [];
      if (cfg.embedder) picks.push('embedder');
      if (cfg.clipper) picks.push('clipper');
      if (cfg.source_specs && cfg.source_specs.length > 0) picks.push('included media');
      if (cfg.cleaners && Object.keys(cfg.cleaners).length > 0) picks.push('cleanup');
      if (picks.length > 0) parts.push(`your ${proseList(picks)} picks`);
    }
    if (cfg.build_projection) parts.push('Build Browse map now');
    if (cfg.merge_near_duplicates) parts.push('Merge near-duplicates');
    return `Next time it starts with ${proseList(parts)}.`;
  }

  /** After a successful import, offer to save the user's advanced
   *  settings as the default for this output mediaType. Skipped silently
   *  when the user accepted the importer's natural defaults or already
   *  has the same config saved - except that a one-dataset import of a type
   *  whose saved default opens Multi-Dataset is offered too, since the user
   *  turned Multi-Dataset off to make it. */
  maybeOfferSaveImportDefaults(typeId: string, cfg: ImportDefaultsForMediaType, mediaTypes: MediaTypeInfo[]): void {
    if (!typeId) return;
    const saved = this.importDefaultsForFolderOrTypeId(typeId, mediaTypes) || {};
    const dropsMulti = !!saved.multi_dataset?.length && !cfg.multi_dataset?.length;
    if (!this.hasMeaningfulOverrides(cfg) && !dropsMulti) return;
    if (this.importDefaultsEqual(saved, cfg)) return;
    const typeLabel = getTabLabel(mediaTypes, typeId) || typeId;
    this.toastService.success({
      message: `Save these as default for ${typeLabel} imports?`,
      detail: this.describeImportDefaults(cfg, dropsMulti, mediaTypes),
      action: {
        label: 'Save as Default',
        onClick: () => this.persistImportDefaults(typeId, cfg, mediaTypes),
      },
      dedupKey: `import-defaults-offer:${typeId}`,
    });
  }

  /** Merge a new per-mediaType entry into the persisted import-defaults
   *  map and push it through ``SettingsStateService`` so the next
   *  importer open picks it up. */
  private persistImportDefaults(typeId: string, cfg: ImportDefaultsForMediaType, mediaTypes: MediaTypeInfo[]): void {
    const current = (this.settingsState.settingsSignal() as Record<string, unknown> | undefined)?.[
      'import_defaults_by_media_type'
    ] as Record<string, ImportDefaultsForMediaType> | undefined;
    const next: Record<string, ImportDefaultsForMediaType> = { ...(current || {}) };
    if (Object.keys(cfg).length > 0) {
      next[typeId] = cfg;
    } else {
      delete next[typeId];
    }
    this.settingsState.update({ import_defaults_by_media_type: next } as Record<string, unknown>).subscribe({
      next: () => {
        this.toastService.success({
          message: 'Saved as default',
          detail: `Future ${getTabLabel(mediaTypes, typeId) || typeId} imports will pre-fill these settings.`,
          dedupKey: `import-defaults-saved:${typeId}`,
        });
      },
    });
  }
}
