import { Injectable, inject } from '@angular/core';

import { SettingsStateService } from '../../../../../services/settings-state.service';
import { ToastService } from '../../../../../services/toast.service';
import {
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsForMediaType,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../../models/api.models';
import { defaultSpecListFor, getTabLabel, toTypeId } from './media-type.util';

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

/** Per-user "sticky" import config: solo mediaType/embedder locks, saved
 *  embedder/clipper/converter defaults per media type, and the
 *  post-import "save these as default?" toast offer.
 *
 *  Shared by every picker view (generic form / server-folder /
 *  local-folder / demo) of the Add Dataset modal - each view used to
 *  carry its own copy of this logic under a ``form*`` / ``sf*`` / ``lf*``
 *  prefix. */
@Injectable({ providedIn: 'root' })
export class ImportDefaultsService {
  private settingsState = inject(SettingsStateService);
  private toastService = inject(ToastService);

  /** Type_id (e.g. ``"image"``) of the solo-mediaType streamlining, or
   *  ``null`` when not active. */
  get effectiveSoloMediaType(): string | null {
    const v = this.settingsState.settingsSignal()?.effective_solo_media_type;
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

  /** Snapshot of the per-view import-config (embedder/clipper/specs) the
   *  user submitted, normalised to the same shape as the persisted
   *  defaults. */
  snapshotImportConfig(
    typeId: string,
    embedder: string,
    clipper: string,
    clipperParams: Record<string, number | string>,
    sourceSpecs: SourceSpec[],
    availableEmbedders: EmbedderInfo[],
    availableClippers: ClipperInfo[],
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
    return out;
  }

  private hasMeaningfulOverrides(cfg: ImportDefaultsForMediaType): boolean {
    return !!(cfg.embedder || cfg.clipper || (cfg.source_specs && cfg.source_specs.length > 0));
  }

  private importDefaultsEqual(a: ImportDefaultsForMediaType, b: ImportDefaultsForMediaType): boolean {
    if ((a.embedder || '') !== (b.embedder || '')) return false;
    if ((a.clipper || '') !== (b.clipper || '')) return false;
    if (!shallowParamsEqual(a.clipper_params, b.clipper_params)) return false;
    if (!shallowSpecsEqual(a.source_specs, b.source_specs)) return false;
    return true;
  }

  /** After a successful import, offer to save the user's advanced
   *  settings as the default for this output mediaType. Skipped silently
   *  when the user accepted the importer's natural defaults or already
   *  has the same config saved. */
  maybeOfferSaveImportDefaults(typeId: string, cfg: ImportDefaultsForMediaType, mediaTypes: MediaTypeInfo[]): void {
    if (!typeId) return;
    if (!this.hasMeaningfulOverrides(cfg)) return;
    const saved = this.importDefaultsForFolderOrTypeId(typeId, mediaTypes) || {};
    if (this.importDefaultsEqual(saved, cfg)) return;
    const typeLabel = getTabLabel(mediaTypes, typeId) || typeId;
    this.toastService.success({
      message: `Save these as default for ${typeLabel} imports?`,
      detail: 'Your embedder, clipper, and converter picks will be auto-filled next time.',
      action: {
        label: 'Save as default',
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
    next[typeId] = cfg;
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
