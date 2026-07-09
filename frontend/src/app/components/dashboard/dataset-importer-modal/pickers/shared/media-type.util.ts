import { ConverterInfo, ImporterInfo, MediaTypeDetectionResponse, MediaTypeInfo, SourceSpec } from '../../../../../models/api.models';

/** Pure helpers for translating between the media-type registry's two
 *  addressing schemes - ``type_id`` (canonical, e.g. ``"image"``) and
 *  ``folder_import_name`` (what the importer form fields use, e.g.
 *  ``"images"``) - and for building the label/icon lookup maps the
 *  picker components pass into ``<vt-import-config>`` /
 *  ``<vt-import-advanced>``.
 *
 *  Extracted from ``DatasetImporterModalComponent`` so every picker
 *  (generic form / server-folder / local-folder / demo) shares one
 *  implementation instead of four copies. */

/** Convert a type_id (e.g. "image") to the corresponding folder_import_name (e.g. "images"). */
export function toFolderName(mediaTypes: MediaTypeInfo[], typeId: string): string {
  if (!typeId) return '';
  const mt = mediaTypes.find((m) => m.type_id === typeId);
  return mt?.folder_import_name || typeId;
}

/** Map a folder_import_name (e.g. "images") to a type_id (e.g. "image"). */
export function toTypeId(mediaTypes: MediaTypeInfo[], folderName: string): string {
  if (!folderName) return '';
  const mt = mediaTypes.find((m) => m.folder_import_name === folderName);
  return mt?.type_id || folderName;
}

export function getTabLabel(mediaTypes: MediaTypeInfo[], mediaType: string): string {
  const mt = mediaTypes.find((m) => m.type_id === mediaType);
  if (mt) return mt.name.trim();
  return mediaType;
}

/** Map of ``type_id`` → human label. */
export function mediaTypeLabels(mediaTypes: MediaTypeInfo[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const mt of mediaTypes) out[mt.type_id] = mt.name.trim();
  return out;
}

/** Map of ``folder_import_name`` → human label. */
export function mediaTypeOptionLabels(mediaTypes: MediaTypeInfo[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const mt of mediaTypes) {
    if (mt.folder_import_name) out[mt.folder_import_name] = mt.name.trim();
  }
  return out;
}

/** Map of ``folder_import_name`` → icon string. */
export function mediaTypeOptionIcons(mediaTypes: MediaTypeInfo[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const mt of mediaTypes) {
    if (mt.folder_import_name && mt.icon) out[mt.folder_import_name] = mt.icon;
  }
  return out;
}

/** Map of ``type_id`` → icon string (used by the demo media-type dropdown,
 *  which is keyed by type_id rather than folder_import_name). */
export function mediaTypeIconsById(mediaTypes: MediaTypeInfo[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const mt of mediaTypes) {
    if (mt.icon) out[mt.type_id] = mt.icon;
  }
  return out;
}

/** Lowercase ``ext → type_id`` map derived from registered media types. */
function extensionToTypeId(mediaTypes: MediaTypeInfo[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const mt of mediaTypes) {
    for (const pattern of mt.file_extensions || []) {
      const dot = pattern.lastIndexOf('.');
      if (dot < 0) continue;
      map.set(pattern.slice(dot).toLowerCase(), mt.type_id);
    }
  }
  return map;
}

/** Count media types in a browser-side ``File[]`` and shape the result
 *  like :type:`MediaTypeDetectionResponse` so the rest of the modal can
 *  treat local- and server-side detections identically.
 *
 *  When ``recursive`` is ``false`` files whose ``webkitRelativePath``
 *  lies in a sub-directory of the picked folder are skipped, matching
 *  the importer's "Include subfolders" toggle. */
export function detectFromFiles(
  mediaTypes: MediaTypeInfo[],
  files: File[],
  recursive: boolean,
  limit = 50,
): MediaTypeDetectionResponse {
  const extMap = extensionToTypeId(mediaTypes);
  const countsByType: Record<string, number> = {};
  const extensions: Record<string, number> = {};
  let examined = 0;
  for (const file of files) {
    if (examined >= limit) break;
    const rel = ((file as any).webkitRelativePath as string | undefined) || '';
    if (!recursive && rel && rel.split('/').length > 2) continue;
    const name = rel || file.name || '';
    const slash = name.lastIndexOf('/');
    const base = slash >= 0 ? name.slice(slash + 1) : name;
    if (base.startsWith('.')) continue;
    const dot = base.lastIndexOf('.');
    const ext = dot > 0 ? base.slice(dot).toLowerCase() : '';
    extensions[ext] = (extensions[ext] || 0) + 1;
    const typeId = (ext && extMap.get(ext)) || 'unknown';
    countsByType[typeId] = (countsByType[typeId] || 0) + 1;
    examined += 1;
  }
  let dominant: string | null = null;
  let bestCount = 0;
  for (const [typeId, count] of Object.entries(countsByType)) {
    if (typeId === 'unknown') continue;
    if (count > bestCount) {
      bestCount = count;
      dominant = typeId;
    }
  }
  return { sample_size: examined, counts_by_type: countsByType, extensions, dominant };
}

/** Human-readable description of a detection result, suitable for a hint
 *  chip next to the media-type dropdown. */
export function detectionHint(
  mediaTypes: MediaTypeInfo[],
  detection: MediaTypeDetectionResponse | null,
): string {
  if (!detection || detection.sample_size === 0) return '';
  const entries = Object.entries(detection.counts_by_type)
    .filter(([typeId]) => typeId !== 'unknown')
    .sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return `No recognised media files in ${detection.sample_size} sampled.`;
  }
  const total = detection.sample_size;
  const fileWord = total === 1 ? 'file' : 'files';
  if (entries.length === 1) {
    const [typeId, count] = entries[0];
    return `Detected: ${getTabLabel(mediaTypes, typeId)} (${count} of ${total} ${fileWord})`;
  }
  const head = entries
    .map(([typeId, count]) => `${getTabLabel(mediaTypes, typeId)} (${count})`)
    .join(' + ');
  return `Detected: ${head} of ${total} ${fileWord}`;
}

/** Apply a detection result to a (mediaType, sourceSpecs) pair.
 *
 *  Sets ``mediaType`` to the dominant type's ``folder_import_name`` when
 *  it's a valid option, then rebuilds the source-spec list: one direct
 *  row for the dominant type plus one converter row per non-dominant
 *  recognised type that has at least one matching converter to the
 *  dominant type.  Returns the new ``(mediaType, sourceSpecs)`` pair -
 *  the caller decides which view's state to update. */
export function autofillFromDetection(
  mediaTypes: MediaTypeInfo[],
  detection: MediaTypeDetectionResponse,
  availableOptions: string[],
  convertersForType: (outputTypeId: string) => ConverterInfo[],
): { mediaType: string | null; sourceSpecs: SourceSpec[] | null } {
  const dominant = detection.dominant;
  if (!dominant) return { mediaType: null, sourceSpecs: null };
  const folderName = mediaTypes.find((m) => m.type_id === dominant)?.folder_import_name || dominant;
  if (!availableOptions.includes(folderName)) {
    return { mediaType: null, sourceSpecs: null };
  }
  const converters = convertersForType(dominant);
  const sourceSpecs: SourceSpec[] = [{ source_type: dominant, converter: null, params: {} }];
  const seenSourceTypes = new Set<string>([dominant]);
  const orderedNonDominant = Object.entries(detection.counts_by_type)
    .filter(([typeId, count]) => typeId !== 'unknown' && typeId !== dominant && count > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([typeId]) => typeId);
  for (const sourceType of orderedNonDominant) {
    if (seenSourceTypes.has(sourceType)) continue;
    const converter = converters.find((c) => c.source_type === sourceType);
    if (!converter) continue;
    const params: Record<string, string> = {};
    for (const f of converter.fields || []) {
      params[f.key] = String(f.default ?? '');
    }
    sourceSpecs.push({ source_type: sourceType, converter: converter.name, params });
    seenSourceTypes.add(sourceType);
  }
  return { mediaType: folderName, sourceSpecs };
}

/** Converters whose ``target_type`` matches *outputTypeId*.  The map
 *  comes from the importer's ``to_dict()`` so each importer can declare
 *  its own filtered list. */
export function availableConvertersFor(
  importers: ImporterInfo[],
  importerName: string,
  outputTypeId: string,
): ConverterInfo[] {
  const importer = importers.find((i) => i.name === importerName);
  const byType = (importer?.available_converters_by_media_type as Record<string, ConverterInfo[]> | undefined) || {};
  return byType[outputTypeId] || [];
}

/** Build the "default" spec list for a freshly-opened picker: one
 *  "include directly" row whose source matches the output type. */
export function defaultSpecListFor(outputTypeId: string): SourceSpec[] {
  return outputTypeId ? [{ source_type: outputTypeId, converter: null, params: {} }] : [];
}

/** Compose the v3 embedder trio request value from a flow's three role
 *  picks.  Returns the deduped, non-empty union with the *primary*
 *  first, or ``null`` when only the primary is bound - then the single
 *  ``embedder`` field carries it (the unchanged pre-trio single-embedder
 *  path). */
/** Read the ``recursive`` field's declared default ("true"/"false") from
 *  the importer metadata; defaults to ``true`` when the field is absent.
 *  Shared by the server-folder and local-folder/files pickers, whose
 *  "Include subfolders" checkbox starts from the importer's declared
 *  default. */
export function readRecursiveDefault(importer: ImporterInfo | null): boolean {
  const field = importer?.fields?.find((f) => f.key === 'recursive');
  if (!field) return true;
  return String(field.default ?? 'true').toLowerCase() !== 'false';
}

export function composeEmbedders(primary: string, patch: string, structural: string): string[] | null {
  const list: string[] = [];
  for (const candidate of [primary, patch, structural]) {
    const name = (candidate || '').trim();
    if (name && !list.includes(name)) list.push(name);
  }
  return list.length > 1 ? list : null;
}
