import {
  CleanerInfo,
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsForMediaType,
  MediaTypeInfo,
} from '../../../../../models/api.models';
import { getTabLabel } from './media-type.util';

/** The option lists an import offered, which give the "Save as default"
 *  offer the names it shows (an embedder's display name, a parameter's
 *  label).  A list left out shows its settings by their registry names. */
export interface ImportOptionLists {
  embedders?: EmbedderInfo[];
  clippers?: ClipperInfo[];
  cleaners?: CleanerInfo[];
  converters?: ConverterInfo[];
}

/** How many changed settings the offer names before it cuts to "…". */
export const MAX_LISTED_CHANGES = 4;

/** One setting of an import default, as the offer names it. */
interface Setting {
  /** Identity across the two configs being compared. */
  key: string;
  label: string;
  value: string;
  /** A parameter's value when left alone.  A parameter only the new config
   *  has (its clipper, cleaner or converter was just picked) is named only
   *  when it is set to something else; the pick itself is named already. */
  untouched?: string;
}

/** A setting's value as the offer shows it: ``on`` / ``off`` for a flag,
 *  ``none`` for an empty value. */
function formatValue(v: unknown): string {
  if (v === null || v === undefined || v === '') return 'none';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (Array.isArray(v)) return `[${v.map(formatValue).join(', ')}]`;
  return String(v);
}

/** Type ids in registry order, then any unregistered ones by name, so two
 *  configs that hold the same kinds in another order read the same. */
function inRegistryOrder(typeIds: string[], mediaTypes: MediaTypeInfo[]): string[] {
  const rank = (t: string) => {
    const i = mediaTypes.findIndex((m) => m.type_id === t);
    return i < 0 ? mediaTypes.length : i;
  };
  return [...new Set(typeIds)].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/** One setting per parameter *declared*, valued from *given* where it is set
 *  and from the declared default where it is not. */
function paramSettings(
  keyPrefix: string,
  declared: { key: string; label?: string; default?: unknown }[],
  given: Record<string, unknown> | undefined,
): Setting[] {
  return declared.map((p) => {
    const v = given?.[p.key];
    const untouched = formatValue(p.default);
    return {
      key: `${keyPrefix}.${p.key}`,
      label: p.label || p.key,
      value: v === undefined || v === null || v === '' ? untouched : formatValue(v),
      untouched,
    };
  });
}

/** Every setting *cfg* makes an import of *typeId* start with, named, in the
 *  order the dialog shows them.  An absent key is the built-in default, as
 *  the importers read it: the first ``is_default`` embedder, the first
 *  clipper, no included media, each cleaner's ``default_enabled``, both
 *  toggles off and the one-dataset form. */
function settingsOf(
  typeId: string,
  cfg: ImportDefaultsForMediaType,
  mediaTypes: MediaTypeInfo[],
  lists: ImportOptionLists,
): Setting[] {
  const out: Setting[] = [];
  const kinds = inRegistryOrder(cfg.multi_dataset || [], mediaTypes);
  out.push({
    key: 'multi_dataset',
    label: 'Multi-Dataset',
    value: kinds.length > 0 ? kinds.map((k) => getTabLabel(mediaTypes, k)).join(' + ') : 'off',
  });

  const embedders = lists.embedders || [];
  const embedderName = cfg.embedder || (embedders.find((e) => e.is_default) || embedders[0])?.name || '';
  const embedder = embedders.find((e) => e.name === embedderName);
  out.push({ key: 'embedder', label: 'Embedder', value: embedder?.display_name || embedderName || 'default' });

  const clippers = lists.clippers || [];
  const clipperName = cfg.clipper || clippers[0]?.name || '';
  const clipper = clippers.find((c) => c.name === clipperName);
  out.push({ key: 'clipper', label: 'Clipper', value: clipper?.display_name || clipperName || 'default' });
  // Only a picked clipper carries parameters; the default one runs as declared.
  if (cfg.clipper && clipper) {
    out.push(...paramSettings(`clipper.${clipper.name}`, clipper.parameters || [], cfg.clipper_params));
  }

  const converters = lists.converters || [];
  const included = (cfg.source_specs || []).filter((s) => !(s.source_type === typeId && s.converter === null));
  const sources = inRegistryOrder(
    included.map((s) => s.source_type),
    mediaTypes,
  );
  out.push({
    key: 'source_specs',
    label: 'Include media',
    value: sources.length > 0 ? sources.map((t) => getTabLabel(mediaTypes, t)).join(' + ') : 'none',
  });
  for (const source of sources) {
    const spec = included.find((s) => s.source_type === source)!;
    const options = converters.filter((c) => c.source_type === source);
    const converter = converters.find((c) => c.name === spec.converter);
    const value = converter?.display_name || spec.converter || 'none';
    out.push({
      key: `source_specs.${source}`,
      label: `${getTabLabel(mediaTypes, source)} converter`,
      value,
      // A newly included kind starts on its first converter, as the picker does.
      untouched: options[0] ? options[0].display_name || options[0].name : value,
    });
    if (converter) {
      const fields = (converter.fields || []).filter((f) => !f.hidden);
      out.push(...paramSettings(`source_specs.${source}.${converter.name}`, fields, spec.params));
    }
  }

  for (const c of lists.cleaners || []) {
    const choice = cfg.cleaners?.[c.name];
    const enabled = choice ? choice.enabled : !!c.default_enabled;
    const label = c.display_name || c.name;
    out.push({ key: `cleaner.${c.name}`, label, value: formatValue(enabled) });
    if (enabled) {
      out.push(...paramSettings(`cleaner.${c.name}`, c.parameters || [], choice?.params));
    }
  }

  out.push({ key: 'build_projection', label: 'Build Browse map now', value: formatValue(!!cfg.build_projection) });
  out.push({ key: 'merge_near_duplicates', label: 'Merge near-duplicates', value: formatValue(!!cfg.merge_near_duplicates) });
  return out;
}

/** The settings saving *next* would change from *saved* (the default an
 *  import of *typeId* starts with now), as ``"Label: value"`` strings in the
 *  order the dialog shows them.  A setting that only *saved* names (a
 *  parameter of a clipper no longer picked, a cleaner turned off) is left
 *  out: the change that dropped it is named instead. */
export function importDefaultChanges(
  typeId: string,
  saved: ImportDefaultsForMediaType,
  next: ImportDefaultsForMediaType,
  mediaTypes: MediaTypeInfo[],
  lists: ImportOptionLists,
): string[] {
  const before = new Map(settingsOf(typeId, saved, mediaTypes, lists).map((s) => [s.key, s.value]));
  return settingsOf(typeId, next, mediaTypes, lists)
    .filter((s) => {
      const was = before.get(s.key);
      return was === undefined ? s.untouched === undefined || s.value !== s.untouched : was !== s.value;
    })
    .map((s) => `${s.label}: ${s.value}`);
}

/** *changes* joined for the offer's detail line, cut to the first
 *  :data:`MAX_LISTED_CHANGES` with a count of the rest. */
export function summarizeChanges(changes: string[]): string {
  if (changes.length <= MAX_LISTED_CHANGES) return changes.join(', ');
  const rest = changes.length - MAX_LISTED_CHANGES;
  return `${changes.slice(0, MAX_LISTED_CHANGES).join(', ')}, … (${rest} more)`;
}
