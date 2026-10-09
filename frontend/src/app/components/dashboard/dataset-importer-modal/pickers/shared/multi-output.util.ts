import {
  CleanerSelection,
  ConverterInfo,
  DetectMediaTypeResponse,
  ImporterInfo,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../../models/api.models';
import { composeEmbedders } from './media-type.util';

/** Pure helpers behind the Add Dataset dialog's **Multi-Dataset** mode (#4703).
 *
 *  In multi mode the single "Dataset media type" dropdown gives way to one
 *  row per *ingestion category* - Image, Audio, Video, Text, Document, Face -
 *  and every ticked row becomes its own dataset of one import.  Each row is an
 *  {@link OutputDraft}: what the user has decided for that dataset (ticked or
 *  not, its conversion target, its source rows, embedders, clipper, cleanup).
 *  The picker components own the draft list; `<vt-multi-output-config>` edits
 *  it; {@link draftToOutputRequest} turns a ticked draft into one entry of the
 *  request's `outputs` list.
 *
 *  Three shapes of category, following the half-media-type model
 *  (`docs/plans/half-media-types.md`):
 *
 *  - an **embeddable, importable** type (image, audio, video, text) is a
 *    dataset of its own files, plus whatever other source types the user
 *    pulls in through converters - the ordinary Include-media matrix;
 *  - a **convert-out** type (document) has no embedder, so its dataset is of
 *    the converted media (`document2image` pages by default, or extracted
 *    text), and its only source is the category itself;
 *  - a **convert-in** type (face) is never scanned from files; its dataset is
 *    whatever the converters into it produce (faces cropped out of the images),
 *    so its include rows are the converter rows and there is no native row. */
export interface OutputDraft {
  /** The ingestion category this row stands for, as a media `type_id`. */
  category: string;
  /** The `type_id` of the dataset this row produces: the category itself,
   *  or its conversion target for a convert-out category. */
  mediaType: string;
  checked: boolean;
  /** Conversion target for a convert-out category (`''` otherwise). */
  convertTarget: string;
  sourceSpecs: SourceSpec[];
  embedder: string;
  patchEmbedder: string;
  structuralEmbedder: string;
  clipper: string;
  clipperParams: Record<string, number | string>;
  cleaners: CleanerSelection[];
}

/** One `outputs` entry as `POST /api/dataset/import/<name>` (and the two
 *  local-upload routes) accept it. */
export interface OutputRequest {
  media_type: string;
  category: string;
  source_specs: SourceSpec[];
  embedder?: string;
  embedders?: string[];
  clipper?: string;
  clipper_params?: Record<string, number | string>;
  cleaners?: CleanerSelection[];
}

/** Whether the Multi-Dataset toggle is offered for *importer*: it must say so
 *  (`supports_multi_output`, true for every importer with a `media_type`
 *  field that has not opted out) - the demo importer, a pickle upload and the
 *  like have one fixed dataset and never get the toggle. */
export function isMultiOutputImporter(importer: ImporterInfo | null | undefined): boolean {
  return !!importer?.supports_multi_output;
}

function convertersInto(convertersByType: Record<string, ConverterInfo[]>, targetTypeId: string): ConverterInfo[] {
  return convertersByType[targetTypeId] || [];
}

function defaultParams(converter: ConverterInfo): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of converter.fields || []) out[f.key] = String(f.default ?? '');
  return out;
}

/** The categories the multi-dataset form lists, in registry order: every
 *  importable type, plus every convert-in type some converter can produce. */
export function multiOutputCategories(
  mediaTypes: MediaTypeInfo[],
  convertersByType: Record<string, ConverterInfo[]>,
): MediaTypeInfo[] {
  return mediaTypes.filter((mt) => {
    if (mt.importable !== false) return true;
    return mt.embeddable !== false && convertersInto(convertersByType, mt.type_id).length > 0;
  });
}

/** The dataset type a category produces when converted to *target* (or to its
 *  default target), with the matching source rows.  A plain embeddable
 *  category keeps its own type and a bare "include directly" row. */
export function sourceRowsFor(
  category: MediaTypeInfo,
  convertersByType: Record<string, ConverterInfo[]>,
  target = '',
): { mediaType: string; convertTarget: string; sourceSpecs: SourceSpec[] } {
  const typeId = category.type_id;
  const convertOut = category.embeddable === false && (category.converts_to?.length ?? 0) > 0;
  if (convertOut) {
    const to = target && category.converts_to!.includes(target) ? target : category.converts_to![0];
    const converter = convertersInto(convertersByType, to).find((c) => c.source_type === typeId);
    return {
      mediaType: to,
      convertTarget: to,
      sourceSpecs: [{ source_type: typeId, converter: converter?.name ?? `${typeId}2${to}`, params: converter ? defaultParams(converter) : {} }],
    };
  }
  if (category.importable === false) {
    // Convert-in: nothing of this type exists on disk; every converter into it
    // is a source row, and there is no native row.
    const rows = convertersInto(convertersByType, typeId).map((c) => ({
      source_type: c.source_type,
      converter: c.name,
      params: defaultParams(c),
    }));
    return { mediaType: typeId, convertTarget: '', sourceSpecs: rows };
  }
  return { mediaType: typeId, convertTarget: '', sourceSpecs: [{ source_type: typeId, converter: null, params: {} }] };
}

/** A fresh, unticked draft per category. */
export function buildOutputDrafts(
  mediaTypes: MediaTypeInfo[],
  convertersByType: Record<string, ConverterInfo[]>,
): OutputDraft[] {
  return multiOutputCategories(mediaTypes, convertersByType).map((mt) => ({
    category: mt.type_id,
    checked: false,
    embedder: '',
    patchEmbedder: '',
    structuralEmbedder: '',
    clipper: '',
    clipperParams: {},
    cleaners: [],
    ...sourceRowsFor(mt, convertersByType),
  }));
}

/** Tick every category the folder scan found files of.  Additive: a row the
 *  user ticked by hand stays ticked, and a convert-in category (which no scan
 *  can find, having no file extensions) is left as it is. */
export function tickDetectedCategories(drafts: OutputDraft[], detection: DetectMediaTypeResponse | null): OutputDraft[] {
  if (!detection) return drafts;
  const counts = detection.counts_by_type || {};
  return drafts.map((d) => ((counts[d.category] ?? 0) > 0 ? { ...d, checked: true } : d));
}

/** Files of a category the last scan found, for the row's count hint. */
export function detectedCount(detection: DetectMediaTypeResponse | null, category: string): number {
  return detection?.counts_by_type?.[category] ?? 0;
}

/** Tick exactly *category* (the type the single-dataset form had picked), so
 *  the multi form opens with the dataset the user was already about to make. */
export function tickCategory(drafts: OutputDraft[], category: string): OutputDraft[] {
  return drafts.map((d) => (d.category === category ? { ...d, checked: true } : d));
}

/** The request entry for one ticked draft. */
export function draftToOutputRequest(draft: OutputDraft): OutputRequest {
  const out: OutputRequest = {
    media_type: draft.mediaType,
    category: draft.category,
    source_specs: draft.sourceSpecs,
  };
  if (draft.embedder) out.embedder = draft.embedder;
  const embedders = composeEmbedders(draft.embedder, draft.patchEmbedder, draft.structuralEmbedder);
  if (embedders) out.embedders = embedders;
  if (draft.clipper) {
    out.clipper = draft.clipper;
    if (Object.keys(draft.clipperParams).length > 0) out.clipper_params = { ...draft.clipperParams };
  }
  if (draft.cleaners.length > 0) out.cleaners = draft.cleaners;
  return out;
}

/** The `outputs` list for a submit: every ticked draft, in category order. */
export function outputsFromDrafts(drafts: OutputDraft[]): OutputRequest[] {
  return drafts.filter((d) => d.checked).map(draftToOutputRequest);
}
