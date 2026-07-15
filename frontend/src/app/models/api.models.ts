/* TypeScript interfaces matching the Flask API response shapes. */

import type { MediaIdsListResponse } from '../generated/api-client/models/media-ids-list-response';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';

// --- Medias ---

/**
 * Strip a type's catch-all index signature (``[key: string]: any``) so
 * intersecting it with another type doesn't force every property access
 * to go through the bracket form (TS4111 under
 * ``noPropertyAccessFromIndexSignature``).
 */
type RemoveIndex<T> = {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

/**
 * The renderable media shape: a stub from ``GET /api/medias/ids``
 * (``id``, ``type``, optional ``embedder``) optionally augmented with the
 * full per-item metadata returned by ``POST /api/medias/batch``
 * (``filename``, ``md5``, ``custom_metadata``, clip extents, …).
 *
 * Derived directly from the two generated DTOs so backend schema changes
 * propagate without a hand-maintained mirror.  Components consume this
 * type wherever the data flow can deliver either a stub (initial listing,
 * cache miss) or a hydrated batch entry.
 */
export type Media = MediaIdsListResponse &
  Partial<Omit<RemoveIndex<MediaBatchResponse>, keyof MediaIdsListResponse>>;

// --- Progress ---

/**
 * The unified shape every long-running operation emits over the SSE stream
 * (`/api/events`; see `progress-events.service.ts`).
 *
 * Backend: every singleton `ProgressTracker` (dataset, sort, eval, find) and
 * every per-task tracker created by `LoadingTasksTracker` carries the same
 * base fields plus the shared optional extras `step`/`total_steps`/`error`.
 * See `vtsearch/concurrency/progress.py:_PROGRESS_COMMON_EXTRAS`.
 *
 * Frontend: a single component/helper (`utils/format-progress.ts`) can render
 * any payload regardless of which operation produced it.
 */
export interface ProgressEvent {
  status?: string;
  message?: string;
  current?: number;
  total?: number;
  /** Sub-step counter for multi-phase operations (e.g. load→embed→stage). */
  step?: number | null;
  total_steps?: number | null;
  /** Error message if the operation failed. */
  error?: string | null;
  /**
   * Smoothed remaining-seconds estimate filled in by
   * ``ProgressTracker._compute_eta`` once the bar has been running long
   * enough (>5s) with a known total. ``null`` means "no estimate yet".
   */
  eta_seconds?: number | null;
  /**
   * Whole-job completion fraction (0..1) for multi-step operations, computed
   * by ``ProgressTracker._compute_overall``. When present, the progress bar
   * fills once across the entire job (download → load → embed → finalize)
   * instead of resetting at each phase. ``null`` for single-phase operations,
   * where consumers fall back to ``current``/``total``.
   */
  overall?: number | null;
  /** Dataset-only: payload returned by combine-datasets staging. */
  staging_result?: unknown;
  [key: string]: unknown;
}

// --- Datasets ---

/**
 * A per-task progress event from the `loading-tasks` /
 * `detector-loading-tasks` SSE channels. Wraps `ProgressEvent` with the
 * task identity/metadata fields the dashboard needs to render one row per
 * concurrent load. The base fields are non-optional here because every task
 * tracker writes them through `ProgressTracker.update`.
 */
export interface LoadingTask extends ProgressEvent {
  status: string;
  message: string;
  current: number;
  total: number;
  task_id: string;
  name: string;
  created_at: number;
  dataset_id?: string;
  detector_id?: string;
  media_type?: string;
  embedder?: string;
}

export interface ImporterInfo {
  name: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  ui_mode?: string;
  /** Which view the dataset-importer modal opens for this card.
   *  ``"form"`` (default), ``"demo"``, ``"server_folder"``, ``"local_folder"``,
   *  or ``"local_files"``. */
  picker_view?: string;
  hidden_from_picker?: boolean;
  /** Picker tab this importer belongs to.  One of ``"services"``,
   *  ``"server"``, ``"local"``, ``"demo"``, or ``""`` (uncategorised). */
  category?: string;
  /** Map of output media-type id → converters whose ``target_type``
   *  matches that id.  Drives the "Include rows" UI without an extra
   *  round-trip to ``/api/converters``. */
  available_converters_by_media_type?: Record<string, ConverterInfo[]>;
  [key: string]: unknown;
}

export interface ConverterInfo {
  name: string;
  source_type: string;
  target_type: string;
  display_name?: string;
  description?: string;
  /** One-line preview with ``{key}`` placeholders for each field.  The
   *  importer modal renders it next to the source-spec row, substituting
   *  the current field values, so the user sees a live summary of what
   *  the converter will do.  Falls back to ``description`` when empty. */
  summary_template?: string;
  fields?: ImporterField[];
}

/** One row of a multi-media import specification.  See
 *  ``docs/EXTENDING-media.md``. */
export interface SourceSpec {
  source_type: string;
  /** ``null`` means "include directly" (source_type must equal output). */
  converter: string | null;
  params: Record<string, string | number | null>;
}

/** Saved per-mediaType defaults for the Add Dataset advanced panel:
 *  the embedder, clipper (+ params), and converter rows the user wants
 *  applied automatically every time they import a dataset whose output
 *  is the matching mediaType. Edited from Settings > Import Defaults;
 *  silently auto-filled into the importer form on importer selection. */
export interface ImportDefaultsForMediaType {
  embedder?: string;
  clipper?: string;
  clipper_params?: Record<string, string | number>;
  /** Source-spec rows to seed into the importer's "Include media" picker.
   *  May omit the native "include directly" row; the importer adds it
   *  implicitly so the form is never empty. */
  source_specs?: SourceSpec[];
}
export type ImportDefaultsByMediaType = Record<string, ImportDefaultsForMediaType>;

/** A single dropdown option for a dynamic-options field.  ``value`` is
 *  what the form submits; ``label`` is the friendly text shown in the
 *  dropdown.  They coincide for plain-string options and differ for
 *  ``(value, label)`` tuple options (submit an opaque id, show a name). */
export interface FieldOption {
  value: string;
  label: string;
}

export interface ImporterField {
  key: string;
  field_type: string;
  label?: string;
  description?: string;
  accept?: string;
  options?: string[];
  default?: string;
  required?: boolean;
  placeholder?: string;
  /** Inline format-hint text rendered as a visible chip below the input.
   *  Distinct from ``description`` (which feeds the placeholder): the hint
   *  stays visible after the user starts typing, so it's the right place
   *  for accepted file extensions, expected schemas, or a short sample of
   *  the file layout. */
  hint?: string;
  /** When true, ``options`` is computed at runtime by calling
   *  ``POST /api/dataset/import/<importer>/options`` with the current
   *  field values.  The frontend re-fetches whenever any field listed in
   *  ``depends_on`` changes. */
  dynamic_options?: boolean;
  /** Field keys whose values this field's options depend on. */
  depends_on?: string[];
  /** For ``select`` fields: when true, render as a combobox the user can
   *  type an arbitrary value into (even one the option list omits).  When
   *  the options refresh, a typed value absent from the new list is kept;
   *  a strict select clears it. */
  allow_free_text?: boolean;
  /** For ``number`` fields: minimum allowed value (empty = no min). */
  min?: string;
  /** For ``number`` fields: maximum allowed value (empty = no max). */
  max?: string;
  /** For ``number`` fields: step increment (empty / ``"any"`` = unconstrained). */
  step?: string;
  /** Field keys this field is mutually exclusive with.  Entering a
   *  non-empty value here blanks each listed field (and they list this
   *  one back), so only one of the set is ever active at a time. */
  clears?: string[];
}

export interface ImporterPickerTab {
  id: string;
  label: string;
  icon?: string;
  order?: number;
}

export interface MediaTypeInfo {
  type_id: string;
  name: string;
  icon?: string;
  folder_import_name?: string;
  /** Glob patterns for files this media type claims, e.g. ``["*.jpg", "*.png"]``. */
  file_extensions?: string[];
  /** Whether items of this type have a browsable thumbnail (image/video/document,
   *  and audio via its waveform PNG). Drives the VTSBrowse square-vs-hex bin shape
   *  and thumbnail painting; the single source of truth the frontend reads via
   *  ``MediaTypeCapabilityService.usesThumbnails``. */
  has_thumbnail?: boolean;
  /** Whether this type is a first-class *ingestion* category the user picks when
   *  importing (folder scan, file upload). ``false`` for a *convert-in* half type
   *  like ``face`` that only ever arises from converting another type. */
  importable?: boolean;
  /** Whether this type can be embedded (and therefore sorted / browsed / text-
   *  queried) on its own. ``false`` for a *convert-out* half type like
   *  ``document`` that must be converted first. */
  embeddable?: boolean;
  /** Embeddable target type_ids a non-embeddable type can convert into (first =
   *  default). ``["image", "text"]`` for ``document``; empty for a directly-
   *  embeddable type. */
  converts_to?: string[];
}

export interface MediaTypeDetectionResponse {
  sample_size: number;
  counts_by_type: Record<string, number>;
  extensions: Record<string, number>;
  dominant: string | null;
  /** ``true`` when the backend stopped walking before reaching the file
   *  cap because the directory-count cap or wall-clock budget fired.  The
   *  rest of the response is still meaningful; it's just a less complete
   *  sample than usual. */
  truncated?: boolean;
}

export interface DatasetRegistryEntry {
  id: string;
  name: string;
  media_type: string;
  loaded?: boolean;
  readers?: string[];
  /** Name of the embedder this dataset's media were vectorised with. */
  embedder?: string;
  /** The embedder *types* this dataset supplies ("semantic" / "patch_semantic"
   *  / "structural"); a v3 trio dataset can supply several. Drives the
   *  detector/dataset compatibility gate. */
  embedder_types?: string[];
  /** Unix timestamp (seconds) at which this dataset ages off and is
   *  automatically removed; `null`/absent means it never expires. */
  expires_at?: number | null;
  [key: string]: unknown;
}

// --- Clippers ---

export interface ClipperParameter {
  key: string;
  label: string;
  description?: string;
  type: 'number' | 'string';
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
}

export interface ClipperInfo {
  name: string;
  display_name?: string;
  description?: string;
  /** One-line preview with ``{key}`` placeholders for each parameter.  The
   *  native row of the importer source-specs picker substitutes the current
   *  parameter values, so the user sees a live summary of what the clipper
   *  will do.  Falls back to ``description`` when empty. */
  summary_template?: string;
  media_type: string;
  parameters?: ClipperParameter[];
  creation_questions?: ClipperParameter[];
  [key: string]: unknown;
}

// --- Eval / Progress Charts ---

export interface ErrorCostDataPoint {
  num_labels: number;
  error_cost: number;
}

export interface StabilityDataPoint {
  num_labels: number;
  num_flips: number;
}

export interface DiversityDataPoint {
  num_labels: number;
  diversity_level: number;
  depth: number;
}

export interface VotingIterationsResponse {
  progress: number;
  total: number;
  done: boolean;
  [key: string]: unknown;
}

// --- Auto-detect Results ---

export interface AutoDetectHit {
  md5: string;
  filename?: string;
  origin_name?: string;
  origin?: {
    importer?: string;
    params?: Record<string, string>;
  };
  label?: string;
  [key: string]: unknown;
}

export interface AutoDetectDetectorResult {
  hits: AutoDetectHit[];
  negative_hits?: AutoDetectHit[];
  total_hits?: number;
  [key: string]: unknown;
}

/** Outcome of auto-exporting Auto-Find results, present on the auto-detect
 *  response only when a results exporter is configured in settings. */
export interface AutoFindExportStatus {
  exporter: string;
  success: boolean;
  message?: string;
  error?: string;
  [key: string]: unknown;
}

export interface AutoDetectResultsData {
  media_type?: string;
  detectors_run?: string | number;
  results: Record<string, AutoDetectDetectorResult>;
  /** Auto-Find list entries whose detector file no longer exists on disk. */
  missing_detectors?: string[];
  /** Auto-export outcome (only when a results exporter ran). */
  auto_export?: AutoFindExportStatus;
  // Find mode fields
  detectors?: string[];
  datasets?: string[];
  multiple_datasets?: boolean;
  multiple_detectors?: boolean;
  [key: string]: unknown;
}

// --- Embedders ---

export interface EmbedderInfo {
  name: string;
  /**
   * Human-readable label shown in pickers, e.g. ``"SigLIP (general images)"``.
   * Falls back to ``name`` for legacy embedders that don't supply a friendlier
   * label; the raw ``name`` is also rendered as a secondary line so power
   * users can still see the registry key.
   */
  display_name?: string;
  /**
   * Concrete pretrained-model identifier the embedder loads — usually a
   * HuggingFace repo id (e.g. ``"google/siglip-base-patch16-224"``), or a
   * direct weights URL. ``null`` for embedders with no single downloadable
   * model id (e.g. the classical SIFT/VLAD structural embedder). Surfaced in
   * the portable-detector export bundle so a recipient knows exactly which
   * model to run new media through.
   */
  model_id?: string | null;
  media_type_id: string;
  /**
   * Whether this embedder is the recommended default for its media type
   * (exactly one per media type). The dropdown surfaces this entry under a
   * "Recommended" optgroup and tucks the rest under "Advanced".
   */
  is_default?: boolean;
  /**
   * Whether this embedder can embed text queries into the same vector space as
   * its media. ``false`` for vision-only encoders (DINOv3, Perception Encoder)
   * so the UI hides text-search affordances for datasets using them.
   */
  supports_text?: boolean;
  /**
   * Whether this embedder produces patch-level vectors and a hierarchical
   * region tree per image. ``true`` for patch-based encoders (DINOv2,
   * DINOv3, EUPE) once their patch pipeline lands; the gallery card draws
   * a faint outline over the matched region for datasets using them.
   */
  supports_patch_regions?: boolean;
  /**
   * Whether this embedder produces local features (keypoints + descriptors)
   * for instance matching. ``true`` for structural embedders (SIFT/VLAD and
   * learned-local-feature variants later); the loader then stores per-image
   * local features and the geometric re-rank + match-stat verification paths
   * activate. ``false`` (the default) for every semantic embedder.
   */
  supports_geometric_verification?: boolean;
  /**
   * User-facing licence warning to show before the user picks this embedder.
   * ``null`` for embedders with no special licensing constraints; a short
   * human-readable string for embedders with restrictive licences (e.g.
   * facebookresearch/EUPE under the FAIR Noncommercial Research Licence).
   * Advisory only; the picker shows a warning chip, it does not gate
   * selection behind an acceptance click.
   */
  license_notice?: string | null;
}

