/* TypeScript interfaces matching the Flask API response shapes. */

// --- Medias ---

/**
 * One media item.
 *
 * Only ``id`` and ``type`` are guaranteed — the dataset listing
 * (``GET /api/medias/ids``) returns just those plus an optional
 * ``embedder``.  The remaining display-worthy fields are populated on
 * demand for items currently in the viewport via the metadata cache
 * (``POST /api/medias/batch``).
 */
export interface MediaItem {
  id: number;
  type: string;
  filename?: string;
  md5?: string;
  custom_metadata?: Record<string, unknown>;
  origin_name?: string;
  description?: string;
  clip_start?: number;
  clip_end?: number;
  clip_index?: number;
  clip_box?: number[];
  /** Name of the embedder that produced this media's vector. */
  embedder?: string;
}

// --- Sorting ---

export interface VotesResponse {
  good: number[];
  bad: number[];
  click_times: Record<string, number>;
  learned_scores: Record<string, number>;
  labelset_good_count?: number;
  labelset_bad_count?: number;
}

// --- Labeling Status ---

export interface StatusIndicator {
  status: string;
  [key: string]: unknown;
}

export interface LabelingStatusResponse {
  good_count?: number;
  bad_count?: number;
  total_count?: number;
  smart?: StatusIndicator;
  stable?: StatusIndicator;
  span?: StatusIndicator;
  [key: string]: unknown;
}

export interface SortProgressResponse {
  [key: string]: unknown;
}

// --- Datasets ---

export interface DatasetStatus {
  loaded: boolean;
  num_medias: number;
  has_votes: boolean;
  media_type?: string;
  display_name?: string;
}

export interface DatasetProgress {
  status?: string;
  message?: string;
  current?: number;
  total?: number;
  step?: number;
  total_steps?: number;
  error?: string;
  [key: string]: unknown;
}

export interface LoadingTask {
  task_id: string;
  name: string;
  status: string;
  message: string;
  current: number;
  total: number;
  step?: number;
  total_steps?: number;
  error?: string;
  created_at: number;
  dataset_id?: string;
  detector_id?: string;
  media_type?: string;
  embedder?: string;
}

export interface LoadingTasksResponse {
  tasks: LoadingTask[];
}

export interface ImporterInfo {
  name: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  ui_mode?: string;
  /** Which view the dataset-importer modal opens for this card.
   *  ``"form"`` (default), ``"demo"``, ``"server_folder"``, ``"local_folder"``. */
  picker_view?: string;
  hidden_from_picker?: boolean;
  /** Picker tab this importer belongs to.  One of ``"services"``,
   *  ``"server"``, ``"local"``, ``"demo"``, or ``""`` (uncategorised). */
  category?: string;
  /** When true, the importer participates in the multi-media flow: the
   *  frontend sends ``source_specs`` (a JSON array of SourceSpec rows)
   *  instead of (or in addition to) a single ``media_type`` field. */
  multi_media?: boolean;
  /** server_folder-specific: map of output media-type id → converters
   *  whose ``target_type`` matches that id.  Drives the "Include rows"
   *  UI without an extra round-trip to ``/api/converters``. */
  available_converters_by_media_type?: Record<string, ConverterInfo[]>;
  [key: string]: unknown;
}

export interface ConverterInfo {
  name: string;
  source_type: string;
  target_type: string;
  display_name?: string;
  description?: string;
  fields?: ImporterField[];
}

/** One row of a multi-media import specification.  See
 *  ``docs/plans/multi-media-import.md``. */
export interface SourceSpec {
  source_type: string;
  /** ``null`` means "include directly" (source_type must equal output). */
  converter: string | null;
  params: Record<string, string | number | null>;
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
  /** When true, ``options`` is computed at runtime by calling
   *  ``POST /api/dataset/import/<importer>/options`` with the current
   *  field values.  The frontend re-fetches whenever any field listed in
   *  ``depends_on`` changes. */
  dynamic_options?: boolean;
  /** Field keys whose values this field's options depend on. */
  depends_on?: string[];
  /** For ``number`` fields: minimum allowed value (empty = no min). */
  min?: string;
  /** For ``number`` fields: maximum allowed value (empty = no max). */
  max?: string;
  /** For ``number`` fields: step increment (empty / ``"any"`` = unconstrained). */
  step?: string;
}

export interface ImporterPickerTab {
  id: string;
  label: string;
  icon?: string;
  order?: number;
}

export interface ImportersResponse {
  importers: ImporterInfo[];
  /** Picker tab declarations.  When present, the frontend renders one tab
   *  per entry; when absent (older backends) the frontend falls back to
   *  inferring tabs from importer ``category`` values. */
  tabs?: ImporterPickerTab[];
}

export interface DemoDataset {
  name: string;
  label: string;
  status: 'ready' | 'needs_embedding' | 'needs_download';
  ready: boolean;
  num_files: number;
  download_size_mb: number;
  description: string;
  media_type: string;
  num_categories: number;
  pkl_embedder?: string;
  pkl_clipper?: string;
}

export interface DemoListResponse {
  datasets: DemoDataset[];
}

export interface MediaTypeInfo {
  type_id: string;
  name: string;
  icon?: string;
  folder_import_name?: string;
  /** Glob patterns for files this media type claims, e.g. ``["*.jpg", "*.png"]``. */
  file_extensions?: string[];
}

export interface MediaTypeDetectionResponse {
  sample_size: number;
  counts_by_type: Record<string, number>;
  extensions: Record<string, number>;
  dominant: string | null;
  /** ``true`` when the backend stopped walking before reaching the file
   *  cap because the directory-count cap or wall-clock budget fired.  The
   *  rest of the response is still meaningful — it's just a less complete
   *  sample than usual. */
  truncated?: boolean;
}

export interface MediaTypesResponse {
  media_types: MediaTypeInfo[];
}

export interface DatasetRegistryEntry {
  id: string;
  name: string;
  media_type: string;
  loaded?: boolean;
  readers?: string[];
  [key: string]: unknown;
}

export interface DatasetStatsResponse {
  num_items: number;
  num_dupes: number;
  file_type_counts: Record<string, number>;
  ingest_started_at: number | null;
  ingest_finished_at: number | null;
  origin: string;
  source: { importer?: string; params?: Record<string, string> } | Record<string, unknown>;
  clipper: string;
  embedder: string;
}

export interface DatasetRegistryResponse {
  datasets: DatasetRegistryEntry[];
}

// --- Detector scoring ---

export interface AutoDetectResponse {
  [key: string]: unknown;
}

// --- Detectors ---

export interface Detector {
  name: string;
  [key: string]: unknown;
}

export interface DetectorsResponse {
  detectors: Detector[];
}

export interface LabelElement {
  id: string;
  label: 'good' | 'bad';
  media_type: string;
  name: string;
  filename: string;
  origin_name: string;
  md5: string;
  cid: number | null;
  time: number;
  score: number;
}

export interface LabelsDetailResponse {
  good: LabelElement[];
  bad: LabelElement[];
  media_type: string;
}

export interface DetectorRegistryEntry {
  id: string;
  name: string;
  media_type: string;
  num_training?: number;
  text_query?: string;
  media_example?: string;
  loaded?: boolean;
  detector_loaded?: boolean;
  autorun?: boolean;
  last_trained_at?: number | null;
  [key: string]: unknown;
}

export interface DetectorsRegistryResponse {
  detectors: DetectorRegistryEntry[];
}

export interface CombineDetectorsResult {
  success: boolean;
  name: string;
  media_type: string;
  num_labels: number;
  combined_from: string[];
  source_label_counts: number[];
  examples: { type: string; value: string }[];
}

// --- Processor Importers ---

export interface ProcessorImporterInfo {
  name: string;
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
  media_type: string;
  parameters?: ClipperParameter[];
  creation_questions?: ClipperParameter[];
  [key: string]: unknown;
}

export interface ClippersResponse {
  clippers: ClipperInfo[];
}

// --- Converters ---

export interface ConverterInfo {
  [key: string]: unknown;
}

export interface ConvertersResponse {
  converters: ConverterInfo[];
}

// --- Error ---

export interface ApiError {
  error: string;
}

// --- OK response ---

export interface OkResponse {
  ok: boolean;
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

export interface AutoDetectResultsData {
  media_type?: string;
  detectors_run?: string | number;
  results: Record<string, AutoDetectDetectorResult>;
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
   * — the UI hides text-search affordances for datasets using them.
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
   * User-facing licence warning to show before the user picks this embedder.
   * ``null`` for embedders with no special licensing constraints; a short
   * human-readable string for embedders with restrictive licences (e.g.
   * facebookresearch/EUPE under the FAIR Noncommercial Research Licence).
   * Advisory only — the picker shows a warning chip, it does not gate
   * selection behind an acceptance click.
   */
  license_notice?: string | null;
}

export interface EmbeddersResponse {
  embedders: EmbedderInfo[];
}

// --- Export Result ---

export interface ExportResult {
  success: boolean;
  message?: string;
  error?: string;
}
