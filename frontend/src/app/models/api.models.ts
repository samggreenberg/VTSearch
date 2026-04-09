/* TypeScript interfaces matching the Flask API response shapes. */

// --- Medias ---

export interface MediaItem {
  id: number;
  type: string;
  filename: string;
  md5: string;
  custom_metadata: Record<string, unknown>;
  origin_name?: string;
  description?: string;
  clip_start?: number;
  clip_end?: number;
  clip_index?: number;
  clip_box?: number[];
}

export interface TextResponse {
  content: string;
  word_count?: number;
  character_count?: number;
}

export interface VoteResponse {
  ok: boolean;
}

// --- Sorting ---

export interface SortResult {
  id: number;
  similarity: number;
}

export interface SortResponse {
  results: SortResult[];
  threshold: number;
}

export interface LearnedSortResult {
  id: number;
  score: number;
}

export interface LearnedSortResponse {
  results: LearnedSortResult[];
  threshold: number;
}

export interface VotesResponse {
  good: number[];
  bad: number[];
  click_times: Record<string, number>;
  learned_scores: Record<string, number>;
}

export interface InclusionResponse {
  inclusion: number;
}

export interface SafeThresholdsResponse {
  safe_thresholds: boolean;
}

export interface TextsortSuggestionsResponse {
  suggestions: string[];
}

export interface FillFromSortRequest {
  sort_results: { id: number; score: number }[];
  threshold: number;
  sides: string;
  confirm: boolean;
}

export interface FillFromSortDryRunResponse {
  good_count: number;
  bad_count: number;
}

export interface FillFromSortConfirmResponse {
  good_applied: number;
  bad_applied: number;
  results: Record<string, unknown>;
}

export interface DiversityTreeNextResponse {
  id: number | null;
  diversity_level: number;
  exhausted: boolean;
}

// --- Labels ---

export interface LabelEntry {
  md5: string;
  label: 'good' | 'bad';
  origin_name?: string;
  filename?: string;
  category?: string;
  is_correction?: boolean;
  custom_metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface LabelsExportResponse {
  labels: LabelEntry[];
  available_columns?: string[];
}

export interface LabelsImportResponse {
  applied: number;
  skipped: number;
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
  model_id?: string;
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
  [key: string]: unknown;
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
}

export interface ImportersResponse {
  importers: ImporterInfo[];
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
  tab_title?: string;
  folder_import_name?: string;
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
}

export interface DatasetRegistryResponse {
  datasets: DatasetRegistryEntry[];
}

// --- Detectors ---

export interface DetectorInfo {
  name: string;
  media_type?: string;
  autodetect?: boolean;
  [key: string]: unknown;
}

export interface AutorunDetectorsResponse {
  detectors: DetectorInfo[];
}

export interface DetectorCreateResponse {
  success: boolean;
  name: string;
}

export interface DetectorDeleteResponse {
  success: boolean;
}

export interface DetectorRenameResponse {
  success: boolean;
  new_name: string;
}

export interface DetectorSortResponse {
  results: { id: number; score: number }[];
  threshold: number;
}

export interface AutoDetectResponse {
  [key: string]: unknown;
}

// --- Settings ---

export interface AppSettings {
  volume: number;
  inclusion?: number;
  theme?: string;
  enrich_descriptions?: boolean;
  safe_thresholds?: boolean;
  calibrate_count?: number;
  calibration_fraction?: number;
  audio_playing?: boolean;
  swipe_animation?: boolean;
  show_metadata?: boolean;
  view_mode_left?: Record<string, 'grid' | 'list'>;
  view_mode_right?: Record<string, 'grid' | 'list'>;
  grid_icon_size_left?: Record<string, string>;
  grid_icon_size_right?: Record<string, string>;
  focus_mode_left?: Record<string, 'click' | 'hover'>;
  focus_mode_right?: Record<string, 'click' | 'hover'>;
  panel_pct_left?: Record<string, number>;
  panel_pct_right?: Record<string, number>;
  autoload_media_types?: string[];
  autoload_media_embedders?: string[];
  autorun_processors?: AutorunProcessor[];
  autopilot_enabled?: boolean;
  hide_autopilot?: boolean;
  autopilot_top_greens?: number;
  autopilot_hard_reds?: number;
  autopilot_resort_interval?: number;
  autopilot_goal_diversity?: number;
  [key: string]: unknown;
}

export interface AutorunProcessor {
  name: string;
  [key: string]: unknown;
}

// --- Exporters ---

export interface ExporterInfo {
  name: string;
  label?: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
  ui_mode?: string;
  hidden_from_picker?: boolean;
  [key: string]: unknown;
}

// --- Trainable Models ---

export interface TrainableModel {
  name: string;
  [key: string]: unknown;
}

export interface TrainableModelsResponse {
  models: TrainableModel[];
}

export interface ModelRegistryEntry {
  id: string;
  name: string;
  media_type: string;
  trainable?: boolean;
  num_training?: number;
  text_query?: string;
  media_example?: string;
  detector_name?: string;
  loaded?: boolean;
  detector_loaded?: boolean;
  [key: string]: unknown;
}

export interface ModelsRegistryResponse {
  models: ModelRegistryEntry[];
}

// --- Label Importers ---

export interface LabelImporterInfo {
  name: string;
  [key: string]: unknown;
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

// --- Server Files ---

export interface ServerFileEntry {
  name: string;
  filename?: string;
  path?: string;
  size_bytes?: number;
}

export interface ServerMediaFilesResponse {
  files: ServerFileEntry[];
}

export interface DetectorServerFilesResponse {
  files: ServerFileEntry[];
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

export interface TrainAndScoreResponse {
  error_cost?: ErrorCostDataPoint[];
  stability?: StabilityDataPoint[];
  diversity?: DiversityDataPoint[];
  [key: string]: unknown;
}

export interface IndicatorScoreHistoryResponse {
  metric: string;
  history: ErrorCostDataPoint[] | StabilityDataPoint[] | DiversityDataPoint[];
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
  models?: string[];
  datasets?: string[];
  multiple_datasets?: boolean;
  multiple_models?: boolean;
  [key: string]: unknown;
}

// --- Embedders ---

export interface EmbedderInfo {
  name: string;
  media_type_id: string;
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
