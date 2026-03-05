/* TypeScript interfaces matching the Flask API response shapes. */

// --- Medias ---

export interface MediaItem {
  id: number;
  type: string;
  duration: number;
  file_size: number;
  filename: string;
  category: string;
  md5: string;
  origin_name?: string;
  description?: string;
}

export interface ParagraphResponse {
  text: string;
  stats?: Record<string, unknown>;
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
  results: unknown[];
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
}

export interface LabelsExportResponse {
  labels: LabelEntry[];
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
  [key: string]: unknown;
}

export interface ImporterInfo {
  name: string;
  label?: string;
  description?: string;
  fields?: ImporterField[];
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
}

export interface DemoListResponse {
  datasets: DemoDataset[];
}

export interface MediaTypeInfo {
  type_id: string;
  name: string;
  icon?: string;
  tab_title?: string;
}

export interface MediaTypesResponse {
  media_types: MediaTypeInfo[];
}

export interface DatasetRegistryEntry {
  [key: string]: unknown;
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
  swipe_animation?: boolean;
  show_thumbnails_left?: boolean;
  show_thumbnails_right?: boolean;
  autoload_media_types?: string[];
  autorun_processors?: AutorunProcessor[];
  autopilot_top_greens?: number;
  autopilot_hard_reds?: number;
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
  description?: string;
  fields?: ImporterField[];
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

export interface ModelsRegistryResponse {
  models: unknown[];
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

// --- Converters ---

export interface ConverterInfo {
  [key: string]: unknown;
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

export interface ServerMediaFilesResponse {
  files: string[];
}

export interface DetectorServerFilesResponse {
  files: string[];
}
