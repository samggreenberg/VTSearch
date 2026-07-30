/* TypeScript interfaces matching the Flask API response shapes.
 *
 * Everything the backend describes in its OpenAPI schemas is re-exported
 * from `../generated/api-client` rather than mirrored by hand, so a backend
 * field rename breaks this build instead of silently reaching runtime.  What
 * remains hand-written below is exactly what the spec cannot describe:
 *
 * - **SSE payloads** (`ProgressEvent`, `LoadingTask`,
 *   `VotingIterationsResponse`) arrive over `/api/events`, a raw streaming
 *   `Response` that flask-smorest does not model.
 * - **Client-side shapes** (`PayloadVariant`, `AutoDetectResultsData`'s
 *   Find-mode fields) that no endpoint returns.
 * - **Request-side settings shapes** (`SourceSpec`, `ImportDefaults*`,
 *   `CleanerSelection`) the frontend builds and posts.
 */

import type { MediaIdsListResponse } from '../generated/api-client/models/media-ids-list-response';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';
import type { AutoDetectResult } from '../generated/api-client/models/auto-detect-result';
import type { AutoFindExportStatus } from '../generated/api-client/models/auto-find-export-status';
import type { FindResultRow } from '../generated/api-client/models/find-result-row';
import type { Hit } from '../generated/api-client/models/hit';

// Plugin-metadata and listing payloads, all described by nested Marshmallow
// schemas in `vtsearch/schemas/datasets.py` and `vtsearch/schemas/eval.py`.
export type { EmbedderInfo } from '../generated/api-client/models/embedder-info';
export type { MediaTypeInfo } from '../generated/api-client/models/media-type-info';
export type { ConverterInfo } from '../generated/api-client/models/converter-info';
export type { ImporterInfo } from '../generated/api-client/models/importer-info';
export type { ImporterField } from '../generated/api-client/models/importer-field';
export type { ImporterPickerTab } from '../generated/api-client/models/importer-picker-tab';
export type { ClipperInfo } from '../generated/api-client/models/clipper-info';
export type { ClipperParameter } from '../generated/api-client/models/clipper-parameter';
export type { CleanerInfo } from '../generated/api-client/models/cleaner-info';
export type { DatasetRegistryEntry } from '../generated/api-client/models/dataset-registry-entry';
export type { DetectMediaTypeResponse } from '../generated/api-client/models/detect-media-type-response';
export type { FieldOptions } from '../generated/api-client/models/field-options';
export type { ErrorCostPoint } from '../generated/api-client/models/error-cost-point';
export type { StabilityPoint } from '../generated/api-client/models/stability-point';
export type { DiversityPoint } from '../generated/api-client/models/diversity-point';

// --- Medias ---

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
  Partial<Omit<MediaBatchResponse, keyof MediaIdsListResponse>>;

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
  /**
   * Detector-only: terminal counts published by a labelset-media ingest task
   * (`{ingested, applied, unresolved, failed}`); see
   * `vtscore/datasets/ingest_task.py`. `null`/absent until the task finishes.
   */
  ingest_result?: unknown;
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

// --- Cleaners ---

/** A cleaner the user enabled for an import, with its parameter overrides.
 *  Serialised as the `cleaners` field of every import request. */
export interface CleanerSelection {
  name: string;
  params: Record<string, number | string>;
}

/** Which payload of an item a detail viewer should show: the canonical
 *  (cleaned) one, or the pre-clean snapshot a cleaner left behind.  `''` means
 *  canonical and is dropped from the request, so an item with no snapshot is
 *  fetched exactly as it was before cleaners existed. */
export type PayloadVariant = '' | 'original';

// --- Eval / Progress Charts ---

export interface VotingIterationsResponse {
  progress: number;
  total: number;
  done: boolean;
  [key: string]: unknown;
}

// --- Auto-detect Results ---

export type { AutoFindExportStatus } from '../generated/api-client/models/auto-find-export-status';

/**
 * One row of the Auto-Find results table.
 *
 * The table is fed from two endpoints with different guarantees: `Hit` from
 * `POST /api/auto-detect`, and `FindResultRow` from `POST /api/find` (adapted
 * into this shape by `dashboard.component.ts`). Every field is therefore
 * optional — but the *names and types* still come from the generated models,
 * so a backend rename breaks the template that renders the column.
 *
 * `label` is the exception: the modal stamps it client-side when showing the
 * good and bad sides in one list.
 */
export type AutoDetectHit = Partial<Hit> &
  Partial<Pick<FindResultRow, 'dataset_name' | 'detector_verdicts'>> & {
    /** `'good'` / `'bad'`, set by the modal when both sides share one list. */
    label?: string;
  };

/** One detector's column of results, from either source (see `AutoDetectHit`). */
export type AutoDetectDetectorResult = Partial<
  Omit<AutoDetectResult, 'hits' | 'negative_hits'>
> & {
  hits: AutoDetectHit[];
  negative_hits?: AutoDetectHit[];
};

/**
 * What the Auto-Find results modal renders: the `POST /api/auto-detect`
 * response, or the `POST /api/find` response adapted to look like one. Find
 * mode contributes the four `detectors`/`datasets` fields, which no endpoint
 * returns under these names.
 */
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
}

