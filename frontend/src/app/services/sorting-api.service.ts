import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DiversityTreeNextResponse } from '../generated/api-client/models/diversity-tree-next-response';
import type { EvalTrainAndScoreResponse } from '../generated/api-client/models/eval-train-and-score-response';
import type { ExampleSortResponse } from '../generated/api-client/models/example-sort-response';
import type { FillFromSortRequest } from '../generated/api-client/models/fill-from-sort-request';
import type { FillFromSortResponse } from '../generated/api-client/models/fill-from-sort-response';
import type { InclusionResponse } from '../generated/api-client/models/inclusion-response';
import type { IndicatorScoreHistoryResponse } from '../generated/api-client/models/indicator-score-history-response';
import type { LabelFileSortResponse } from '../generated/api-client/models/label-file-sort-response';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';
import type { LabelsExportResponse } from '../generated/api-client/models/labels-export-response';
import type { LabelsImportRequest } from '../generated/api-client/models/labels-import-request';
import type { LabelsImportResponse } from '../generated/api-client/models/labels-import-response';
import type { LearnedSortResponse } from '../generated/api-client/models/learned-sort-response';
import type { OkResponse } from '../generated/api-client/models/ok-response';
import type { SafeThresholdsResponse } from '../generated/api-client/models/safe-thresholds-response';
import type { ServerMediaListResponse } from '../generated/api-client/models/server-media-list-response';
import type { ServerMediaUploadResponse } from '../generated/api-client/models/server-media-upload-response';
import type { SortResponse } from '../generated/api-client/models/sort-response';
import type { TextsortSuggestionsResponse } from '../generated/api-client/models/textsort-suggestions-response';
import type { VotesResponse } from '../generated/api-client/models/votes-response';
import { apiDiversityTreeNextGet } from '../generated/api-client/fn/sorting/api-diversity-tree-next-get';
import { apiInclusionGet } from '../generated/api-client/fn/sorting/api-inclusion-get';
import { apiInclusionPost } from '../generated/api-client/fn/sorting/api-inclusion-post';
import { apiLearnedSortPost } from '../generated/api-client/fn/sorting/api-learned-sort-post';
import { apiLearnedSortResultGet } from '../generated/api-client/fn/sorting/api-learned-sort-result-get';
import { apiSafeThresholdsGet } from '../generated/api-client/fn/sorting/api-safe-thresholds-get';
import { apiSafeThresholdsPost } from '../generated/api-client/fn/sorting/api-safe-thresholds-post';
import { apiSortPost } from '../generated/api-client/fn/sorting/api-sort-post';
import { apiTextsortSuggestionsGet } from '../generated/api-client/fn/sorting/api-textsort-suggestions-get';
import { apiTextsortSuggestionsPost } from '../generated/api-client/fn/sorting/api-textsort-suggestions-post';
import { apiVotesClearPost } from '../generated/api-client/fn/sorting/api-votes-clear-post';
import { apiVotesGet } from '../generated/api-client/fn/sorting/api-votes-get';
import { apiLabelsExportGet } from '../generated/api-client/fn/labels/api-labels-export-get';
import { apiLabelsFillFromSortPost } from '../generated/api-client/fn/labels/api-labels-fill-from-sort-post';
import { apiLabelsImportPost } from '../generated/api-client/fn/labels/api-labels-import-post';
import { apiEvalTrainAndScorePost } from '../generated/api-client/fn/eval/api-eval-train-and-score-post';
import { apiEvalTrainAndScoreResultGet } from '../generated/api-client/fn/eval/api-eval-train-and-score-result-get';
import { apiIndicatorScoreHistoryGet } from '../generated/api-client/fn/eval/api-indicator-score-history-get';
import { apiLabelingStatusGet } from '../generated/api-client/fn/eval/api-labeling-status-get';
import { apiExampleSortByIdPost } from '../generated/api-client/fn/media-server/api-example-sort-by-id-post';
import { apiExampleSortOriginPost } from '../generated/api-client/fn/media-server/api-example-sort-origin-post';
import { apiExampleSortServerPost } from '../generated/api-client/fn/media-server/api-example-sort-server-post';
import { apiServerMediaFilesFromMediaIdPost } from '../generated/api-client/fn/media-server/api-server-media-files-from-media-id-post';
import { apiServerMediaFilesGet } from '../generated/api-client/fn/media-server/api-server-media-files-get';

@Injectable({ providedIn: 'root' })
export class SortingApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  sort(params: { text: string }): Observable<SortResponse> {
    return apiSortPost(this.http, this.config.rootUrl, { body: params }).pipe(map((r) => r.body));
  }

  /** Kick off a learned-sort training job.  The response will be ``done``
   *  immediately when the cached signature matches; otherwise the caller
   *  must poll {@link getLearnedSortResult} with the returned ``job_id``. */
  learnedSort(): Observable<LearnedSortResponse> {
    return apiLearnedSortPost(this.http, this.config.rootUrl, { body: {} }).pipe(map((r) => r.body));
  }

  /** Poll for a learned-sort job's completion. */
  getLearnedSortResult(jobId: string): Observable<LearnedSortResponse> {
    return apiLearnedSortResultGet(this.http, this.config.rootUrl, { job_id: jobId }).pipe(
      map((r) => r.body),
    );
  }

  getVotes(): Observable<VotesResponse> {
    return apiVotesGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  clearVotes(): Observable<OkResponse> {
    return apiVotesClearPost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getInclusion(): Observable<InclusionResponse> {
    return apiInclusionGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  setInclusion(value: number): Observable<InclusionResponse> {
    return apiInclusionPost(this.http, this.config.rootUrl, { body: { inclusion: value } }).pipe(
      map((r) => r.body),
    );
  }

  getSafeThresholds(): Observable<SafeThresholdsResponse> {
    return apiSafeThresholdsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  setSafeThresholds(value: boolean): Observable<SafeThresholdsResponse> {
    return apiSafeThresholdsPost(this.http, this.config.rootUrl, {
      body: { safe_thresholds: value },
    }).pipe(map((r) => r.body));
  }

  exportLabels(goodsOnly?: boolean, options?: { enrich?: boolean }): Observable<LabelsExportResponse> {
    return apiLabelsExportGet(this.http, this.config.rootUrl, {
      goods_only: goodsOnly || undefined,
      enrich: options?.enrich || undefined,
    }).pipe(map((r) => r.body));
  }

  importLabels(data: LabelsImportRequest): Observable<LabelsImportResponse> {
    return apiLabelsImportPost(this.http, this.config.rootUrl, { body: data }).pipe(map((r) => r.body));
  }

  fillFromSort(request: FillFromSortRequest): Observable<FillFromSortResponse> {
    return apiLabelsFillFromSortPost(this.http, this.config.rootUrl, { body: request }).pipe(
      map((r) => r.body),
    );
  }

  /** Multipart upload — stays on plain HttpClient because ng-openapi-gen
   *  doesn't model multipart bodies (the generated function's ``$Params``
   *  has no ``body`` field). */
  exampleSort(file: File, cropParams?: Record<string, unknown>): Observable<SortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (cropParams) {
      formData.append('crop_params', JSON.stringify(cropParams));
    }
    return this.http.post<SortResponse>('/api/example-sort', formData);
  }

  getServerMediaFiles(): Observable<ServerMediaListResponse> {
    return apiServerMediaFilesGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  exampleSortServer(params: {
    filename: string;
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return apiExampleSortServerPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  exampleSortOrigin(params: {
    origin: Record<string, unknown>;
    key: string;
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return apiExampleSortOriginPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Sort the loaded snapshot by similarity to an already-loaded media.
   *  Skips re-embedding when ``crop_params`` is absent — the in-memory
   *  embedding is reused directly. */
  exampleSortById(params: {
    media_id: number;
    crop_params?: Record<string, unknown>;
  }): Observable<ExampleSortResponse> {
    return apiExampleSortByIdPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Save a loaded media's bytes to example_media/ so the new-detector
   *  flow can reference it via ``media_example``. */
  saveServerMediaFromMediaId(params: {
    media_id: number;
    crop_params?: Record<string, unknown>;
  }): Observable<ServerMediaUploadResponse> {
    return apiServerMediaFilesFromMediaIdPost(this.http, this.config.rootUrl, {
      body: params,
    }).pipe(map((r) => r.body));
  }

  /** Multipart upload — see {@link exampleSort}. */
  uploadServerMediaFile(
    file: File,
    options?: { mediaType?: string; cropParams?: Record<string, unknown> },
  ): Observable<ServerMediaUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.cropParams) {
      formData.append('crop_params', JSON.stringify(options.cropParams));
      if (options.mediaType) {
        formData.append('media_type', options.mediaType);
      }
    }
    return this.http.post<ServerMediaUploadResponse>('/api/server-media-files/upload', formData);
  }

  /** Multipart upload — see {@link exampleSort}. */
  labelFileSort(file: File): Observable<LabelFileSortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<LabelFileSortResponse>('/api/label-file-sort', formData);
  }

  getTextsortSuggestions(): Observable<TextsortSuggestionsResponse> {
    return apiTextsortSuggestionsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  addTextsortSuggestion(text: string): Observable<OkResponse> {
    return apiTextsortSuggestionsPost(this.http, this.config.rootUrl, { body: { text } }).pipe(
      map((r) => r.body),
    );
  }

  /** ``/api/labeling-progress`` reads global state (votes, label history) and
   *  takes no request body — the spec's ``ApiLabelingProgressPost$Params``
   *  reflects that.  Production callers were removed; the method is kept for
   *  parity with the legacy surface. */
  getLabelingProgress(): Observable<unknown> {
    return this.http.post('/api/labeling-progress', {});
  }

  getLabelingStatus(): Observable<LabelingStatusResponse> {
    return apiLabelingStatusGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getIndicatorScoreHistory(
    metric: 'smart' | 'stable' | 'diverse',
  ): Observable<IndicatorScoreHistoryResponse> {
    return apiIndicatorScoreHistoryGet(this.http, this.config.rootUrl, { metric }).pipe(
      map((r) => r.body),
    );
  }

  /** The POST branch carries an optional ``{scores, threshold}`` body that the
   *  backend reads via ``request.get_json(silent=True)`` — the OpenAPI spec
   *  intentionally omits that body so GET and POST share one declaration, so
   *  the POST call stays on plain ``HttpClient``.  The GET branch uses the
   *  generated function. */
  getDiversityTreeNext(
    scores?: Record<string, number>,
    threshold?: number,
  ): Observable<DiversityTreeNextResponse> {
    if (scores) {
      return this.http.post<DiversityTreeNextResponse>('/api/diversity-tree/next', { scores, threshold });
    }
    return apiDiversityTreeNextGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Kick off an eval train-and-score job.  Like {@link learnedSort}, the
   *  response will be ``done`` immediately on a cache hit; otherwise poll
   *  {@link getEvalTrainAndScoreResult}. */
  trainAndScore(metric: 'smart' | 'stable' | 'diverse'): Observable<EvalTrainAndScoreResponse> {
    return apiEvalTrainAndScorePost(this.http, this.config.rootUrl, { body: { metric } }).pipe(
      map((r) => r.body),
    );
  }

  getEvalTrainAndScoreResult(jobId: string): Observable<EvalTrainAndScoreResponse> {
    return apiEvalTrainAndScoreResultGet(this.http, this.config.rootUrl, { job_id: jobId }).pipe(
      map((r) => r.body),
    );
  }
}
