import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  SortResponse,
  LearnedSortResponse,
  LearnedSortJobResponse,
  VotesResponse,
  InclusionResponse,
  SafeThresholdsResponse,
  LabelsExportResponse,
  LabelsImportResponse,
  FillFromSortRequest,
  FillFromSortDryRunResponse,
  FillFromSortConfirmResponse,
  TextsortSuggestionsResponse,
  OkResponse,
  LabelingStatusResponse,
  DiversityTreeNextResponse,
  ServerMediaFilesResponse,
  EvalTrainAndScoreJobResponse,
  IndicatorScoreHistoryResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class SortingApiService {
  constructor(private http: HttpClient) {}

  sort(params: { text: string }): Observable<SortResponse> {
    return this.http.post<SortResponse>('/api/sort', params);
  }

  /** Kick off a learned-sort training job.  The response will be ``done``
   *  immediately when the cached signature matches; otherwise the caller
   *  must poll {@link getLearnedSortResult} with the returned ``job_id``. */
  learnedSort(): Observable<LearnedSortJobResponse> {
    return this.http.post<LearnedSortJobResponse>('/api/learned-sort', {});
  }

  /** Poll for a learned-sort job's completion. */
  getLearnedSortResult(jobId: string): Observable<LearnedSortJobResponse> {
    return this.http.get<LearnedSortJobResponse>('/api/learned-sort/result', {
      params: { job_id: jobId },
    });
  }

  getVotes(): Observable<VotesResponse> {
    return this.http.get<VotesResponse>('/api/votes');
  }

  clearVotes(): Observable<OkResponse> {
    return this.http.post<OkResponse>('/api/votes/clear', {});
  }

  getInclusion(): Observable<InclusionResponse> {
    return this.http.get<InclusionResponse>('/api/inclusion');
  }

  setInclusion(value: number): Observable<InclusionResponse> {
    return this.http.post<InclusionResponse>('/api/inclusion', { inclusion: value });
  }

  getSafeThresholds(): Observable<SafeThresholdsResponse> {
    return this.http.get<SafeThresholdsResponse>('/api/safe-thresholds');
  }

  setSafeThresholds(value: boolean): Observable<SafeThresholdsResponse> {
    return this.http.post<SafeThresholdsResponse>('/api/safe-thresholds', { safe_thresholds: value });
  }

  exportLabels(goodsOnly?: boolean, options?: { enrich?: boolean }): Observable<LabelsExportResponse> {
    const params: Record<string, string> = {};
    if (goodsOnly) {
      params['goods_only'] = 'true';
    }
    if (options?.enrich) {
      params['enrich'] = 'true';
    }
    return this.http.get<LabelsExportResponse>('/api/labels/export', { params });
  }

  importLabels(data: { labels: { md5: string; label: string }[] }): Observable<LabelsImportResponse> {
    return this.http.post<LabelsImportResponse>('/api/labels/import', data);
  }

  fillFromSort(request: FillFromSortRequest): Observable<FillFromSortDryRunResponse | FillFromSortConfirmResponse> {
    return this.http.post<FillFromSortDryRunResponse | FillFromSortConfirmResponse>(
      '/api/labels/fill-from-sort',
      request,
    );
  }

  exampleSort(file: File, cropParams?: Record<string, unknown>): Observable<SortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (cropParams) {
      formData.append('crop_params', JSON.stringify(cropParams));
    }
    return this.http.post<SortResponse>('/api/example-sort', formData);
  }

  getServerMediaFiles(): Observable<ServerMediaFilesResponse> {
    return this.http.get<ServerMediaFilesResponse>('/api/server-media-files');
  }

  exampleSortServer(params: { filename: string; crop_params?: Record<string, unknown> }): Observable<SortResponse> {
    return this.http.post<SortResponse>('/api/example-sort-server', params);
  }

  exampleSortOrigin(
    params: { origin: Record<string, unknown>; key: string; crop_params?: Record<string, unknown> },
  ): Observable<SortResponse> {
    return this.http.post<SortResponse>('/api/example-sort-origin', params);
  }

  uploadServerMediaFile(
    file: File,
    options?: { mediaType?: string; cropParams?: Record<string, unknown> },
  ): Observable<{ filename: string; original_name: string }> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.cropParams) {
      formData.append('crop_params', JSON.stringify(options.cropParams));
      if (options.mediaType) {
        formData.append('media_type', options.mediaType);
      }
    }
    return this.http.post<{ filename: string; original_name: string }>('/api/server-media-files/upload', formData);
  }

  labelFileSort(file: File): Observable<LearnedSortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<LearnedSortResponse>('/api/label-file-sort', formData);
  }

  getTextsortSuggestions(): Observable<TextsortSuggestionsResponse> {
    return this.http.get<TextsortSuggestionsResponse>('/api/textsort-suggestions');
  }

  addTextsortSuggestion(text: string): Observable<OkResponse> {
    return this.http.post<OkResponse>('/api/textsort-suggestions', { text });
  }

  getLabelingProgress(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/labeling-progress', params);
  }

  getLabelingStatus(): Observable<LabelingStatusResponse> {
    return this.http.get<LabelingStatusResponse>('/api/labeling-status');
  }

  getIndicatorScoreHistory(metric: string): Observable<IndicatorScoreHistoryResponse> {
    return this.http.get<IndicatorScoreHistoryResponse>('/api/indicator-score-history', {
      params: { metric },
    });
  }

  getDiversityTreeNext(scores?: Record<string, number>, threshold?: number): Observable<DiversityTreeNextResponse> {
    if (scores) {
      return this.http.post<DiversityTreeNextResponse>('/api/diversity-tree/next', { scores, threshold });
    }
    return this.http.get<DiversityTreeNextResponse>('/api/diversity-tree/next');
  }

  /** Kick off an eval train-and-score job.  Like {@link learnedSort}, the
   *  response will be ``done`` immediately on a cache hit; otherwise poll
   *  {@link getEvalTrainAndScoreResult}. */
  trainAndScore(metric: string): Observable<EvalTrainAndScoreJobResponse> {
    return this.http.post<EvalTrainAndScoreJobResponse>('/api/eval/train-and-score', { metric });
  }

  getEvalTrainAndScoreResult(jobId: string): Observable<EvalTrainAndScoreJobResponse> {
    return this.http.get<EvalTrainAndScoreJobResponse>('/api/eval/train-and-score/result', {
      params: { job_id: jobId },
    });
  }

}
