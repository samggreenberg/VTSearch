import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  SortResponse,
  LearnedSortResponse,
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
  SortProgressResponse,
  LabelingStatusResponse,
  DiversityTreeNextResponse,
  ServerMediaFilesResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class SortingApiService {
  constructor(private http: HttpClient) {}

  sort(params: { text: string }): Observable<SortResponse> {
    return this.http.post<SortResponse>('/api/sort', params);
  }

  getSortProgress(): Observable<SortProgressResponse> {
    return this.http.get<SortProgressResponse>('/api/sort/progress');
  }

  learnedSort(): Observable<LearnedSortResponse> {
    return this.http.post<LearnedSortResponse>('/api/learned-sort', {});
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

  exportLabels(goodsOnly?: boolean): Observable<LabelsExportResponse> {
    const params: Record<string, string> = {};
    if (goodsOnly) {
      params['goods_only'] = 'true';
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

  exampleSort(file: File): Observable<SortResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<SortResponse>('/api/example-sort', formData);
  }

  getServerMediaFiles(): Observable<ServerMediaFilesResponse> {
    return this.http.get<ServerMediaFilesResponse>('/api/server-media-files');
  }

  exampleSortServer(params: { filename: string }): Observable<SortResponse> {
    return this.http.post<SortResponse>('/api/example-sort-server', params);
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

  getDiversityTreeNext(scores?: Record<string, number>, threshold?: number): Observable<DiversityTreeNextResponse> {
    if (scores) {
      return this.http.post<DiversityTreeNextResponse>('/api/diversity-tree/next', { scores, threshold });
    }
    return this.http.get<DiversityTreeNextResponse>('/api/diversity-tree/next');
  }
}
