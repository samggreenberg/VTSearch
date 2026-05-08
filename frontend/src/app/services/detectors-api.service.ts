import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  AutoDetectResponse,
  CombineDetectorsResult,
  DetectorsRegistryResponse,
  DetectorsResponse,
  LabelsDetailResponse,
  LoadingTasksResponse,
} from '../models/api.models';

export interface FindLabelWarning {
  detector_name: string;
  total_labels: number;
  resolved_labels: number;
  failed_labels: number;
}

/**
 * API surface for detector CRUD, the detector registry, and detector-driven
 * scoring (auto-detect, find, find-label).  Also covers the autorun
 * extractor / localizer / pregen-processor endpoints, which share the
 * detector lifecycle from the dashboard's perspective.
 */
@Injectable({ providedIn: 'root' })
export class DetectorsApiService {
  constructor(private http: HttpClient) {}

  // --- Detector CRUD ---

  list(): Observable<DetectorsResponse> {
    return this.http.get<DetectorsResponse>('/api/detectors');
  }

  create(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/detectors', params);
  }

  get(name: string): Observable<unknown> {
    return this.http.get(`/api/detectors/${encodeURIComponent(name)}`);
  }

  delete(name: string): Observable<unknown> {
    return this.http.delete(`/api/detectors/${encodeURIComponent(name)}`);
  }

  rename(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/detectors/${encodeURIComponent(name)}/rename`, { new_name: newName });
  }

  setExamples(name: string, examples: unknown[]): Observable<unknown> {
    return this.http.put(`/api/detectors/${encodeURIComponent(name)}/examples`, { examples });
  }

  saveLabels(name: string): Observable<unknown> {
    return this.http.post(`/api/detectors/${encodeURIComponent(name)}/labels`, {});
  }

  getLabelsDetail(name: string): Observable<LabelsDetailResponse> {
    return this.http.get<LabelsDetailResponse>(
      `/api/detectors/${encodeURIComponent(name)}/labels-detail`,
    );
  }

  voteLabelElement(name: string, elementId: string, vote: 'good' | 'bad'): Observable<unknown> {
    return this.http.post(
      `/api/detectors/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/vote`,
      { vote },
    );
  }

  labelPreviewUrl(name: string, elementId: string): string {
    return `/api/detectors/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/preview`;
  }

  labelThumbnailUrl(name: string, elementId: string): string {
    return `/api/detectors/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/thumbnail`;
  }

  combine(
    names: string[],
    newName: string,
    conflictPolicy: 'drop' = 'drop',
  ): Observable<CombineDetectorsResult> {
    return this.http.post<CombineDetectorsResult>('/api/detectors/combine', {
      names,
      new_name: newName,
      conflict_policy: conflictPolicy,
    });
  }

  // --- Detector Registry ---

  getRegistry(): Observable<DetectorsRegistryResponse> {
    return this.http.get<DetectorsRegistryResponse>('/api/detectors/registry');
  }

  registerDetector(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/detectors/registry', params);
  }

  registerDetectorFromLabelset(
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown> {
    const url = `/api/detectors/registry/from-labelset/${encodeURIComponent(importerName)}`;
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post(url, formData);
    }
    return this.http.post(url, params);
  }

  deleteFromRegistry(detectorId: string): Observable<unknown> {
    return this.http.delete(`/api/detectors/registry/${encodeURIComponent(detectorId)}`);
  }

  renameInRegistry(detectorId: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/detectors/registry/${encodeURIComponent(detectorId)}/rename`, { name: newName });
  }

  loadDetector(detectorId: string | null): Observable<unknown> {
    return this.http.post('/api/detectors/registry/load', { detector_id: detectorId });
  }

  unloadDetector(detectorId: string): Observable<unknown> {
    return this.http.post(`/api/detectors/registry/${encodeURIComponent(detectorId)}/unload`, {});
  }

  getDetectorLoadingTasks(): Observable<LoadingTasksResponse> {
    return this.http.get<LoadingTasksResponse>('/api/detectors/loading-tasks');
  }

  cancelDetectorLoadingTask(taskId: string): Observable<unknown> {
    return this.http.post(`/api/detectors/cancel/${encodeURIComponent(taskId)}`, {});
  }

  setAutorun(detectorId: string, autorun: boolean): Observable<unknown> {
    return this.http.put(`/api/detectors/registry/${encodeURIComponent(detectorId)}/autorun`, { autorun });
  }

  // --- Extractors ---

  getAutorunExtractors(): Observable<{ extractors: unknown[] }> {
    return this.http.get<{ extractors: unknown[] }>('/api/autorun-extractors');
  }

  createExtractor(params: { name: string; media_type?: string }): Observable<unknown> {
    return this.http.post('/api/autorun-extractors', params);
  }

  deleteExtractor(name: string): Observable<unknown> {
    return this.http.delete(`/api/autorun-extractors/${encodeURIComponent(name)}`);
  }

  renameExtractor(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/autorun-extractors/${encodeURIComponent(name)}/rename`, { new_name: newName });
  }

  // --- Localizers ---

  getAutorunLocalizers(): Observable<{ localizers: unknown[] }> {
    return this.http.get<{ localizers: unknown[] }>('/api/autorun-localizers');
  }

  createLocalizer(params: { name: string; media_type?: string }): Observable<unknown> {
    return this.http.post('/api/autorun-localizers', params);
  }

  deleteLocalizer(name: string): Observable<unknown> {
    return this.http.delete(`/api/autorun-localizers/${encodeURIComponent(name)}`);
  }

  renameLocalizer(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/autorun-localizers/${encodeURIComponent(name)}/rename`, { new_name: newName });
  }

  // --- Scoring ---

  autoDetect(params: Record<string, unknown>): Observable<AutoDetectResponse> {
    return this.http.post<AutoDetectResponse>('/api/auto-detect', params);
  }

  extract(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/extract', params);
  }

  autoExtract(): Observable<unknown> {
    return this.http.post('/api/auto-extract', {});
  }

  localize(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/localize', params);
  }

  autoLocalize(): Observable<unknown> {
    return this.http.post('/api/auto-localize', {});
  }

  // --- Find ---

  findCheckLabels(params: Record<string, unknown>): Observable<{ warnings: FindLabelWarning[] }> {
    return this.http.post<{ warnings: FindLabelWarning[] }>('/api/find/check-labels', params);
  }

  find(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/find', params);
  }

  getFindProgress(): Observable<unknown> {
    return this.http.get('/api/find/progress');
  }

  findLabel(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/find-label', params);
  }

  // --- Pregen processors ---

  getPregenProcessors(): Observable<{ processors: unknown[] }> {
    return this.http.get<{ processors: unknown[] }>('/api/pregen-processors');
  }

  addPregenProcessors(): Observable<unknown> {
    return this.http.post('/api/pregen-processors/add', {});
  }
}
