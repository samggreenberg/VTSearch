import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AutoDetectResponse } from '../models/api.models';

export interface FindLabelWarning {
  model_name: string;
  total_labels: number;
  resolved_labels: number;
  failed_labels: number;
}

/**
 * API surface for trainable-model scoring and shared processor (extractor /
 * localizer) endpoints.  Detectors no longer exist as a separate concept —
 * every "model" is a trainable model managed via TrainableModelsApiService
 * and the model registry.
 */
@Injectable({ providedIn: 'root' })
export class DetectorsApiService {
  constructor(private http: HttpClient) {}

  // --- Autorun toggle (model registry) ---

  setAutorun(modelId: string, autorun: boolean): Observable<unknown> {
    return this.http.put(`/api/models/registry/${modelId}/autorun`, { autorun });
  }

  // --- Extractors ---

  getAutorunExtractors(): Observable<{ extractors: unknown[] }> {
    return this.http.get<{ extractors: unknown[] }>('/api/autorun-extractors');
  }

  createExtractor(params: { name: string; media_type?: string }): Observable<unknown> {
    return this.http.post('/api/autorun-extractors', params);
  }

  deleteExtractor(name: string): Observable<unknown> {
    return this.http.delete(`/api/autorun-extractors/${name}`);
  }

  renameExtractor(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/autorun-extractors/${name}/rename`, { new_name: newName });
  }

  // --- Localizers ---

  getAutorunLocalizers(): Observable<{ localizers: unknown[] }> {
    return this.http.get<{ localizers: unknown[] }>('/api/autorun-localizers');
  }

  createLocalizer(params: { name: string; media_type?: string }): Observable<unknown> {
    return this.http.post('/api/autorun-localizers', params);
  }

  deleteLocalizer(name: string): Observable<unknown> {
    return this.http.delete(`/api/autorun-localizers/${name}`);
  }

  renameLocalizer(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/autorun-localizers/${name}/rename`, { new_name: newName });
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
