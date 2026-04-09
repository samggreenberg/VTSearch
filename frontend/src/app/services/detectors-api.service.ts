import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  AutorunDetectorsResponse,
  DetectorCreateResponse,
  DetectorDeleteResponse,
  DetectorRenameResponse,
  DetectorSortResponse,
  AutoDetectResponse,
  DetectorServerFilesResponse,
} from '../models/api.models';

export interface FindLabelWarning {
  model_name: string;
  total_labels: number;
  resolved_labels: number;
  failed_labels: number;
}

@Injectable({ providedIn: 'root' })
export class DetectorsApiService {
  constructor(private http: HttpClient) {}

  // --- CRUD ---

  getAutorunDetectors(): Observable<AutorunDetectorsResponse> {
    return this.http.get<AutorunDetectorsResponse>('/api/autorun-detectors');
  }

  createDetector(params: { name: string; media_type: string }): Observable<DetectorCreateResponse> {
    return this.http.post<DetectorCreateResponse>('/api/autorun-detectors', params);
  }

  deleteDetector(name: string): Observable<DetectorDeleteResponse> {
    return this.http.delete<DetectorDeleteResponse>(`/api/autorun-detectors/${name}`);
  }

  renameDetector(name: string, newName: string): Observable<DetectorRenameResponse> {
    return this.http.put<DetectorRenameResponse>(`/api/autorun-detectors/${name}/rename`, { new_name: newName });
  }

  setAutodetect(name: string, autodetect: boolean): Observable<unknown> {
    return this.http.put(`/api/autorun-detectors/${name}/autodetect`, { autodetect });
  }

  exportDetector(name: string): Observable<unknown> {
    return this.http.get(`/api/autorun-detectors/${name}/export`);
  }

  exportDetectorToServer(name: string): Observable<unknown> {
    return this.http.post(`/api/autorun-detectors/${name}/export-server`, {});
  }

  getDetectorExamples(name: string): Observable<unknown> {
    return this.http.get(`/api/autorun-detectors/${name}/examples`);
  }

  setDetectorExamples(name: string, examples: unknown[]): Observable<unknown> {
    return this.http.put(`/api/autorun-detectors/${name}/examples`, { examples });
  }

  importDetectorPkl(file: File, name?: string): Observable<unknown> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    if (name) {
      formData.append('name', name);
    }
    return this.http.post('/api/autorun-detectors/import-pkl', formData);
  }

  getServerFiles(): Observable<DetectorServerFilesResponse> {
    return this.http.get<DetectorServerFilesResponse>('/api/detector/server-files');
  }

  getServerFile(name: string): Observable<unknown> {
    return this.http.get(`/api/detector/server-files/${name}`);
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

  detectorSort(params: Record<string, unknown>): Observable<DetectorSortResponse> {
    return this.http.post<DetectorSortResponse>('/api/detector-sort', params);
  }

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

  // --- Training ---

  exportWeightsToServer(): Observable<unknown> {
    return this.http.post('/api/detector/export-server', {});
  }

  importFromLabels(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/autorun-detectors/import-labels', params);
  }

  importFromLabelImporter(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/autorun-detectors/from-label-import/${importerName}`, params);
  }

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
