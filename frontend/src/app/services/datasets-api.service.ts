import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  DatasetStatus,
  DatasetProgress,
  ImportersResponse,
  DemoListResponse,
  MediaTypesResponse,
  DatasetRegistryResponse,
  ConverterInfo,
  OkResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class DatasetsApiService {
  constructor(private http: HttpClient) {}

  getStatus(): Observable<DatasetStatus> {
    return this.http.get<DatasetStatus>('/api/dataset/status');
  }

  getProgress(): Observable<DatasetProgress> {
    return this.http.get<DatasetProgress>('/api/dataset/progress');
  }

  getImporters(): Observable<ImportersResponse> {
    return this.http.get<ImportersResponse>('/api/dataset/importers');
  }

  getAllImporters(): Observable<ImportersResponse> {
    return this.http.get<ImportersResponse>('/api/dataset/all-importers');
  }

  getDemoList(): Observable<DemoListResponse> {
    return this.http.get<DemoListResponse>('/api/dataset/demo-list');
  }

  getMediaTypes(): Observable<MediaTypesResponse> {
    return this.http.get<MediaTypesResponse>('/api/media-types');
  }

  getConverters(target?: string): Observable<ConverterInfo[]> {
    const params: Record<string, string> = {};
    if (target) {
      params['target'] = target;
    }
    return this.http.get<ConverterInfo[]>('/api/converters', { params });
  }

  getAvailableFiles(): Observable<{ files: string[] }> {
    return this.http.get<{ files: string[] }>('/api/dataset/available-files');
  }

  runImporter(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/import/${importerName}`, params);
  }

  loadDemo(name: string): Observable<unknown> {
    return this.http.post('/api/dataset/load-demo', { name });
  }

  loadFile(file: File): Observable<unknown> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post('/api/dataset/load-file', formData);
  }

  stageFile(file: File): Observable<unknown> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post('/api/dataset/stage-file', formData);
  }

  stageImport(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/stage-import/${importerName}`, params);
  }

  stageDemo(name: string): Observable<unknown> {
    return this.http.post(`/api/dataset/stage-demo/${name}`, {});
  }

  clearStaging(): Observable<unknown> {
    return this.http.delete('/api/dataset/staging');
  }

  combineDatasets(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/dataset/combine', params);
  }

  clearDataset(): Observable<OkResponse> {
    return this.http.post<OkResponse>('/api/dataset/clear', {});
  }

  exportDataset(): Observable<Blob> {
    return this.http.get('/api/dataset/export', { responseType: 'blob' });
  }

  getRegistry(): Observable<DatasetRegistryResponse> {
    return this.http.get<DatasetRegistryResponse>('/api/datasets/registry');
  }

  loadRegistered(datasetId: string): Observable<unknown> {
    return this.http.post(`/api/datasets/registry/${datasetId}/load`, {});
  }

  unloadRegistered(datasetId: string): Observable<unknown> {
    return this.http.post(`/api/datasets/registry/${datasetId}/unload`, {});
  }

  deleteRegistered(datasetId: string): Observable<unknown> {
    return this.http.delete(`/api/datasets/registry/${datasetId}`);
  }

  renameRegistered(datasetId: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/datasets/registry/${datasetId}/rename`, { name: newName });
  }

  loadSource(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/dataset/load-source', params);
  }
}
