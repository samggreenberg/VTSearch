import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import {
  DatasetStatus,
  DatasetStatsResponse,
  ImportersResponse,
  DemoListResponse,
  MediaTypesResponse,
  MediaTypeDetectionResponse,
  DatasetRegistryResponse,
  ClipperInfo,
  ClippersResponse,
  ConverterInfo,
  ConvertersResponse,
  EmbedderInfo,
  EmbeddersResponse,
  OkResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class DatasetsApiService {
  constructor(private http: HttpClient) {}

  getStatus(): Observable<DatasetStatus> {
    return this.http.get<DatasetStatus>('/api/dataset/status');
  }

  getImporters(): Observable<ImportersResponse> {
    return this.http.get<ImportersResponse>('/api/dataset/importers');
  }

  getAllImporters(): Observable<ImportersResponse> {
    return this.http.get<ImportersResponse>('/api/dataset/all-importers');
  }

  getDemoList(embedder?: string, clipper?: string): Observable<DemoListResponse> {
    const params: Record<string, string> = {};
    if (embedder) {
      params['embedder'] = embedder;
    }
    if (clipper) {
      params['clipper'] = clipper;
    }
    return this.http.get<DemoListResponse>('/api/dataset/demo-list', { params });
  }

  getDemoCategories(name: string): Observable<{ categories: string[] }> {
    return this.http.get<{ categories: string[] }>(`/api/dataset/demo-categories/${encodeURIComponent(name)}`);
  }

  browseMediaFiles(
    source: string,
    path: string,
  ): Observable<{
    directories: { name: string; path: string; modified_at?: string }[];
    files: { name: string; path: string; size_bytes: number; modified_at?: string }[];
    root_path: string;
  }> {
    return this.http.get<{
      directories: { name: string; path: string; modified_at?: string }[];
      files: { name: string; path: string; size_bytes: number; modified_at?: string }[];
      root_path: string;
    }>('/api/browse-media-files', { params: { source, path } });
  }

  selectBrowsedFile(
    source: string,
    path: string,
  ): Observable<{ filename: string; original_name: string }> {
    return this.http.post<{ filename: string; original_name: string }>(
      '/api/browse-media-files/select',
      { source, path },
    );
  }

  getMediaTypes(): Observable<MediaTypesResponse> {
    return this.http.get<MediaTypesResponse>('/api/media-types');
  }

  detectMediaType(
    source: string,
    path: string,
    recursive: boolean,
    limit = 50,
  ): Observable<MediaTypeDetectionResponse> {
    return this.http.get<MediaTypeDetectionResponse>('/api/dataset/detect-media-type', {
      params: {
        source,
        path,
        recursive: recursive ? 'true' : 'false',
        limit: String(limit),
      },
    });
  }

  getClippers(mediaType?: string): Observable<ClipperInfo[]> {
    const params: Record<string, string> = {};
    if (mediaType) {
      params['media_type'] = mediaType;
    }
    return this.http.get<ClippersResponse>('/api/clippers', { params }).pipe(
      map((res) => res.clippers),
    );
  }

  getEmbedders(mediaType?: string): Observable<EmbedderInfo[]> {
    const params: Record<string, string> = {};
    if (mediaType) {
      params['media_type'] = mediaType;
    }
    return this.http.get<EmbeddersResponse>('/api/embedders', { params }).pipe(
      map((res) => res.embedders),
    );
  }

  getConverters(target?: string): Observable<ConverterInfo[]> {
    const params: Record<string, string> = {};
    if (target) {
      params['target'] = target;
    }
    return this.http.get<ConvertersResponse>('/api/converters', { params }).pipe(
      map((res) => res.converters),
    );
  }

  getAvailableFiles(): Observable<{ files: { name: string; path: string; size_mb: number }[] }> {
    return this.http.get<{ files: { name: string; path: string; size_mb: number }[] }>('/api/dataset/available-files');
  }

  runImporter(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/import/${encodeURIComponent(importerName)}`, params);
  }

  /**
   * Fetch dropdown options for an importer field whose options are computed
   * at runtime by the importer (``dynamic_options=true``).
   *
   * The backend calls the importer's ``get_field_options(field_key, values)``
   * Python method and returns the resulting list.  Errors from the remote
   * service are surfaced as HTTP errors with an ``error`` message body.
   */
  getImporterFieldOptions(
    importerName: string,
    fieldKey: string,
    values: Record<string, unknown>,
  ): Observable<{ options: string[] }> {
    return this.http.post<{ options: string[] }>(
      `/api/dataset/import/${encodeURIComponent(importerName)}/options`,
      { field_key: fieldKey, values },
    );
  }

  /**
   * Upload a folder selected from the user's *browser* machine.
   *
   * The caller is responsible for building the FormData with the files
   * (each appended under the key ``"files"`` with their
   * ``webkitRelativePath`` as the multipart filename) plus ``media_type``
   * and the optional ``embedder`` / ``clipper`` / ``clipper_params``
   * fields.
   */
  importLocalFolder(formData: FormData): Observable<unknown> {
    return this.http.post('/api/dataset/import-local-folder', formData);
  }

  loadDemo(name: string, params?: Record<string, string>): Observable<unknown> {
    return this.http.post('/api/dataset/load-demo', { name, ...params });
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
    return this.http.post(`/api/dataset/stage-import/${encodeURIComponent(importerName)}`, params);
  }

  stageDemo(name: string): Observable<unknown> {
    return this.http.post(`/api/dataset/stage-demo/${encodeURIComponent(name)}`, {});
  }

  clearStaging(): Observable<unknown> {
    return this.http.delete('/api/dataset/staging');
  }

  combineDatasets(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/dataset/combine', params);
  }

  cancelIngest(): Observable<OkResponse> {
    return this.http.post<OkResponse>('/api/dataset/cancel', {});
  }

  cancelTask(taskId: string): Observable<OkResponse> {
    return this.http.post<OkResponse>(`/api/dataset/cancel/${encodeURIComponent(taskId)}`, {});
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
    return this.http.post(`/api/datasets/registry/${encodeURIComponent(datasetId)}/load`, {});
  }

  unloadRegistered(datasetId: string): Observable<unknown> {
    return this.http.post(`/api/datasets/registry/${encodeURIComponent(datasetId)}/unload`, {});
  }

  deleteRegistered(datasetId: string): Observable<unknown> {
    return this.http.delete(`/api/datasets/registry/${encodeURIComponent(datasetId)}`);
  }

  renameRegistered(datasetId: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/datasets/registry/${encodeURIComponent(datasetId)}/rename`, { name: newName });
  }

  updateReaders(datasetId: string, readers: string[]): Observable<unknown> {
    return this.http.put(`/api/datasets/registry/${encodeURIComponent(datasetId)}/readers`, { readers });
  }

  loadSource(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/dataset/load-source', params);
  }

  getDatasetStats(datasetId: string): Observable<DatasetStatsResponse> {
    return this.http.get<DatasetStatsResponse>(`/api/datasets/registry/${encodeURIComponent(datasetId)}/stats`);
  }

  getDiskUsage(): Observable<{ total: number; used: number; free: number; path: string }> {
    return this.http.get<{ total: number; used: number; free: number; path: string }>(
      '/api/dashboard/disk-usage',
    );
  }
}
