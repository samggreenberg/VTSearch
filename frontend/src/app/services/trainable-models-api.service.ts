import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  CombineModelsResult,
  LabelsDetailResponse,
  LoadingTasksResponse,
  TrainableModelsResponse,
  ModelsRegistryResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class TrainableModelsApiService {
  constructor(private http: HttpClient) {}

  list(): Observable<TrainableModelsResponse> {
    return this.http.get<TrainableModelsResponse>('/api/trainable-models');
  }

  create(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/trainable-models', params);
  }

  get(name: string): Observable<unknown> {
    return this.http.get(`/api/trainable-models/${name}`);
  }

  delete(name: string): Observable<unknown> {
    return this.http.delete(`/api/trainable-models/${name}`);
  }

  rename(name: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/trainable-models/${name}/rename`, { new_name: newName });
  }

  setExamples(name: string, examples: unknown[]): Observable<unknown> {
    return this.http.put(`/api/trainable-models/${name}/examples`, { examples });
  }

  saveLabels(name: string): Observable<unknown> {
    return this.http.post(`/api/trainable-models/${name}/labels`, {});
  }

  getLabelsDetail(name: string): Observable<LabelsDetailResponse> {
    return this.http.get<LabelsDetailResponse>(
      `/api/trainable-models/${encodeURIComponent(name)}/labels-detail`,
    );
  }

  voteLabelElement(name: string, elementId: string, vote: 'good' | 'bad'): Observable<unknown> {
    return this.http.post(
      `/api/trainable-models/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/vote`,
      { vote },
    );
  }

  labelPreviewUrl(name: string, elementId: string): string {
    return `/api/trainable-models/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/preview`;
  }

  combine(
    names: string[],
    newName: string,
    conflictPolicy: 'drop' = 'drop',
  ): Observable<CombineModelsResult> {
    return this.http.post<CombineModelsResult>('/api/trainable-models/combine', {
      names,
      new_name: newName,
      conflict_policy: conflictPolicy,
    });
  }

  // --- Models Registry ---

  getRegistry(): Observable<ModelsRegistryResponse> {
    return this.http.get<ModelsRegistryResponse>('/api/models/registry');
  }

  registerModel(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/models/registry', params);
  }

  registerModelFromLabelset(
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown> {
    const url = `/api/models/registry/from-labelset/${importerName}`;
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

  deleteFromRegistry(modelId: string): Observable<unknown> {
    return this.http.delete(`/api/models/registry/${modelId}`);
  }

  renameInRegistry(modelId: string, newName: string): Observable<unknown> {
    return this.http.put(`/api/models/registry/${modelId}/rename`, { name: newName });
  }

  loadModel(modelId: string | null): Observable<unknown> {
    return this.http.post('/api/models/registry/load', { model_id: modelId });
  }

  unloadModel(modelId: string): Observable<unknown> {
    return this.http.post(`/api/models/registry/${modelId}/unload`, {});
  }

  getModelLoadingTasks(): Observable<LoadingTasksResponse> {
    return this.http.get<LoadingTasksResponse>('/api/models/loading-tasks');
  }

  cancelModelLoadingTask(taskId: string): Observable<unknown> {
    return this.http.post(`/api/models/cancel/${taskId}`, {});
  }
}
