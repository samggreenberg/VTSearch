import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { LoadingTasksResponse, TrainableModelsResponse, ModelsRegistryResponse } from '../models/api.models';

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

  // --- Models Registry ---

  getRegistry(): Observable<ModelsRegistryResponse> {
    return this.http.get<ModelsRegistryResponse>('/api/models/registry');
  }

  registerModel(params: Record<string, unknown>): Observable<unknown> {
    return this.http.post('/api/models/registry', params);
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

  activateModel(modelId: string): Observable<unknown> {
    return this.http.post(`/api/models/registry/${modelId}/activate`, {});
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
