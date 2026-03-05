import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { TrainableModelsResponse, ModelsRegistryResponse } from '../models/api.models';

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
    return this.http.put(`/api/trainable-models/${name}/rename`, { name: newName });
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
}
