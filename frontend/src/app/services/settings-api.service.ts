import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AppSettings, AutorunProcessor, EmbeddersResponse } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  constructor(private http: HttpClient) {}

  getSettings(): Observable<AppSettings> {
    return this.http.get<AppSettings>('/api/settings');
  }

  updateSettings(data: Partial<AppSettings>): Observable<AppSettings> {
    return this.http.put<AppSettings>('/api/settings', data);
  }

  getDefaults(): Observable<AppSettings> {
    return this.http.get<AppSettings>('/api/settings/defaults');
  }

  getAutorunProcessors(): Observable<{ autorun_processors: AutorunProcessor[] }> {
    return this.http.get<{ autorun_processors: AutorunProcessor[] }>('/api/settings/autorun-processors');
  }

  addAutorunProcessor(processor: { processor_name: string; processor_importer: string; field_values?: Record<string, unknown> }): Observable<unknown> {
    return this.http.post('/api/settings/autorun-processors', processor);
  }

  deleteAutorunProcessor(name: string): Observable<unknown> {
    return this.http.delete(`/api/settings/autorun-processors/${name}`);
  }

  getEmbedders(): Observable<EmbeddersResponse> {
    return this.http.get<EmbeddersResponse>('/api/embedders');
  }
}
