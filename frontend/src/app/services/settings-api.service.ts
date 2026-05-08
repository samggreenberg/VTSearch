import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AppSettings, EmbeddersResponse } from '../models/api.models';

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

  getEmbedders(): Observable<EmbeddersResponse> {
    return this.http.get<EmbeddersResponse>('/api/embedders');
  }

  getVersion(): Observable<{ version: string }> {
    return this.http.get<{ version: string }>('/api/version');
  }
}
