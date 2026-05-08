import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { LabelImporterInfo } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class LabelImportersApiService {
  constructor(private http: HttpClient) {}

  list(): Observable<LabelImporterInfo[]> {
    return this.http.get<LabelImporterInfo[]>('/api/label-importers');
  }

  runImport(importerName: string, params: Record<string, unknown>, file?: File, fileFieldKey?: string): Observable<unknown> {
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post(`/api/label-importers/import/${importerName}`, formData);
    }
    return this.http.post(`/api/label-importers/import/${importerName}`, params);
  }

  runModelImport(
    modelName: string,
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown> {
    const url = `/api/detectors/${encodeURIComponent(modelName)}/import-labels/${importerName}`;
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

  ingestMissing(entries: unknown[]): Observable<unknown> {
    return this.http.post('/api/label-importers/ingest-missing', { entries });
  }
}
