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

  runImport(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/label-importers/import/${importerName}`, params);
  }

  ingestMissing(): Observable<unknown> {
    return this.http.post('/api/label-importers/ingest-missing', {});
  }
}
