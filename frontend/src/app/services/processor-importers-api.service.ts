import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ProcessorImporterInfo } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ProcessorImportersApiService {
  constructor(private http: HttpClient) {}

  list(): Observable<ProcessorImporterInfo[]> {
    return this.http.get<ProcessorImporterInfo[]>('/api/processor-importers');
  }

  runImport(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/processor-importers/import/${encodeURIComponent(importerName)}`, params);
  }
}
