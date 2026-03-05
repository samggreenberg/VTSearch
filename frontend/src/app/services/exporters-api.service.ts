import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ExporterInfo } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ExportersApiService {
  constructor(private http: HttpClient) {}

  getExporters(): Observable<ExporterInfo[]> {
    return this.http.get<ExporterInfo[]>('/api/exporters');
  }

  runExport(params: { exporter_name: string; [key: string]: unknown }): Observable<unknown> {
    return this.http.post('/api/exporters/export', params);
  }
}
