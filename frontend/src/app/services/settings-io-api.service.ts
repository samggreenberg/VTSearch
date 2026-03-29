import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ImporterInfo, ExporterInfo } from '../models/api.models';

export interface SettingsImportResponse {
  success: boolean;
  message: string;
  keys?: string[];
}

export interface SettingsExportResponse {
  success: boolean;
  message: string;
  download?: boolean;
  data?: Record<string, unknown>;
  filename?: string;
  filepath?: string;
}

@Injectable({ providedIn: 'root' })
export class SettingsIoApiService {
  constructor(private http: HttpClient) {}

  listImporters(): Observable<ImporterInfo[]> {
    return this.http.get<ImporterInfo[]>('/api/settings-importers');
  }

  runImport(
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<SettingsImportResponse> {
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post<SettingsImportResponse>(
        `/api/settings-importers/import/${importerName}`,
        formData,
      );
    }
    return this.http.post<SettingsImportResponse>(
      `/api/settings-importers/import/${importerName}`,
      params,
    );
  }

  listExporters(): Observable<ExporterInfo[]> {
    return this.http.get<ExporterInfo[]>('/api/settings-exporters');
  }

  runExport(
    exporterName: string,
    fieldValues: Record<string, unknown>,
  ): Observable<SettingsExportResponse> {
    return this.http.post<SettingsExportResponse>('/api/settings-exporters/export', {
      exporter_name: exporterName,
      field_values: fieldValues,
    });
  }
}
