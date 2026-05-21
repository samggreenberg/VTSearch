import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { LabelImporterEntry } from '../generated/api-client/models/label-importer-entry';
import type { IngestMissingResponse } from '../generated/api-client/models/ingest-missing-response';
import { getLabelImporters } from '../generated/api-client/fn/label-importers/get-label-importers';
import { ingestMissing } from '../generated/api-client/fn/label-importers/ingest-missing';

@Injectable({ providedIn: 'root' })
export class LabelImportersApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  list(): Observable<LabelImporterEntry[]> {
    return getLabelImporters(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Plugin-field route — request body shape is plugin-dependent and not
   *  described in the OpenAPI spec, so this stays on plain HttpClient (same
   *  pattern as ``SettingsIoApiService.runImport``). See
   *  ``docs/plans/openapi-schema.md`` § Resolved questions / Plugin field
   *  endpoints. */
  runImport(importerName: string, params: Record<string, unknown>, file?: File, fileFieldKey?: string): Observable<unknown> {
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post(`/api/label-importers/import/${encodeURIComponent(importerName)}`, formData);
    }
    return this.http.post(`/api/label-importers/import/${encodeURIComponent(importerName)}`, params);
  }

  /** Plugin-field route — see ``runImport`` above. */
  runModelImport(
    modelName: string,
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown> {
    const url = `/api/detectors/${encodeURIComponent(modelName)}/import-labels/${encodeURIComponent(importerName)}`;
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

  ingestMissing(entries: Record<string, unknown>[]): Observable<IngestMissingResponse> {
    return ingestMissing(this.http, this.config.rootUrl, {
      body: { entries },
    }).pipe(map((r) => r.body));
  }
}
