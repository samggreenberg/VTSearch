import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { FieldOptions } from '../generated/api-client/models/field-options';
import type { SettingsImporterEntry } from '../generated/api-client/models/settings-importer-entry';
import type { SettingsExporterEntry } from '../generated/api-client/models/settings-exporter-entry';
import type { RunSettingsExportRequest } from '../generated/api-client/models/run-settings-export-request';
import type { RunSettingsExportResponse } from '../generated/api-client/models/run-settings-export-response';
import { getSettingsImporters } from '../generated/api-client/fn/settings-io/get-settings-importers';
import { getSettingsExporters } from '../generated/api-client/fn/settings-io/get-settings-exporters';
import { runSettingsExport } from '../generated/api-client/fn/settings-io/run-settings-export';
import { settingsImporterFieldOptions } from '../generated/api-client/fn/settings-io/settings-importer-field-options';
import { settingsExporterFieldOptions } from '../generated/api-client/fn/settings-io/settings-exporter-field-options';

/** Response shape for the plugin-field import route. The body shape is
 *  plugin-dependent and not described in the OpenAPI spec (the spec
 *  declares it as just an Error response), so we keep a local interface
 *  matching what the backend actually returns. */
export interface SettingsImportResponse {
  success: boolean;
  message: string;
  keys?: string[];
}

@Injectable({ providedIn: 'root' })
export class SettingsIoApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  listImporters(): Observable<SettingsImporterEntry[]> {
    return getSettingsImporters(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  runImport(
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<SettingsImportResponse> {
    // Plugin-field route: body shape is plugin-dependent and not
    // described in the OpenAPI spec (see plan "Open follow-ups /
    // Per-plugin schemas"), so this stays on plain HttpClient.
    const url = `/api/settings-importers/import/${encodeURIComponent(importerName)}`;
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post<SettingsImportResponse>(url, formData);
    }
    return this.http.post<SettingsImportResponse>(url, params);
  }

  /** Current option list for one of *importerName*'s ``dynamic_options``
   *  select fields, given a snapshot of the form's current values. */
  getImporterFieldOptions(
    importerName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }> {
    return settingsImporterFieldOptions(this.http, this.config.rootUrl, {
      importer_name: importerName,
      body: { field_key: fieldKey, values },
    }).pipe(map((r) => ({ options: r.body.options ?? [] })));
  }

  listExporters(): Observable<SettingsExporterEntry[]> {
    return getSettingsExporters(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Current option list for one of *exporterName*'s ``dynamic_options``
   *  select fields, given a snapshot of the form's current values. */
  getExporterFieldOptions(
    exporterName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }> {
    return settingsExporterFieldOptions(this.http, this.config.rootUrl, {
      exporter_name: exporterName,
      body: { field_key: fieldKey, values },
    }).pipe(map((r) => ({ options: r.body.options ?? [] })));
  }

  runExport(
    exporterName: string,
    fieldValues: Record<string, unknown>,
  ): Observable<RunSettingsExportResponse> {
    const body: RunSettingsExportRequest = {
      exporter_name: exporterName,
      field_values: fieldValues as { [key: string]: unknown },
    };
    return runSettingsExport(this.http, this.config.rootUrl, { body }).pipe(
      map((r) => r.body),
    );
  }
}
