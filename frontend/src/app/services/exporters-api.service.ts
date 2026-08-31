import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { ExporterEntry } from '../generated/api-client/models/exporter-entry';
import type { FieldOptions } from '../generated/api-client/models/field-options';
import type { RunExportRequest } from '../generated/api-client/models/run-export-request';
import type { RunExportResponse } from '../generated/api-client/models/run-export-response';
import { exporterFieldOptions } from '../generated/api-client/fn/exporters/exporter-field-options';
import { getExporters } from '../generated/api-client/fn/exporters/get-exporters';
import { runExport } from '../generated/api-client/fn/exporters/run-export';

@Injectable({ providedIn: 'root' })
export class ExportersApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getExporters(): Observable<ExporterEntry[]> {
    return getExporters(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Current option list for one of *exporterName*'s ``dynamic_options``
   *  select fields, given a snapshot of the form's current values. */
  getFieldOptions(
    exporterName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }> {
    return exporterFieldOptions(this.http, this.config.rootUrl, {
      exporter_name: exporterName,
      body: { field_key: fieldKey, values },
    }).pipe(map((r) => ({ options: r.body.options ?? [] })));
  }

  runExport(body: RunExportRequest): Observable<RunExportResponse> {
    return runExport(this.http, this.config.rootUrl, { body }).pipe(
      map((r) => r.body),
    );
  }
}
