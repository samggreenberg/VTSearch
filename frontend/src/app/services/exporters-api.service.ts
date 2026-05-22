import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { ExporterEntry } from '../generated/api-client/models/exporter-entry';
import type { RunExportRequest } from '../generated/api-client/models/run-export-request';
import type { RunExportResponse } from '../generated/api-client/models/run-export-response';
import { getExporters } from '../generated/api-client/fn/exporters/get-exporters';
import { runExport } from '../generated/api-client/fn/exporters/run-export';

@Injectable({ providedIn: 'root' })
export class ExportersApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getExporters(): Observable<ExporterEntry[]> {
    return getExporters(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  runExport(body: RunExportRequest): Observable<RunExportResponse> {
    return runExport(this.http, this.config.rootUrl, { body }).pipe(
      map((r) => r.body),
    );
  }
}
