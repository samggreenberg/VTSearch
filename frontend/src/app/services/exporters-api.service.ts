import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { ExporterEntry } from '../generated/api-client/models/exporter-entry';
import type { RunExportRequest } from '../generated/api-client/models/run-export-request';
import type { RunExportResponse } from '../generated/api-client/models/run-export-response';
import { apiExportersGet } from '../generated/api-client/fn/exporters/api-exporters-get';
import { apiExportersExportPost } from '../generated/api-client/fn/exporters/api-exporters-export-post';

@Injectable({ providedIn: 'root' })
export class ExportersApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getExporters(): Observable<ExporterEntry[]> {
    return apiExportersGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  runExport(body: RunExportRequest): Observable<RunExportResponse> {
    return apiExportersExportPost(this.http, this.config.rootUrl, { body }).pipe(
      map((r) => r.body),
    );
  }
}
