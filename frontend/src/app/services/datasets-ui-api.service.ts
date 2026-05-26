import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { BrowseMediaFilesResponse } from '../generated/api-client/models/browse-media-files-response';
import type { BrowseMediaFilesSelectResponse } from '../generated/api-client/models/browse-media-files-select-response';
import type { DashboardDiskUsageResponse } from '../generated/api-client/models/dashboard-disk-usage-response';
import type { DashboardRamUsageResponse } from '../generated/api-client/models/dashboard-ram-usage-response';
import { browseMediaFiles } from '../generated/api-client/fn/datasets-ui/browse-media-files';
import { dashboardDiskUsage } from '../generated/api-client/fn/datasets-ui/dashboard-disk-usage';
import { dashboardRamUsage } from '../generated/api-client/fn/datasets-ui/dashboard-ram-usage';
import { selectBrowsedFile } from '../generated/api-client/fn/datasets-ui/select-browsed-file';

/** Dashboard-only UI helpers: server-side file browsing for the import /
 *  load-sort modals, and the dashboard disk / RAM usage strip. */
@Injectable({ providedIn: 'root' })
export class DatasetsUiApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  browseMediaFiles(source: string, path: string): Observable<BrowseMediaFilesResponse> {
    return browseMediaFiles(this.http, this.config.rootUrl, { source, path }).pipe(
      map((r) => r.body),
    );
  }

  selectBrowsedFile(source: string, path: string): Observable<BrowseMediaFilesSelectResponse> {
    return selectBrowsedFile(this.http, this.config.rootUrl, {
      body: { source, path },
    }).pipe(map((r) => r.body));
  }

  getDiskUsage(): Observable<DashboardDiskUsageResponse> {
    return dashboardDiskUsage(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getRamUsage(): Observable<DashboardRamUsageResponse> {
    return dashboardRamUsage(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
