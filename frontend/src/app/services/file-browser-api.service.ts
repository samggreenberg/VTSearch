import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { BrowseResponse } from '../generated/api-client/models/browse-response';
import { apiBrowseGet } from '../generated/api-client/fn/file-browser/api-browse-get';

@Injectable({ providedIn: 'root' })
export class FileBrowserApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  browse(path: string, extensions?: string): Observable<BrowseResponse> {
    return apiBrowseGet(this.http, this.config.rootUrl, { path, extensions }).pipe(
      map((r) => r.body),
    );
  }
}
