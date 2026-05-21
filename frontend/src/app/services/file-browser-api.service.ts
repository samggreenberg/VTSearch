import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { BrowseResponse } from '../generated/api-client/models/browse-response';
import { browse } from '../generated/api-client/fn/file-browser/browse';

@Injectable({ providedIn: 'root' })
export class FileBrowserApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  browse(path: string, extensions?: string): Observable<BrowseResponse> {
    return browse(this.http, this.config.rootUrl, { path, extensions }).pipe(
      map((r) => r.body),
    );
  }
}
