import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AutoDetectRequest } from '../generated/api-client/models/auto-detect-request';
import type { AutoDetectResponse } from '../generated/api-client/models/auto-detect-response';
import type { AutoExtractResponse } from '../generated/api-client/models/auto-extract-response';
import type { AutoLocalizeResponse } from '../generated/api-client/models/auto-localize-response';
import type { ExtractRequest } from '../generated/api-client/models/extract-request';
import type { ExtractResponse } from '../generated/api-client/models/extract-response';
import type { LocalizeRequest } from '../generated/api-client/models/localize-request';
import type { LocalizeResponse } from '../generated/api-client/models/localize-response';
import { autoDetect } from '../generated/api-client/fn/detector-scoring/auto-detect';
import { autoExtract } from '../generated/api-client/fn/processors-scoring/auto-extract';
import { autoLocalize } from '../generated/api-client/fn/processors-scoring/auto-localize';
import { runExtract } from '../generated/api-client/fn/processors-scoring/run-extract';
import { runLocalize } from '../generated/api-client/fn/processors-scoring/run-localize';

/** Detector-driven scoring: auto-detect plus the extractor / localizer
 *  run-once endpoints. */
@Injectable({ providedIn: 'root' })
export class DetectorsScoringApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  autoDetect(params: AutoDetectRequest): Observable<AutoDetectResponse> {
    return autoDetect(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  extract(params: ExtractRequest): Observable<ExtractResponse> {
    return runExtract(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoExtract(): Observable<AutoExtractResponse> {
    return autoExtract(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  localize(params: LocalizeRequest): Observable<LocalizeResponse> {
    return runLocalize(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoLocalize(): Observable<AutoLocalizeResponse> {
    return autoLocalize(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
