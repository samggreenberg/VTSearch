import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { FindCancelResponse } from '../generated/api-client/models/find-cancel-response';
import type { FindCheckLabelsRequest } from '../generated/api-client/models/find-check-labels-request';
import type { FindCheckLabelsResponse } from '../generated/api-client/models/find-check-labels-response';
import type { FindLabelRequest } from '../generated/api-client/models/find-label-request';
import type { FindLabelResponse } from '../generated/api-client/models/find-label-response';
import type { FindRequest } from '../generated/api-client/models/find-request';
import type { FindResponse } from '../generated/api-client/models/find-response';
import type { FindStatsResponse } from '../generated/api-client/models/find-stats-response';
import { cancelFind } from '../generated/api-client/fn/detector-find/cancel-find';
import { findCheckLabels } from '../generated/api-client/fn/detector-find/find-check-labels';
import { findLabel } from '../generated/api-client/fn/detector-scoring/find-label';
import { findStats } from '../generated/api-client/fn/detector-scoring/find-stats';
import { multiFind } from '../generated/api-client/fn/detector-find/multi-find';

/** Multi-dataset Find, single-label Find, the check-labels precheck, and
 *  Find cancellation. */
@Injectable({ providedIn: 'root' })
export class DetectorsFindApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  findCheckLabels(params: FindCheckLabelsRequest): Observable<FindCheckLabelsResponse> {
    return findCheckLabels(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  find(params: FindRequest): Observable<FindResponse> {
    return multiFind(this.http, this.config.rootUrl, { body: params }).pipe(map((r) => r.body));
  }

  findLabel(params: FindLabelRequest): Observable<FindLabelResponse> {
    return findLabel(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Cancel any in-flight find / find-label / auto-detect scoring. */
  cancelFind(): Observable<FindCancelResponse> {
    return cancelFind(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Detector-evaluation stats over the adopted Find label set (2x2 confusion
   *  + the FP/FN-vs-inclusion sweep). Pure read. */
  getFindStats(): Observable<FindStatsResponse> {
    return findStats(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
