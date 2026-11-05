import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DetectorForkResponse } from '../generated/api-client/models/detector-fork-response';
import type { DetectorSharingCapability } from '../generated/api-client/models/detector-sharing-capability';
import type { DetectorSharingRefreshRequest } from '../generated/api-client/models/detector-sharing-refresh-request';
import type { DetectorSharingRefreshResponse } from '../generated/api-client/models/detector-sharing-refresh-response';
import type { DetectorSharingView } from '../generated/api-client/models/detector-sharing-view';
import { forkRegisteredDetector } from '../generated/api-client/fn/detectors-sharing/fork-registered-detector';
import { getDetectorSharing } from '../generated/api-client/fn/detectors-sharing/get-detector-sharing';
import { getSharingCapability } from '../generated/api-client/fn/detectors-sharing/get-sharing-capability';
import { refreshDetector } from '../generated/api-client/fn/detectors-sharing/refresh-detector';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

/** What a container's foreign records get on a receive or refresh. */
export type ForeignPolicy = NonNullable<DetectorSharingRefreshRequest['foreign']>;

/** Detector sharing (`vtsearch/routes/detectors/sharing.py`): the capability,
 *  a detector's container, Check for updates and Fork. */
@Injectable({ providedIn: 'root' })
export class DetectorsSharingApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /** Never toasts: a server without sharing simply says so, and the caller treats a failure the same way. */
  getCapability(): Observable<DetectorSharingCapability> {
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return getSharingCapability(this.http, this.config.rootUrl, undefined, context).pipe(map((r) => r.body));
  }

  /** Never toasts: the Shared badge's title asks on hover, and goes without on failure. */
  getSharingView(detectorId: string): Observable<DetectorSharingView> {
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return getDetectorSharing(this.http, this.config.rootUrl, { detector_id: detectorId }, context).pipe(
      map((r) => r.body),
    );
  }

  refresh(detectorId: string, foreign: ForeignPolicy): Observable<DetectorSharingRefreshResponse> {
    return refreshDetector(this.http, this.config.rootUrl, { detector_id: detectorId, body: { foreign } }).pipe(
      map((r) => r.body),
    );
  }

  fork(detectorId: string, name?: string): Observable<DetectorForkResponse> {
    return forkRegisteredDetector(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: name ? { name } : {},
    }).pipe(map((r) => r.body));
  }
}
