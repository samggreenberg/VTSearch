import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DetectorForkResponse } from '../generated/api-client/models/detector-fork-response';
import type { DetectorShareRequest } from '../generated/api-client/models/detector-share-request';
import type { DetectorShareResponse } from '../generated/api-client/models/detector-share-response';
import type { DetectorSharingCapability } from '../generated/api-client/models/detector-sharing-capability';
import type { DetectorSharingReceiveResponse } from '../generated/api-client/models/detector-sharing-receive-response';
import type { DetectorSharingRecipient } from '../generated/api-client/models/detector-sharing-recipient';
import type { DetectorSharingRefreshRequest } from '../generated/api-client/models/detector-sharing-refresh-request';
import type { DetectorSharingRefreshResponse } from '../generated/api-client/models/detector-sharing-refresh-response';
import type { DetectorSharingSummary } from '../generated/api-client/models/detector-sharing-summary';
import type { DetectorSharingView } from '../generated/api-client/models/detector-sharing-view';
import type { DetectorUnshareResponse } from '../generated/api-client/models/detector-unshare-response';
import { forkRegisteredDetector } from '../generated/api-client/fn/detectors-sharing/fork-registered-detector';
import { getDetectorSharing } from '../generated/api-client/fn/detectors-sharing/get-detector-sharing';
import { getSharingCapability } from '../generated/api-client/fn/detectors-sharing/get-sharing-capability';
import { listReceivedDetectors } from '../generated/api-client/fn/detectors-sharing/list-received-detectors';
import { receiveSharedDetector } from '../generated/api-client/fn/detectors-sharing/receive-shared-detector';
import { refreshDetector } from '../generated/api-client/fn/detectors-sharing/refresh-detector';
import { searchSharingRecipients } from '../generated/api-client/fn/detectors-sharing/search-sharing-recipients';
import { shareDetector } from '../generated/api-client/fn/detectors-sharing/share-detector';
import { unshareDetector } from '../generated/api-client/fn/detectors-sharing/unshare-detector';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

/** What a container's foreign records get on a receive or refresh. */
export type ForeignPolicy = NonNullable<DetectorSharingRefreshRequest['foreign']>;

/** Detector sharing (`vtsearch/routes/detectors/sharing.py`): the capability,
 *  a detector's container, sharing it and who with, the containers shared with
 *  the user and receiving one, Check for updates and Fork. */
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

  /** Never toasts: the Received list shows the failure in place of the rows. */
  listReceived(): Observable<DetectorSharingSummary[]> {
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return listReceivedDetectors(this.http, this.config.rootUrl, undefined, context).pipe(
      map((r) => r.body.received),
    );
  }

  /** `applied: false` (a 200) is the `abort` policy meeting foreign records, not a failure. */
  receive(remoteRef: string, foreign: ForeignPolicy): Observable<DetectorSharingReceiveResponse> {
    return receiveSharedDetector(this.http, this.config.rootUrl, { body: { remote_ref: remoteRef, foreign } }).pipe(
      map((r) => r.body),
    );
  }

  /** Never toasts: it runs as the user types, and no suggestions is its failure. */
  searchRecipients(query: string): Observable<DetectorSharingRecipient[]> {
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return searchSharingRecipients(this.http, this.config.rootUrl, { body: { query } }, context).pipe(
      map((r) => r.body.suggestions),
    );
  }

  /** The first call creates the detector's container; `{}` only does that. */
  share(detectorId: string, body: DetectorShareRequest): Observable<DetectorShareResponse> {
    return shareDetector(this.http, this.config.rootUrl, { detector_id: detectorId, body }).pipe(map((r) => r.body));
  }

  unshare(detectorId: string): Observable<DetectorUnshareResponse> {
    return unshareDetector(this.http, this.config.rootUrl, { detector_id: detectorId }).pipe(map((r) => r.body));
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
