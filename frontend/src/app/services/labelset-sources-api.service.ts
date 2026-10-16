import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DefaultLabelsetSource } from '../generated/api-client/models/default-labelset-source';
import type { DetectorLabelsetSource } from '../generated/api-client/models/detector-labelset-source';
import type { OkMessage } from '../generated/api-client/models/ok-message';
import type { SyncSourceEntry } from '../generated/api-client/models/sync-source-entry';
import { getDefaultLabelsetSource } from '../generated/api-client/fn/sync-sources/get-default-labelset-source';
import { getDetectorLabelsetSource } from '../generated/api-client/fn/sync-sources/get-detector-labelset-source';
import { listLabelsetSourcesRoute } from '../generated/api-client/fn/sync-sources/list-labelset-sources-route';
import { setDetectorLabelsetSource } from '../generated/api-client/fn/sync-sources/set-detector-labelset-source';

/**
 * Where a detector's labels are kept current outside VTSearch (#4913).
 *
 * A *labelset source* is a sync target a detector's whole labelset is pushed
 * to as it changes. The deployment may name a default every detector follows
 * ({@link getDefault}); the admin fixes some of its field values and users
 * fill in the rest, which is what the New Detector form and the Dashboard's
 * "Auto-export…" action collect and {@link setDetectorSource} stores on the
 * detector.
 */
@Injectable({ providedIn: 'root' })
export class LabelsetSourcesApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /** The deployment-wide source, with only the fields a user still fills in;
   *  `null` when the deployment has none. */
  getDefault(): Observable<DefaultLabelsetSource | null> {
    return getDefaultLabelsetSource(this.http, this.config.rootUrl).pipe(map((r) => r.body ?? null));
  }

  /** Every registered labelset source plugin, with all its fields. */
  listSources(): Observable<SyncSourceEntry[]> {
    return listLabelsetSourcesRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body ?? []));
  }

  /** The source *detectorId* syncs with, or `null` when it syncs nowhere. */
  getDetectorSource(detectorId: string): Observable<DetectorLabelsetSource | null> {
    return getDetectorLabelsetSource(this.http, this.config.rootUrl, { detector_ref: detectorId }).pipe(
      map((r) => r.body ?? null),
    );
  }

  /** Link *detectorId* to *sourceName* with its own field values. The server
   *  validates them, then levels the detector with what the source already
   *  holds. */
  setDetectorSource(
    detectorId: string,
    sourceName: string,
    fieldValues: Record<string, string>,
  ): Observable<OkMessage> {
    return setDetectorLabelsetSource(this.http, this.config.rootUrl, {
      detector_ref: detectorId,
      body: { source_name: sourceName, field_values: fieldValues },
    }).pipe(map((r) => r.body));
  }
}
