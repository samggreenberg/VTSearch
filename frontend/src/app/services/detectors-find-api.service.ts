import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AutoFindBrowsePrepResponse } from '../generated/api-client/models/auto-find-browse-prep-response';
import type { AutoFindRunResponse } from '../generated/api-client/models/auto-find-run-response';
import type { FindCancelResponse } from '../generated/api-client/models/find-cancel-response';
import type { FindEndSessionResponse } from '../generated/api-client/models/find-end-session-response';
import type { FindLabelRequest } from '../generated/api-client/models/find-label-request';
import type { FindLabelResponse } from '../generated/api-client/models/find-label-response';
import type { FindStatsResponse } from '../generated/api-client/models/find-stats-response';
import type { FindEvidenceCoverageResponse } from '../generated/api-client/models/find-evidence-coverage-response';
import type { FindCorrectionsToDetectorResponse } from '../generated/api-client/models/find-corrections-to-detector-response';
import { cancelFind } from '../generated/api-client/fn/detector-find/cancel-find';
import { endFindSessionRoute } from '../generated/api-client/fn/detector-find/end-find-session-route';
import { findLabel } from '../generated/api-client/fn/detector-scoring/find-label';
import { findStats } from '../generated/api-client/fn/detector-scoring/find-stats';
import { findEvidenceCoverage } from '../generated/api-client/fn/detector-scoring/find-evidence-coverage';
import { findCorrectionsToDetector } from '../generated/api-client/fn/detector-scoring/find-corrections-to-detector';
import { getAutofindRun } from '../generated/api-client/fn/detector-scoring/get-autofind-run';
import { prepAutofindRunBrowse } from '../generated/api-client/fn/detector-scoring/prep-autofind-run-browse';

/** Single-label Find, its session / stats / evidence reads, Find
 *  cancellation, and the results of a background AutoFind. */
@Injectable({ providedIn: 'root' })
export class DetectorsFindApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  findLabel(params: FindLabelRequest): Observable<FindLabelResponse> {
    return findLabel(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Cancel any in-flight find / find-label / auto-detect scoring. */
  cancelFind(): Observable<FindCancelResponse> {
    return cancelFind(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Discard the active detector's live Find session (if any), restoring its
   *  votes from its labelset. Find's bulk presumptions live in the same vote
   *  dicts the Train window trains from, so Train calls this on entry before it
   *  reads them — otherwise the whole collection reads as voted and Autopilot
   *  lands in a terminal phase on arrival (#3212). A no-op when Find never ran. */
  endFindSession(): Observable<FindEndSessionResponse> {
    return endFindSessionRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Detector-evaluation stats over the adopted Find label set (2x2 confusion,
   *  Kept rate, and the precision-vs-returned curve). Pure read. */
  getFindStats(): Observable<FindStatsResponse> {
    return findStats(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Evidence-coverage report for the active detector on the active dataset:
   *  how much of the scored dataset the detector is calling *without labeled
   *  evidence behind the call*, measured from the detector's own labelset. The
   *  cross-user complement to the atlas domain-shift report — it needs no
   *  reference dataset loaded. `available` is false when there's nothing to
   *  measure (no scored Find run / no resolvable labelset). Pure read. */
  getEvidenceCoverage(): Observable<FindEvidenceCoverageResponse> {
    return findEvidenceCoverage(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Fold the Find corrections into the active detector's labelset and retrain
   *  its MLP. Destructive: changes the detector, so the current Find evaluation
   *  no longer applies and the caller should re-score afterwards. */
  addCorrectionsToDetector(): Observable<FindCorrectionsToDetectorResponse> {
    return findCorrectionsToDetector(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Results of a finished background AutoFind, for the user who started it.
   *  ``runId`` is the run's ``task_id``. 404 once it has aged out of the
   *  server's small in-memory window. */
  getAutofindRun(runId: string, context?: HttpContext): Observable<AutoFindRunResponse> {
    return getAutofindRun(this.http, this.config.rootUrl, { run_id: runId }, context).pipe(map((r) => r.body));
  }

  /** Ask the server to start laying out a finished run's Good results for
   *  Browse while the Find Results dialog is open (#4683). Starts nothing while
   *  other work is in flight: `busy` means ask again later. */
  prepAutofindRunBrowse(runId: string, context?: HttpContext): Observable<AutoFindBrowsePrepResponse> {
    return prepAutofindRunBrowse(this.http, this.config.rootUrl, { run_id: runId }, context).pipe(map((r) => r.body));
  }
}
