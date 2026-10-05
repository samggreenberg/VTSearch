import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { LineTestResponse } from '../generated/api-client/models/line-test-response';
import type { PrecisionCheckVote } from '../generated/api-client/models/precision-check-vote';
import { cancelLineTest } from '../generated/api-client/fn/line-test/cancel-line-test';
import { forgetLineTest } from '../generated/api-client/fn/line-test/forget-line-test';
import { getLineTest } from '../generated/api-client/fn/line-test/get-line-test';
import { startLineTest } from '../generated/api-client/fn/line-test/start-line-test';
import { unvoteLineTest } from '../generated/api-client/fn/line-test/unvote-line-test';
import { voteLineTest } from '../generated/api-client/fn/line-test/vote-line-test';

/**
 * Test mode's test of the line (`/api/line-test`, #4524): start it over the
 * Find pass's frozen scores, record a pick's vote, take one back, cancel, and
 * read its state. Every verb answers with the same body: the balance's state,
 * the test (running, else the last finished one), whether the result is stale
 * or the line has moved, and the line each balance preset would ship.
 */
@Injectable({ providedIn: 'root' })
export class LineTestApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /** The running test, or the last finished one; `test` null when there is neither. */
  get(): Observable<LineTestResponse> {
    return getLineTest(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Freeze the ranking and the line and deal the first round. 409 with no Find pass to test. */
  start(): Observable<LineTestResponse> {
    return startLineTest(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Record votes on the round's picks: session votes with provenance `test`, never training labels. */
  vote(votes: PrecisionCheckVote[]): Observable<LineTestResponse> {
    return voteLineTest(this.http, this.config.rootUrl, { body: { votes } }).pipe(map((r) => r.body));
  }

  /** Take one vote of the current round back (the ↓ key). */
  unvote(id: number): Observable<LineTestResponse> {
    return unvoteLineTest(this.http, this.config.rootUrl, { body: { id } }).pipe(map((r) => r.body));
  }

  /** Drop a running test; its votes stay session votes. A finished test is kept. */
  cancel(): Observable<LineTestResponse> {
    return cancelLineTest(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Forget the verdict the detector keeps for this dataset (#4526), and a finished test with it: the next start tests afresh. */
  forget(): Observable<LineTestResponse> {
    return forgetLineTest(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
