import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import { listRecentSessions } from '../generated/api-client/fn/sessions/list-recent-sessions';
import { bumpRecentSession } from '../generated/api-client/fn/sessions/bump-recent-session';
import type { RecentSession } from '../generated/api-client/models/recent-session';

/**
 * Tracks the user's recent (dataset, detector) labelling sessions.
 *
 * Backed by ``/api/sessions/recent`` (per-user, persisted). The burger
 * menu subscribes to ``sessions$`` to render its "Recent sessions"
 * submenu; the active-context route guard calls ``bump()`` whenever
 * the user enters ``/label/:ds/:det`` or ``/find/:ds/:det``.
 *
 * Bump failures are swallowed; the "recent" surface is a convenience,
 * not a correctness signal, so a transient backend error shouldn't
 * block route activation.
 */
@Injectable({ providedIn: 'root' })
export class RecentSessionsService {
  private readonly http = inject(HttpClient);
  private readonly config = inject(ApiConfiguration);

  private readonly sessionsSubject = new BehaviorSubject<RecentSession[]>([]);
  readonly sessions$ = this.sessionsSubject.asObservable();

  get sessions(): RecentSession[] {
    return this.sessionsSubject.value;
  }

  refresh(): Observable<RecentSession[]> {
    return listRecentSessions(this.http, this.config.rootUrl).pipe(
      map((r) => r.body.sessions ?? []),
      tap((sessions) => this.sessionsSubject.next(sessions)),
      catchError(() => of(this.sessionsSubject.value)),
    );
  }

  bump(datasetId: string, detectorId: string): void {
    if (!datasetId || !detectorId) return;
    bumpRecentSession(this.http, this.config.rootUrl, {
      body: { dataset_id: datasetId, detector_id: detectorId },
    })
      .pipe(
        map((r) => r.body.sessions ?? []),
        catchError(() => of(this.sessionsSubject.value)),
      )
      .subscribe((sessions) => this.sessionsSubject.next(sessions));
  }
}
