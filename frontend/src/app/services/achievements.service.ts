import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AchievementState } from '../generated/api-client/models/achievement-state';
import type { CheckPhraseResponse } from '../generated/api-client/models/check-phrase-response';
import type { PendingAnnouncement } from '../generated/api-client/models/pending-announcement';
import { acknowledgeAchievement } from '../generated/api-client/fn/achievements/acknowledge-achievement';
import { checkPhrase } from '../generated/api-client/fn/achievements/check-phrase';
import { getAchievements } from '../generated/api-client/fn/achievements/get-achievements';
import { SettingsStateService } from './settings-state.service';

const EMPTY_STATE: AchievementState = {
  tier_names: [],
  achievements: [],
  pending_announcements: [],
  docs: [],
};

/**
 * AchievementsService: polls /api/achievements, exposes the current state
 * as an observable, and queues unlock notifications one at a time so the
 * consumer (a global host component) can render dialogs sequentially.
 *
 * Trigger `refresh()` after any action that might unlock a tier (vote,
 * find, dataset load complete, label import).  Acknowledged announcements
 * are persisted server-side so they don't replay on reload.
 */
@Injectable({ providedIn: 'root' })
export class AchievementsService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);
  private settingsState = inject(SettingsStateService);

  private readonly state$ = new BehaviorSubject<AchievementState>(EMPTY_STATE);
  private readonly unlock$ = new Subject<PendingAnnouncement>();
  private inFlight = false;
  private disabled = false;

  constructor() {
    this.settingsState.settings$.subscribe((s) => {
      const next = !!s?.disable_achievements;
      const flipped = next && !this.disabled;
      this.disabled = next;
      if (flipped) {
        // Disabled: drop the cached state so the UI doesn't show
        // counters/unlocks from before the toggle.
        this.state$.next(EMPTY_STATE);
      }
    });
  }

  /** Stream of state snapshots for UI binding. */
  get state(): Observable<AchievementState> {
    return this.state$.asObservable();
  }

  /** Stream of unlocks to display one at a time. */
  get unlocks(): Observable<PendingAnnouncement> {
    return this.unlock$.asObservable();
  }

  get snapshot(): AchievementState {
    return this.state$.value;
  }

  /**
   * Fetch latest state and emit any new unlocks.  Coalesces concurrent
   * calls so a tight burst of actions doesn't fire N requests.
   */
  refresh(): void {
    if (this.disabled) {
      this.state$.next(EMPTY_STATE);
      return;
    }
    if (this.inFlight) return;
    this.inFlight = true;
    getAchievements(this.http, this.config.rootUrl)
      .pipe(
        map((r) => r.body),
        catchError(() => of(EMPTY_STATE)),
      )
      .subscribe((next) => {
        this.inFlight = false;
        this.state$.next(next);
        for (const p of next.pending_announcements) {
          this.unlock$.next(p);
        }
      });
  }

  /**
   * Acknowledge an unlock so it isn't shown again on the next refresh.
   * Refreshes state afterward to update the panel display.
   */
  acknowledge(categoryId: string, tierIdx: number): void {
    acknowledgeAchievement(this.http, this.config.rootUrl, {
      category_id: categoryId,
      body: { tier_idx: tierIdx },
    })
      .pipe(catchError(() => of(null)))
      .subscribe(() => this.refresh());
  }

  /**
   * Submit a Readme Reader phrase guess.  Returns the server's classification
   * (matched / which doc / whether already credited) and refreshes the state
   * so the docs panel reflects the new read state on success.
   */
  checkPhrase(phrase: string): Observable<CheckPhraseResponse> {
    return new Observable<CheckPhraseResponse>((subscriber) => {
      checkPhrase(this.http, this.config.rootUrl, { body: { phrase } })
        .pipe(
          map((r) => r.body),
          catchError(() =>
            of<CheckPhraseResponse>({
              matched: false,
              doc_id: null,
              doc_name: null,
              already_read: false,
            }),
          ),
        )
        .subscribe((result) => {
          if (result.matched && !result.already_read) {
            this.refresh();
          }
          subscriber.next(result);
          subscriber.complete();
        });
    });
  }

  /** Absolute URL to the raw markdown for a doc. */
  docRawUrl(docId: string): string {
    return `/api/achievements/docs/${encodeURIComponent(docId)}/raw`;
  }
}
