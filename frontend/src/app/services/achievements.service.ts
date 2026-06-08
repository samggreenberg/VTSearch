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
  media_types: [],
  hours: [],
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
  private readonly openPanel$ = new Subject<void>();
  readonly hasPending$ = this.state$.pipe(map((s) => s.pending_announcements.length > 0));
  readonly openPanelRequest$ = this.openPanel$.asObservable();
  private inFlight = false;
  private disabled = false;

  /**
   * Milestones already pushed to {@link unlock$} this session, keyed by
   * `categoryId:tierIdx`. The server keeps returning a milestone in
   * `pending_announcements` until the user opens the panel (which ACKs it),
   * so without this guard every `refresh()` — fired after votes, finds, and
   * navigation — would re-pop a toast for an already-shown unlock. The toast
   * fires once per real unlock; the notification dot stays server-driven.
   */
  private readonly emittedUnlocks = new Set<string>();

  constructor() {
    this.settingsState.settings$.subscribe((s) => {
      const next = !!s?.disable_achievements;
      const flipped = next && !this.disabled;
      this.disabled = next;
      if (flipped) {
        // Disabled: drop the cached state so the UI doesn't show
        // counters/unlocks from before the toggle. Clear the emitted set
        // too: a later re-enable wipes server counters, so nothing should
        // be remembered as already-toasted across the reset.
        this.state$.next(EMPTY_STATE);
        this.emittedUnlocks.clear();
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
          const key = `${p.id}:${p.tier_idx}`;
          if (this.emittedUnlocks.has(key)) continue;
          this.emittedUnlocks.add(key);
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

  /** Request the achievements panel to open (e.g. from a toast action button). */
  requestOpenPanel(): void {
    this.openPanel$.next();
  }

  /** Acknowledge all pending announcements so the notification dot clears. */
  acknowledgeAll(): void {
    for (const p of this.state$.value.pending_announcements) {
      this.acknowledge(p.id, p.tier_idx);
    }
  }

  /** Absolute URL to the raw markdown for a doc. */
  docRawUrl(docId: string): string {
    return `/api/achievements/docs/${encodeURIComponent(docId)}/raw`;
  }
}
