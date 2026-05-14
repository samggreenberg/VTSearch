import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject, of } from 'rxjs';
import { catchError } from 'rxjs/operators';

export interface AchievementInfo {
  id: string;
  name: string;
  description: string;
  icon: string;
  tiers: number[];
  counter: number;
  tier_idx: number;
  next_threshold: number | null;
}

export interface PendingAnnouncement {
  id: string;
  name: string;
  icon: string;
  tier_idx: number;
  tier_name: string;
  threshold: number;
}

export interface DocInfo {
  id: string;
  name: string;
  path: string;
  read: boolean;
}

export interface AchievementsState {
  tier_names: string[];
  achievements: AchievementInfo[];
  pending_announcements: PendingAnnouncement[];
  docs: DocInfo[];
}

export interface PhraseCheckResult {
  matched: boolean;
  doc_id: string | null;
  doc_name: string | null;
  already_read: boolean;
}

const EMPTY_STATE: AchievementsState = {
  tier_names: [],
  achievements: [],
  pending_announcements: [],
  docs: [],
};

/**
 * AchievementsService — polls /api/achievements, exposes the current state
 * as an observable, and queues unlock notifications one at a time so the
 * consumer (a global host component) can render dialogs sequentially.
 *
 * Trigger `refresh()` after any action that might unlock a tier (vote,
 * find, dataset load complete, label import).  Acknowledged announcements
 * are persisted server-side so they don't replay on reload.
 */
@Injectable({ providedIn: 'root' })
export class AchievementsService {
  private readonly state$ = new BehaviorSubject<AchievementsState>(EMPTY_STATE);
  private readonly unlock$ = new Subject<PendingAnnouncement>();
  private inFlight = false;

  constructor(private http: HttpClient) {}

  /** Stream of state snapshots for UI binding. */
  get state(): Observable<AchievementsState> {
    return this.state$.asObservable();
  }

  /** Stream of unlocks to display one at a time. */
  get unlocks(): Observable<PendingAnnouncement> {
    return this.unlock$.asObservable();
  }

  get snapshot(): AchievementsState {
    return this.state$.value;
  }

  /**
   * Fetch latest state and emit any new unlocks.  Coalesces concurrent
   * calls so a tight burst of actions doesn't fire N requests.
   */
  refresh(): void {
    if (this.inFlight) return;
    this.inFlight = true;
    this.http
      .get<AchievementsState>('/api/achievements')
      .pipe(catchError(() => of(EMPTY_STATE)))
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
    this.http
      .post(`/api/achievements/${encodeURIComponent(categoryId)}/acknowledge`, {
        tier_idx: tierIdx,
      })
      .pipe(catchError(() => of(null)))
      .subscribe(() => this.refresh());
  }

  /**
   * Submit a Readme Reader phrase guess.  Returns the server's classification
   * (matched / which doc / whether already credited) and refreshes the state
   * so the docs panel reflects the new read state on success.
   */
  checkPhrase(phrase: string): Observable<PhraseCheckResult> {
    return new Observable<PhraseCheckResult>((subscriber) => {
      this.http
        .post<PhraseCheckResult>('/api/achievements/check-phrase', { phrase })
        .pipe(
          catchError(() =>
            of<PhraseCheckResult>({
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
