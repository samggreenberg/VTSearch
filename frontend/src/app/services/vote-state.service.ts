import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, timer } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { VotesResponse } from '../models/api.models';
import { MediasApiService } from './medias-api.service';
import { SortingApiService } from './sorting-api.service';

/**
 * One vote captured for Cmd/Ctrl-Z undo.  `previousPolarity` is the polarity
 * the media had *before* the click that produced this entry, so undo can
 * restore it with a single inverse POST to /api/medias/<id>/vote.
 */
export interface UndoEntry {
  mediaId: number;
  clickedDirection: 'good' | 'bad';
  previousPolarity: 'good' | 'bad' | null;
  mediaName: string;
}

export interface UndoToast {
  action: 'undo' | 'redo';
  mediaName: string;
}

const UNDO_STACK_MAX = 20;

@Injectable({ providedIn: 'root' })
export class VoteStateService implements OnDestroy {
  private readonly goodVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly badVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly clickTimesSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly learnedScoresSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly labelsetGoodCountSubject = new BehaviorSubject<number>(0);
  private readonly labelsetBadCountSubject = new BehaviorSubject<number>(0);
  private readonly destroy$ = new Subject<void>();
  private readonly stopPolling$ = new Subject<void>();
  private polling = false;
  /** Tracks optimistic votes not yet confirmed by the server. */
  private pendingOptimistic = new Map<number, { vote: 'good' | 'bad'; clickTime: number }>();

  /** Past votes available to undo, most-recent last.  Capped at UNDO_STACK_MAX. */
  private past: UndoEntry[] = [];
  /** Votes that have been undone and can be redone via Cmd/Ctrl-Shift-Z. */
  private future: UndoEntry[] = [];
  private readonly toastSubject = new Subject<UndoToast>();

  readonly goodVotes$ = this.goodVotesSubject.asObservable();
  readonly badVotes$ = this.badVotesSubject.asObservable();
  readonly clickTimes$ = this.clickTimesSubject.asObservable();
  readonly learnedScores$ = this.learnedScoresSubject.asObservable();
  readonly labelsetGoodCount$ = this.labelsetGoodCountSubject.asObservable();
  readonly labelsetBadCount$ = this.labelsetBadCountSubject.asObservable();
  /** Emits a short message every time an undo or redo executes. */
  readonly toast$ = this.toastSubject.asObservable();

  constructor(
    private sortingApi: SortingApiService,
    private mediasApi: MediasApiService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.stopPolling$.next();
    this.stopPolling$.complete();
  }

  get goodVotes(): Set<number> {
    return this.goodVotesSubject.value;
  }

  get badVotes(): Set<number> {
    return this.badVotesSubject.value;
  }

  get clickTimes(): Record<string, number> {
    return this.clickTimesSubject.value;
  }

  get learnedScores(): Record<string, number> {
    return this.learnedScoresSubject.value;
  }

  /**
   * Number of "good" labels in the active detector's saved labelset (across
   * all datasets the detector has been used with).  Falls back to current
   * dataset's good vote count when no detector is loaded.
   */
  get labelsetGoodCount(): number {
    return this.labelsetGoodCountSubject.value;
  }

  /** Bad-label counterpart of {@link labelsetGoodCount}. */
  get labelsetBadCount(): number {
    return this.labelsetBadCountSubject.value;
  }

  /**
   * True when the active detector has at least one good and one bad label
   * available for training — i.e. `/api/learned-sort` would succeed.
   */
  get learnedSortAvailable(): boolean {
    return this.labelsetGoodCount > 0 && this.labelsetBadCount > 0;
  }

  /**
   * Optimistically update local vote sets to reflect a toggle vote.
   *
   * This mirrors the backend toggle semantics so that callers checking
   * vote state immediately after a vote see the updated counts without
   * waiting for the async loadVotes() HTTP round-trip.
   */
  applyOptimisticVote(id: number, vote: 'good' | 'bad'): void {
    const good = new Set(this.goodVotesSubject.value);
    const bad = new Set(this.badVotesSubject.value);

    const isAdd = vote === 'good' ? !good.has(id) : !bad.has(id);

    if (vote === 'good') {
      if (good.has(id)) {
        good.delete(id);
      } else {
        good.add(id);
        bad.delete(id);
      }
    } else {
      if (bad.has(id)) {
        bad.delete(id);
      } else {
        bad.add(id);
        good.delete(id);
      }
    }

    // Set an optimistic click time so the item sorts correctly immediately,
    // rather than appearing with time=-1 and then jumping when the server responds.
    const times = { ...this.clickTimesSubject.value };
    if (isAdd) {
      const maxTime = Object.values(times).reduce((m, t) => Math.max(m, t), 0);
      const optimisticTime = maxTime + 1;
      times[String(id)] = optimisticTime;
      this.pendingOptimistic.set(id, { vote, clickTime: optimisticTime });
    } else {
      this.pendingOptimistic.delete(id);
    }

    // Emit all changes together so Angular sees a single consistent state.
    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);
    this.clickTimesSubject.next(times);
  }

  loadVotes(): void {
    this.sortingApi
      .getVotes()
      .pipe(takeUntil(this.destroy$))
      .subscribe((votes) => this.applyVotes(votes));
  }

  startPolling(intervalMs = 2000): void {
    if (this.polling) return;
    this.polling = true;
    timer(0, intervalMs)
      .pipe(
        takeUntil(this.stopPolling$),
        takeUntil(this.destroy$),
        switchMap(() => this.sortingApi.getVotes()),
      )
      .subscribe((votes) => this.applyVotes(votes));
  }

  stopPolling(): void {
    this.stopPolling$.next();
    this.polling = false;
  }

  clear(): void {
    this.goodVotesSubject.next(new Set());
    this.badVotesSubject.next(new Set());
    this.clickTimesSubject.next({});
    this.learnedScoresSubject.next({});
    this.labelsetGoodCountSubject.next(0);
    this.labelsetBadCountSubject.next(0);
    this.pendingOptimistic.clear();
    this.past = [];
    this.future = [];
  }

  /**
   * Snapshot the polarity *before* a vote click and push it onto the undo
   * stack.  Must be called BEFORE applyOptimisticVote (otherwise the snapshot
   * would already reflect the toggle).  Any pending redo entries are dropped,
   * matching standard editor undo semantics.
   */
  recordVote(mediaId: number, clickedDirection: 'good' | 'bad', mediaName: string): void {
    const previousPolarity: 'good' | 'bad' | null = this.goodVotesSubject.value.has(mediaId)
      ? 'good'
      : this.badVotesSubject.value.has(mediaId)
        ? 'bad'
        : null;
    this.past.push({ mediaId, clickedDirection, previousPolarity, mediaName });
    if (this.past.length > UNDO_STACK_MAX) this.past.shift();
    this.future = [];
  }

  canUndo(): boolean {
    return this.past.length > 0;
  }

  canRedo(): boolean {
    return this.future.length > 0;
  }

  /**
   * Reverse the most recent vote.  The inverse POST is:
   *   - {@code previousPolarity} if non-null (restores prior state, including
   *     polarity flips since /vote toggles mutual exclusion server-side), or
   *   - {@code clickedDirection} if previousPolarity was null (toggles off).
   *
   * Side effects that aren't reversible (achievements, label_history append,
   * click_counter monotonicity) are accepted — the user really did make the
   * click; we just put the item back where it was.
   */
  undo(): void {
    const entry = this.past.pop();
    if (!entry) return;
    this.future.push(entry);
    const direction: 'good' | 'bad' = entry.previousPolarity ?? entry.clickedDirection;
    this.applyOptimisticVote(entry.mediaId, direction);
    this.mediasApi.vote(entry.mediaId, direction).subscribe({
      next: () => this.loadVotes(),
      error: () => this.loadVotes(),
    });
    this.toastSubject.next({ action: 'undo', mediaName: entry.mediaName });
  }

  /** Re-apply the most recently undone vote — POST the original direction. */
  redo(): void {
    const entry = this.future.pop();
    if (!entry) return;
    this.past.push(entry);
    if (this.past.length > UNDO_STACK_MAX) this.past.shift();
    this.applyOptimisticVote(entry.mediaId, entry.clickedDirection);
    this.mediasApi.vote(entry.mediaId, entry.clickedDirection).subscribe({
      next: () => this.loadVotes(),
      error: () => this.loadVotes(),
    });
    this.toastSubject.next({ action: 'redo', mediaName: entry.mediaName });
  }

  private applyVotes(votes: VotesResponse): void {
    const good = new Set(votes.good);
    const bad = new Set(votes.bad);
    const times = { ...votes.click_times };

    // Preserve optimistic votes that the server hasn't acknowledged yet.
    // Without this, a stale polling response (or a loadVotes() response that
    // raced ahead of the vote POST) would remove the item from the grid,
    // causing it to disappear and reappear (flicker).
    for (const [id, opt] of this.pendingOptimistic) {
      const serverHasIt = opt.vote === 'good' ? good.has(id) : bad.has(id);
      if (serverHasIt) {
        // Server caught up — stop preserving this optimistic vote.
        this.pendingOptimistic.delete(id);
      } else {
        // Server hasn't processed the vote yet — keep optimistic state.
        if (opt.vote === 'good') {
          good.add(id);
          bad.delete(id);
        } else {
          bad.add(id);
          good.delete(id);
        }
        times[String(id)] = opt.clickTime;
      }
    }

    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);
    this.clickTimesSubject.next(times);
    this.learnedScoresSubject.next(votes.learned_scores);
    this.labelsetGoodCountSubject.next(votes.labelset_good_count ?? good.size);
    this.labelsetBadCountSubject.next(votes.labelset_bad_count ?? bad.size);
  }
}
