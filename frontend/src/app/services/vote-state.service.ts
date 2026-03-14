import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, timer } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { VotesResponse } from '../models/api.models';
import { SortingApiService } from './sorting-api.service';

@Injectable({ providedIn: 'root' })
export class VoteStateService implements OnDestroy {
  private readonly goodVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly badVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly clickTimesSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly learnedScoresSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly destroy$ = new Subject<void>();
  private readonly stopPolling$ = new Subject<void>();
  private polling = false;

  readonly goodVotes$ = this.goodVotesSubject.asObservable();
  readonly badVotes$ = this.badVotesSubject.asObservable();
  readonly clickTimes$ = this.clickTimesSubject.asObservable();
  readonly learnedScores$ = this.learnedScoresSubject.asObservable();

  constructor(private sortingApi: SortingApiService) {}

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

    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);

    // Set an optimistic click time so the item sorts correctly immediately,
    // rather than appearing with time=-1 and then jumping when the server responds.
    if (isAdd) {
      const times = { ...this.clickTimesSubject.value };
      const maxTime = Object.values(times).reduce((m, t) => Math.max(m, t), 0);
      times[String(id)] = maxTime + 1;
      this.clickTimesSubject.next(times);
    }
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
  }

  private applyVotes(votes: VotesResponse): void {
    this.goodVotesSubject.next(new Set(votes.good));
    this.badVotesSubject.next(new Set(votes.bad));
    this.clickTimesSubject.next(votes.click_times);
    this.learnedScoresSubject.next(votes.learned_scores);
  }
}
