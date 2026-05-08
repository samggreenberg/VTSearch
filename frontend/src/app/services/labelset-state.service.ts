import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, Subject, timer } from 'rxjs';
import { catchError, switchMap, takeUntil, tap } from 'rxjs/operators';
import { LabelElement, LabelsDetailResponse } from '../models/api.models';
import { DetectorsApiService } from './detectors-api.service';

@Injectable({ providedIn: 'root' })
export class LabelsetStateService implements OnDestroy {
  private readonly goodSubject = new BehaviorSubject<LabelElement[]>([]);
  private readonly badSubject = new BehaviorSubject<LabelElement[]>([]);
  private readonly mediaTypeSubject = new BehaviorSubject<string>('');
  private readonly stopPolling$ = new Subject<void>();
  private readonly destroy$ = new Subject<void>();
  private polling = false;
  private modelName: string | null = null;

  readonly good$ = this.goodSubject.asObservable();
  readonly bad$ = this.badSubject.asObservable();
  readonly mediaType$ = this.mediaTypeSubject.asObservable();

  constructor(private api: DetectorsApiService) {}

  ngOnDestroy(): void {
    this.stopPolling();
    this.destroy$.next();
    this.destroy$.complete();
  }

  get good(): LabelElement[] {
    return this.goodSubject.value;
  }

  get bad(): LabelElement[] {
    return this.badSubject.value;
  }

  setModel(name: string | null): void {
    if (this.modelName === name) return;
    this.modelName = name;
    if (!name) {
      this.goodSubject.next([]);
      this.badSubject.next([]);
      this.mediaTypeSubject.next('');
      return;
    }
    this.refresh();
  }

  startPolling(intervalMs: number = 1500): void {
    if (this.polling) return;
    this.polling = true;
    timer(0, intervalMs)
      .pipe(takeUntil(this.stopPolling$), switchMap(() => this.fetch$()))
      .subscribe();
  }

  stopPolling(): void {
    if (!this.polling) return;
    this.polling = false;
    this.stopPolling$.next();
  }

  refresh(): void {
    this.fetch$().subscribe();
  }

  vote(elementId: string, vote: 'good' | 'bad'): void {
    if (!this.modelName) return;
    const name = this.modelName;
    this.applyOptimisticVote(elementId, vote);
    this.api.voteLabelElement(name, elementId, vote).subscribe({
      next: () => this.refresh(),
      error: () => this.refresh(),
    });
  }

  /** Optimistic: flip the element's label or remove it on same-vote toggle. */
  private applyOptimisticVote(elementId: string, vote: 'good' | 'bad'): void {
    const allCurrent = [...this.goodSubject.value, ...this.badSubject.value];
    const elem = allCurrent.find((e) => e.id === elementId);
    if (!elem) return;

    const nextGood = this.goodSubject.value.filter((e) => e.id !== elementId);
    const nextBad = this.badSubject.value.filter((e) => e.id !== elementId);
    if (elem.label === vote) {
      // Same vote → remove
      this.goodSubject.next(nextGood);
      this.badSubject.next(nextBad);
      return;
    }
    // Flip
    const flipped: LabelElement = { ...elem, label: vote };
    if (vote === 'good') nextGood.push(flipped);
    else nextBad.push(flipped);
    this.goodSubject.next(nextGood);
    this.badSubject.next(nextBad);
  }

  private fetch$(): Observable<LabelsDetailResponse | never> {
    if (!this.modelName) {
      this.goodSubject.next([]);
      this.badSubject.next([]);
      this.mediaTypeSubject.next('');
      return EMPTY;
    }
    const name = this.modelName;
    return this.api.getLabelsDetail(name).pipe(
      tap((resp) => {
        if (this.modelName !== name) return;
        this.goodSubject.next(resp.good ?? []);
        this.badSubject.next(resp.bad ?? []);
        this.mediaTypeSubject.next(resp.media_type ?? '');
      }),
      catchError(() => EMPTY),
    );
  }
}
