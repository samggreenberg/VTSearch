import { Injectable, OnDestroy, inject } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, Subject, timer } from 'rxjs';
import { catchError, switchMap, takeUntil, tap } from 'rxjs/operators';
import type { DetectorLabelView } from '../generated/api-client/models/detector-label-view';
import type { DetectorLabelsDetailResponse } from '../generated/api-client/models/detector-labels-detail-response';
import { DetectorsCrudApiService } from './detectors-crud-api.service';

@Injectable({ providedIn: 'root' })
export class LabelsetStateService implements OnDestroy {
  private api = inject(DetectorsCrudApiService);

  private readonly goodSubject = new BehaviorSubject<DetectorLabelView[]>([]);
  private readonly badSubject = new BehaviorSubject<DetectorLabelView[]>([]);
  private readonly mediaTypeSubject = new BehaviorSubject<string>('');
  private readonly stopPolling$ = new Subject<void>();
  private readonly destroy$ = new Subject<void>();
  private polling = false;
  private modelName: string | null = null;

  readonly good$ = this.goodSubject.asObservable();
  readonly bad$ = this.badSubject.asObservable();
  readonly mediaType$ = this.mediaTypeSubject.asObservable();

  ngOnDestroy(): void {
    this.stopPolling();
    this.destroy$.next();
    this.destroy$.complete();
  }

  get good(): DetectorLabelView[] {
    return this.goodSubject.value;
  }

  get bad(): DetectorLabelView[] {
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

  /**
   * Apply a clicked Good/Bad direction to a labelset element.
   *
   * The click direction is translated into an absolute ``target`` (the
   * desired end state) before it leaves the client, mirroring the
   * center-pane media vote: re-clicking an element's current label removes
   * it, the opposite label flips it. Sending an absolute target keeps
   * repeated requests from stale tabs idempotent on the server
   * (logical-bug-audit H1).
   */
  vote(elementId: string, clickedDirection: 'good' | 'bad'): void {
    if (!this.modelName) return;
    const name = this.modelName;
    const target = this.targetFor(elementId, clickedDirection);
    this.applyOptimisticState(elementId, target);
    this.api.voteLabelElement(name, elementId, target).subscribe({
      next: () => this.refresh(),
      error: () => this.refresh(),
    });
  }

  /** Translate a clicked direction into an absolute target by the element's
   *  current label: same label → ``'remove'``, otherwise the clicked label. */
  private targetFor(elementId: string, clickedDirection: 'good' | 'bad'): 'good' | 'bad' | 'remove' {
    const elem = [...this.goodSubject.value, ...this.badSubject.value].find((e) => e.id === elementId);
    return elem?.label === clickedDirection ? 'remove' : clickedDirection;
  }

  /** Optimistically move the element to its absolute target state. */
  private applyOptimisticState(elementId: string, target: 'good' | 'bad' | 'remove'): void {
    const allCurrent = [...this.goodSubject.value, ...this.badSubject.value];
    const elem = allCurrent.find((e) => e.id === elementId);
    if (!elem) return;

    const nextGood = this.goodSubject.value.filter((e) => e.id !== elementId);
    const nextBad = this.badSubject.value.filter((e) => e.id !== elementId);
    if (target === 'remove') {
      this.goodSubject.next(nextGood);
      this.badSubject.next(nextBad);
      return;
    }
    const updated: DetectorLabelView = { ...elem, label: target };
    if (target === 'good') nextGood.push(updated);
    else nextBad.push(updated);
    this.goodSubject.next(nextGood);
    this.badSubject.next(nextBad);
  }

  private fetch$(): Observable<DetectorLabelsDetailResponse | never> {
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
