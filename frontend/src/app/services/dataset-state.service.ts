import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, forkJoin, of } from 'rxjs';
import { catchError, switchMap, takeUntil } from 'rxjs/operators';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';

@Injectable({ providedIn: 'root' })
export class DatasetStateService implements OnDestroy {
  private readonly datasetsSubject = new BehaviorSubject<DatasetRegistryEntry[]>([]);
  private readonly detectorsSubject = new BehaviorSubject<DetectorRegistryEntry[]>([]);
  private readonly loadingSubject = new BehaviorSubject<boolean>(false);
  private readonly progressMessageSubject = new BehaviorSubject<string>('');
  /** Last registry-fetch error, or null if the most recent fetch
   *  succeeded. Surfaced inline in the context-pulldowns so the user can
   *  retry without leaving their current view. */
  private readonly errorSubject = new BehaviorSubject<string | null>(null);
  /** Flips to `true` the first time the registry returns from the
   *  server, success or empty. Used by the active-context route guard
   *  to know when it's safe to validate a URL pair against the
   *  registry — on a deep-link cold start, the guard may run before the
   *  initial fetch lands. */
  private readonly loadedSubject = new BehaviorSubject<boolean>(false);
  private readonly destroy$ = new Subject<void>();
  /** Emits whenever a refresh is requested; switchMap ensures only the latest response is used. */
  private readonly refreshTrigger$ = new Subject<void>();

  readonly datasets$ = this.datasetsSubject.asObservable();
  readonly detectors$ = this.detectorsSubject.asObservable();
  readonly loading$ = this.loadingSubject.asObservable();
  readonly progressMessage$ = this.progressMessageSubject.asObservable();
  readonly error$ = this.errorSubject.asObservable();
  readonly loaded$ = this.loadedSubject.asObservable();

  constructor(
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private detectorsRegistryApi: DetectorsRegistryApiService,
  ) {
    // Single subscription that uses switchMap to cancel in-flight requests
    // when a new refresh is triggered, preventing stale responses from
    // overwriting fresh data. `catchError` keeps the outer pipeline alive
    // after a failed fetch so the next `refresh()` can retry.
    this.refreshTrigger$
      .pipe(
        switchMap(() =>
          forkJoin({
            datasets: this.datasetsRegistryApi.getRegistry(),
            detectors: this.detectorsRegistryApi.getRegistry(),
          }).pipe(catchError(() => of(null))),
        ),
        takeUntil(this.destroy$),
      )
      .subscribe({
        next: (res) => {
          if (res === null) {
            this.errorSubject.next("Couldn't load datasets and detectors.");
            // A failed fetch still resolves the "have we tried?" question,
            // so the route guard doesn't hang forever.
            if (!this.loadedSubject.value) this.loadedSubject.next(true);
            return;
          }
          this.datasetsSubject.next(
            (res.datasets.datasets || []) as unknown as DatasetRegistryEntry[],
          );
          this.detectorsSubject.next(res.detectors.detectors || []);
          if (this.errorSubject.value !== null) this.errorSubject.next(null);
          if (!this.loadedSubject.value) this.loadedSubject.next(true);
        },
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get datasets(): DatasetRegistryEntry[] {
    return this.datasetsSubject.value;
  }

  get detectors(): DetectorRegistryEntry[] {
    return this.detectorsSubject.value;
  }

  get loading(): boolean {
    return this.loadingSubject.value;
  }

  get progressMessage(): string {
    return this.progressMessageSubject.value;
  }

  get error(): string | null {
    return this.errorSubject.value;
  }

  get loaded(): boolean {
    return this.loadedSubject.value;
  }

  setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }

  setProgressMessage(message: string): void {
    this.progressMessageSubject.next(message);
  }

  refresh(): void {
    this.refreshTrigger$.next();
  }

  clear(): void {
    this.datasetsSubject.next([]);
    this.detectorsSubject.next([]);
    this.loadingSubject.next(false);
    this.progressMessageSubject.next('');
    this.errorSubject.next(null);
    this.loadedSubject.next(false);
  }
}
