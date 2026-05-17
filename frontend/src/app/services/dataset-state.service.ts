import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, forkJoin, of } from 'rxjs';
import { catchError, switchMap, takeUntil } from 'rxjs/operators';
import { DatasetRegistryEntry, LoadingTask, DetectorRegistryEntry } from '../models/api.models';
import { DatasetsApiService } from './datasets-api.service';
import { DetectorsApiService } from './detectors-api.service';

@Injectable({ providedIn: 'root' })
export class DatasetStateService implements OnDestroy {
  private readonly datasetsSubject = new BehaviorSubject<DatasetRegistryEntry[]>([]);
  private readonly detectorsSubject = new BehaviorSubject<DetectorRegistryEntry[]>([]);
  private readonly loadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly loadingSubject = new BehaviorSubject<boolean>(false);
  private readonly progressMessageSubject = new BehaviorSubject<string>('');
  /** Last registry-fetch error, or null if the most recent fetch
   *  succeeded. Surfaced inline in the context-pulldowns so the user can
   *  retry without leaving their current view. */
  private readonly errorSubject = new BehaviorSubject<string | null>(null);
  private readonly destroy$ = new Subject<void>();
  /** Emits whenever a refresh is requested; switchMap ensures only the latest response is used. */
  private readonly refreshTrigger$ = new Subject<void>();

  readonly datasets$ = this.datasetsSubject.asObservable();
  readonly detectors$ = this.detectorsSubject.asObservable();
  readonly loadingTasks$ = this.loadingTasksSubject.asObservable();
  readonly loading$ = this.loadingSubject.asObservable();
  readonly progressMessage$ = this.progressMessageSubject.asObservable();
  readonly error$ = this.errorSubject.asObservable();

  constructor(
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
  ) {
    // Single subscription that uses switchMap to cancel in-flight requests
    // when a new refresh is triggered, preventing stale responses from
    // overwriting fresh data. `catchError` keeps the outer pipeline alive
    // after a failed fetch so the next `refresh()` can retry.
    this.refreshTrigger$
      .pipe(
        switchMap(() =>
          forkJoin({
            datasets: this.datasetsApi.getRegistry(),
            detectors: this.detectorsApi.getRegistry(),
          }).pipe(catchError(() => of(null))),
        ),
        takeUntil(this.destroy$),
      )
      .subscribe({
        next: (res) => {
          if (res === null) {
            this.errorSubject.next("Couldn't load datasets and detectors.");
            return;
          }
          this.datasetsSubject.next(res.datasets.datasets || []);
          this.detectorsSubject.next(res.detectors.detectors || []);
          if (this.errorSubject.value !== null) this.errorSubject.next(null);
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

  get loadingTasks(): LoadingTask[] {
    return this.loadingTasksSubject.value;
  }

  setLoadingTasks(tasks: LoadingTask[]): void {
    this.loadingTasksSubject.next(tasks);
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
    this.loadingTasksSubject.next([]);
    this.loadingSubject.next(false);
    this.progressMessageSubject.next('');
    this.errorSubject.next(null);
  }
}
