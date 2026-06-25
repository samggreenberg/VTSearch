import { Injectable, OnDestroy, inject, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { Subject, forkJoin, of } from 'rxjs';
import { catchError, switchMap, takeUntil } from 'rxjs/operators';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';

@Injectable({ providedIn: 'root' })
export class DatasetStateService implements OnDestroy {
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);

  // Signals (not BehaviorSubjects) so direct template reads of the value
  // getters (`datasets`/`loading`/`loaded`/…) repaint under zoneless OnPush
  // change detection: reading a signal through a getter during template
  // evaluation is tracked, so a `.set()` from the registry fetch schedules CD.
  // The `$` observables are kept as `toObservable` bridges so the remaining
  // RxJS consumers (context-pulldown, app.component, the route guards,
  // active-context-watcher) work unchanged.
  private readonly _datasets = signal<DatasetRegistryEntry[]>([]);
  private readonly _detectors = signal<DetectorRegistryEntry[]>([]);
  private readonly _loading = signal(false);
  private readonly _progressMessage = signal('');
  /** Last registry-fetch error, or null if the most recent fetch
   *  succeeded. Surfaced inline in the context-pulldowns so the user can
   *  retry without leaving their current view. */
  private readonly _error = signal<string | null>(null);
  /** Flips to `true` the first time the registry returns from the
   *  server, success or empty. Used by the active-context route guard
   *  to know when it's safe to validate a URL pair against the
   *  registry; on a deep-link cold start, the guard may run before the
   *  initial fetch lands. */
  private readonly _loaded = signal(false);
  private readonly destroy$ = new Subject<void>();
  /** Emits whenever a refresh is requested; switchMap ensures only the latest response is used. */
  private readonly refreshTrigger$ = new Subject<void>();

  readonly datasets$ = toObservable(this._datasets);
  readonly detectors$ = toObservable(this._detectors);
  readonly loading$ = toObservable(this._loading);
  readonly progressMessage$ = toObservable(this._progressMessage);
  readonly error$ = toObservable(this._error);
  readonly loaded$ = toObservable(this._loaded);

  constructor() {
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
            this._error.set("Couldn't load datasets and detectors.");
            // A failed fetch still resolves the "have we tried?" question,
            // so the route guard doesn't hang forever.
            if (!this._loaded()) this._loaded.set(true);
            return;
          }
          this._datasets.set(
            (res.datasets.datasets || []) as unknown as DatasetRegistryEntry[],
          );
          this._detectors.set(res.detectors.detectors || []);
          if (this._error() !== null) this._error.set(null);
          if (!this._loaded()) this._loaded.set(true);
        },
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get datasets(): DatasetRegistryEntry[] {
    return this._datasets();
  }

  get detectors(): DetectorRegistryEntry[] {
    return this._detectors();
  }

  get loading(): boolean {
    return this._loading();
  }

  get progressMessage(): string {
    return this._progressMessage();
  }

  get error(): string | null {
    return this._error();
  }

  get loaded(): boolean {
    return this._loaded();
  }

  setLoading(loading: boolean): void {
    this._loading.set(loading);
  }

  setProgressMessage(message: string): void {
    this._progressMessage.set(message);
  }

  refresh(): void {
    this.refreshTrigger$.next();
  }

  clear(): void {
    this._datasets.set([]);
    this._detectors.set([]);
    this._loading.set(false);
    this._progressMessage.set('');
    this._error.set(null);
    this._loaded.set(false);
  }
}
