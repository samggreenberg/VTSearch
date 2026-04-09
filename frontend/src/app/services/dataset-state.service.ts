import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, forkJoin } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { DatasetRegistryEntry, LoadingTask, ModelRegistryEntry } from '../models/api.models';
import { DatasetsApiService } from './datasets-api.service';
import { TrainableModelsApiService } from './trainable-models-api.service';

@Injectable({ providedIn: 'root' })
export class DatasetStateService implements OnDestroy {
  private readonly datasetsSubject = new BehaviorSubject<DatasetRegistryEntry[]>([]);
  private readonly modelsSubject = new BehaviorSubject<ModelRegistryEntry[]>([]);
  private readonly loadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly loadingSubject = new BehaviorSubject<boolean>(false);
  private readonly progressMessageSubject = new BehaviorSubject<string>('');
  private readonly errorMessageSubject = new BehaviorSubject<string>('');
  private readonly destroy$ = new Subject<void>();
  /** Emits whenever a refresh is requested; switchMap ensures only the latest response is used. */
  private readonly refreshTrigger$ = new Subject<void>();

  readonly datasets$ = this.datasetsSubject.asObservable();
  readonly models$ = this.modelsSubject.asObservable();
  readonly loadingTasks$ = this.loadingTasksSubject.asObservable();
  readonly loading$ = this.loadingSubject.asObservable();
  readonly progressMessage$ = this.progressMessageSubject.asObservable();
  readonly errorMessage$ = this.errorMessageSubject.asObservable();

  constructor(
    private datasetsApi: DatasetsApiService,
    private modelsApi: TrainableModelsApiService,
  ) {
    // Single subscription that uses switchMap to cancel in-flight requests
    // when a new refresh is triggered, preventing stale responses from
    // overwriting fresh data.
    this.refreshTrigger$
      .pipe(
        switchMap(() =>
          forkJoin({
            datasets: this.datasetsApi.getRegistry(),
            models: this.modelsApi.getRegistry(),
          }),
        ),
        takeUntil(this.destroy$),
      )
      .subscribe({
        next: ({ datasets, models }) => {
          this.datasetsSubject.next(datasets.datasets || []);
          this.modelsSubject.next(models.models || []);
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

  get models(): ModelRegistryEntry[] {
    return this.modelsSubject.value;
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

  get errorMessage(): string {
    return this.errorMessageSubject.value;
  }

  setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }

  setProgressMessage(message: string): void {
    this.progressMessageSubject.next(message);
  }

  setErrorMessage(message: string): void {
    this.errorMessageSubject.next(message);
  }

  refresh(): void {
    this.refreshTrigger$.next();
  }

  clear(): void {
    this.datasetsSubject.next([]);
    this.modelsSubject.next([]);
    this.loadingTasksSubject.next([]);
    this.loadingSubject.next(false);
    this.progressMessageSubject.next('');
    this.errorMessageSubject.next('');
  }
}
