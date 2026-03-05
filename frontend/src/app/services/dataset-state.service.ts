import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, forkJoin } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DatasetsApiService } from './datasets-api.service';
import { TrainableModelsApiService } from './trainable-models-api.service';

@Injectable({ providedIn: 'root' })
export class DatasetStateService implements OnDestroy {
  private readonly datasetsSubject = new BehaviorSubject<any[]>([]);
  private readonly modelsSubject = new BehaviorSubject<any[]>([]);
  private readonly loadingSubject = new BehaviorSubject<boolean>(false);
  private readonly progressMessageSubject = new BehaviorSubject<string>('');
  private readonly destroy$ = new Subject<void>();

  readonly datasets$ = this.datasetsSubject.asObservable();
  readonly models$ = this.modelsSubject.asObservable();
  readonly loading$ = this.loadingSubject.asObservable();
  readonly progressMessage$ = this.progressMessageSubject.asObservable();

  constructor(
    private datasetsApi: DatasetsApiService,
    private modelsApi: TrainableModelsApiService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get datasets(): any[] {
    return this.datasetsSubject.value;
  }

  get models(): any[] {
    return this.modelsSubject.value;
  }

  get loading(): boolean {
    return this.loadingSubject.value;
  }

  get progressMessage(): string {
    return this.progressMessageSubject.value;
  }

  setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }

  setProgressMessage(message: string): void {
    this.progressMessageSubject.next(message);
  }

  refresh(): void {
    forkJoin({
      datasets: this.datasetsApi.getRegistry(),
      models: this.modelsApi.getRegistry(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ datasets, models }) => {
          this.datasetsSubject.next((datasets as any).datasets || []);
          this.modelsSubject.next((models as any).models || []);
        },
      });
  }

  clear(): void {
    this.datasetsSubject.next([]);
    this.modelsSubject.next([]);
    this.loadingSubject.next(false);
    this.progressMessageSubject.next('');
  }
}
