import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DetectorsApiService } from './detectors-api.service';

/**
 * Shared cache for autorun extractors / localizers loaded from the backend.
 * The DetectorContext (loaded detector state) lives in the detector registry
 * (see DetectorsApiService); this service only caches the autorun
 * extractor / localizer lists.
 */
@Injectable({ providedIn: 'root' })
export class DetectorStateService implements OnDestroy {
  private readonly extractorsSubject = new BehaviorSubject<unknown[]>([]);
  private readonly localizersSubject = new BehaviorSubject<unknown[]>([]);
  private readonly destroy$ = new Subject<void>();

  readonly extractors$ = this.extractorsSubject.asObservable();
  readonly localizers$ = this.localizersSubject.asObservable();

  constructor(private detectorsApi: DetectorsApiService) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get extractors(): unknown[] {
    return this.extractorsSubject.value;
  }

  get localizers(): unknown[] {
    return this.localizersSubject.value;
  }

  loadExtractors(): void {
    this.detectorsApi
      .getAutorunExtractors()
      .pipe(takeUntil(this.destroy$))
      .subscribe((resp) => this.extractorsSubject.next(resp.extractors));
  }

  loadLocalizers(): void {
    this.detectorsApi
      .getAutorunLocalizers()
      .pipe(takeUntil(this.destroy$))
      .subscribe((resp) => this.localizersSubject.next(resp.localizers));
  }

  loadAll(): void {
    this.loadExtractors();
    this.loadLocalizers();
  }

  clear(): void {
    this.extractorsSubject.next([]);
    this.localizersSubject.next([]);
  }
}
