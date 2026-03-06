import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DetectorInfo } from '../models/api.models';
import { DetectorsApiService } from './detectors-api.service';

@Injectable({ providedIn: 'root' })
export class DetectorStateService implements OnDestroy {
  private readonly detectorsSubject = new BehaviorSubject<DetectorInfo[]>([]);
  private readonly extractorsSubject = new BehaviorSubject<unknown[]>([]);
  private readonly localizersSubject = new BehaviorSubject<unknown[]>([]);
  private readonly destroy$ = new Subject<void>();

  readonly detectors$ = this.detectorsSubject.asObservable();
  readonly extractors$ = this.extractorsSubject.asObservable();
  readonly localizers$ = this.localizersSubject.asObservable();

  constructor(private detectorsApi: DetectorsApiService) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get detectors(): DetectorInfo[] {
    return this.detectorsSubject.value;
  }

  get extractors(): unknown[] {
    return this.extractorsSubject.value;
  }

  get localizers(): unknown[] {
    return this.localizersSubject.value;
  }

  loadDetectors(): void {
    this.detectorsApi
      .getAutorunDetectors()
      .pipe(takeUntil(this.destroy$))
      .subscribe((resp) => this.detectorsSubject.next(resp.detectors));
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
    this.loadDetectors();
    this.loadExtractors();
    this.loadLocalizers();
  }

  clear(): void {
    this.detectorsSubject.next([]);
    this.extractorsSubject.next([]);
    this.localizersSubject.next([]);
  }
}
