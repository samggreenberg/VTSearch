import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, Observable } from 'rxjs';
import { takeUntil, tap } from 'rxjs/operators';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { SettingsApiService } from './settings-api.service';
import { ANIMATIONS_OFF_CLASS } from '../utils/reduced-motion';

@Injectable({ providedIn: 'root' })
export class SettingsStateService implements OnDestroy {
  private readonly settingsSubject = new BehaviorSubject<AppSettings | null>(null);
  private readonly destroy$ = new Subject<void>();
  private loading = false;

  readonly settings$ = this.settingsSubject.asObservable();

  constructor(private settingsApi: SettingsApiService) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get settings(): AppSettings | null {
    return this.settingsSubject.value;
  }

  load(): void {
    if (this.loading) return;
    this.loading = true;
    this.settingsApi
      .getSettings()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (settings) => {
          this.emit(settings);
          this.loading = false;
        },
        error: () => {
          this.loading = false;
        },
      });
  }

  update(changes: SettingsUpdate): Observable<AppSettings> {
    return this.settingsApi.updateSettings(changes).pipe(
      takeUntil(this.destroy$),
      tap((updated) => this.emit(updated)),
    );
  }

  clear(): void {
    this.emit(null);
  }

  /**
   * Publish new settings and reflect document-level effects. Currently this
   * mirrors the "Show Animations" toggle onto `<html>` as the
   * `animations-off` class, which the global stylesheet and the
   * `prefersReducedMotion()` util both honor so the one setting silences every
   * decorative animation at once.
   */
  private emit(settings: AppSettings | null): void {
    this.settingsSubject.next(settings);
    if (typeof document !== 'undefined') {
      const animationsOff = settings?.show_animations === false;
      document.documentElement.classList.toggle(ANIMATIONS_OFF_CLASS, animationsOff);
    }
  }
}
