import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, Observable } from 'rxjs';
import { takeUntil, tap } from 'rxjs/operators';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { SettingsApiService } from './settings-api.service';

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
          this.settingsSubject.next(settings);
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
      tap((updated) => this.settingsSubject.next(updated)),
    );
  }

  clear(): void {
    this.settingsSubject.next(null);
  }
}
