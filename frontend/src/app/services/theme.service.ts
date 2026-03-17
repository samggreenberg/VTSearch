import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { SettingsApiService } from './settings-api.service';

export type Theme = 'dark' | 'light' | 'highviz';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private themeSubject = new BehaviorSubject<Theme>('dark');
  theme$: Observable<Theme> = this.themeSubject.asObservable();

  constructor(private settingsApi: SettingsApiService) {}

  get currentTheme(): Theme {
    return this.themeSubject.value;
  }

  /** Load the persisted theme from the backend. */
  loadFromSettings(): void {
    this.settingsApi.getSettings().subscribe({
      next: (settings) => {
        if (settings.theme) {
          this.applyTheme(settings.theme as Theme);
        }
      },
      error: () => {
        // Settings unavailable — keep the default dark theme.
      },
    });
  }

  /** Set and persist the theme. */
  setTheme(theme: Theme): void {
    this.applyTheme(theme);
    this.settingsApi.updateSettings({ theme }).subscribe();
  }

  private applyTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);
    this.themeSubject.next(theme);
  }
}
