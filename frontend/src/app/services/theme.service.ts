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

  /** Load the persisted theme from the backend.
   *
   * If the backend has no theme stored yet (first load for this user),
   * read `prefers-color-scheme` from the browser, apply it, and persist
   * it as the initial value so subsequent loads stay consistent.
   */
  loadFromSettings(): void {
    this.settingsApi.getSettings().subscribe({
      next: (settings) => {
        const stored = settings.theme as Theme | null | undefined;
        if (stored) {
          this.applyTheme(stored);
        } else {
          this.setTheme(this.detectOsTheme());
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

  /** Return the OS-level color-scheme preference ('dark' or 'light').
   *
   * Falls back to 'dark' when ``matchMedia`` is unavailable (SSR,
   * tests without a real browser environment).
   */
  detectOsTheme(): Theme {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }

  private applyTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);
    this.themeSubject.next(theme);
  }
}
