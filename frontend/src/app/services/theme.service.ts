import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { SettingsApiService } from './settings-api.service';

/** The persisted user choice. ``'system'`` means "follow the OS preference". */
export type Theme = 'dark' | 'light' | 'highviz' | 'system';

/** The concrete theme actually rendered (no ``'system'``). */
export type EffectiveTheme = 'dark' | 'light' | 'highviz';

const OS_MEDIA_QUERY = '(prefers-color-scheme: light)';

@Injectable({ providedIn: 'root' })
export class ThemeService implements OnDestroy {
  private themeSubject = new BehaviorSubject<Theme>('system');
  theme$: Observable<Theme> = this.themeSubject.asObservable();

  private osMedia: MediaQueryList | null = null;
  private osListener: ((e: MediaQueryListEvent) => void) | null = null;

  constructor(private settingsApi: SettingsApiService) {}

  ngOnDestroy(): void {
    this.unsubscribeOs();
  }

  get currentTheme(): Theme {
    return this.themeSubject.value;
  }

  /** Load the persisted theme from the backend and apply it. */
  loadFromSettings(): void {
    this.settingsApi.getSettings().subscribe({
      next: (settings) => {
        const stored = (settings.theme as Theme | null | undefined) ?? 'system';
        this.applyTheme(stored);
      },
      error: () => {
        // Settings unavailable; keep the default 'system' theme.
        this.applyTheme('system');
      },
    });
  }

  /** Set and persist the theme. */
  setTheme(theme: Theme): void {
    this.applyTheme(theme);
    this.settingsApi.updateSettings({ theme }).subscribe();
  }

  /** Return the OS-level color-scheme preference.
   *
   * Falls back to 'dark' when ``matchMedia`` is unavailable (SSR, tests
   * without a real browser environment).
   */
  detectOsTheme(): EffectiveTheme {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return 'dark';
    }
    return window.matchMedia(OS_MEDIA_QUERY).matches ? 'light' : 'dark';
  }

  /** Resolve the user's chosen theme to a concrete rendered value. */
  resolveEffectiveTheme(theme: Theme): EffectiveTheme {
    return theme === 'system' ? this.detectOsTheme() : theme;
  }

  private applyTheme(theme: Theme): void {
    this.themeSubject.next(theme);
    document.documentElement.setAttribute('data-theme', this.resolveEffectiveTheme(theme));
    // Only listen for OS theme changes while the user is on 'system'.
    if (theme === 'system') {
      this.subscribeOs();
    } else {
      this.unsubscribeOs();
    }
  }

  private subscribeOs(): void {
    if (this.osListener) return;
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    this.osMedia = window.matchMedia(OS_MEDIA_QUERY);
    this.osListener = () => {
      // Paranoia: only re-apply if still on 'system'. The listener is
      // removed when the user leaves 'system', but a stale event could
      // theoretically fire between change and removal.
      if (this.currentTheme === 'system') {
        document.documentElement.setAttribute('data-theme', this.detectOsTheme());
      }
    };
    this.osMedia.addEventListener('change', this.osListener);
  }

  private unsubscribeOs(): void {
    if (this.osMedia && this.osListener) {
      this.osMedia.removeEventListener('change', this.osListener);
    }
    this.osMedia = null;
    this.osListener = null;
  }
}
