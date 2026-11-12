import { Injectable, Injector, OnDestroy, computed, inject } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { BehaviorSubject, Observable } from 'rxjs';
import { filter, take } from 'rxjs/operators';
import { SettingsApiService } from './settings-api.service';
import { SettingsStateService } from './settings-state.service';

/** The persisted user choice. ``'system'`` means "follow the OS preference". */
export type Theme = 'dark' | 'light' | 'highviz' | 'system';

/** The concrete theme actually rendered (no ``'system'``). */
export type EffectiveTheme = 'dark' | 'light' | 'highviz';

const OS_MEDIA_QUERY = '(prefers-color-scheme: light)';

@Injectable({ providedIn: 'root' })
export class ThemeService implements OnDestroy {
  private settingsApi = inject(SettingsApiService);
  private settingsState = inject(SettingsStateService);
  private injector = inject(Injector);

  private themeSubject = new BehaviorSubject<Theme>('system');
  theme$: Observable<Theme> = this.themeSubject.asObservable();

  private osMedia: MediaQueryList | null = null;
  private osListener: ((e: MediaQueryListEvent) => void) | null = null;

  ngOnDestroy(): void {
    this.unsubscribeOs();
  }

  get currentTheme(): Theme {
    return this.themeSubject.value;
  }

  /**
   * Apply the persisted theme once the app's settings first load. It rides
   * {@link SettingsStateService}'s request rather than sending one of its own,
   * which was a second `GET /api/settings` in the opening burst of requests,
   * all of them queued for the browser's few connections to the server
   * (#5649). Only that first answer is applied, as when this sent its own:
   * after it the theme is the user's to set ({@link setTheme}).
   */
  loadFromSettings(): void {
    this.settingsState.load();
    const stored = computed<Theme | null>(() => {
      // Settings unavailable; keep the default 'system' theme. Asked first:
      // after a failed load, reading the settings themselves throws.
      if (this.settingsState.error() != null) return 'system';
      const settings = this.settingsState.settingsSignal();
      return settings ? ((settings.theme as Theme | null | undefined) ?? 'system') : null;
    });
    toObservable(stored, { injector: this.injector })
      .pipe(
        filter((theme): theme is Theme => theme !== null),
        take(1),
      )
      .subscribe((theme) => this.applyTheme(theme));
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
