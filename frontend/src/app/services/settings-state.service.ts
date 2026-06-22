import { Injectable, computed, effect, inject, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { SettingsApiService } from './settings-api.service';
import { ANIMATIONS_OFF_CLASS } from '../utils/reduced-motion';

/**
 * App-settings store, migrated onto Angular's reactive resource primitives
 * (see `docs/plans/httpresource-migration.md`). The GET is driven by an
 * `rxResource` wrapping the existing generated-client method
 * (`SettingsApiService.getSettings()`), so it keeps the typed client and the
 * interceptor chain while dropping the hand-rolled subscribe/`BehaviorSubject`
 * bookkeeping. The public surface is signal-based: read `settingsSignal()`
 * (and `isLoading()` / `error()`); call `load()`/`update()`/`clear()` to drive
 * it. Consumers react with `effect()` rather than subscribing to an Observable.
 */
@Injectable({ providedIn: 'root' })
export class SettingsStateService {
  private readonly settingsApi = inject(SettingsApiService);

  // A monotonic load counter doubles as the resource request. `0` means
  // "not requested yet" -> `params()` returns `undefined` -> the resource stays
  // idle (no fetch). Each `load()` bumps the counter, which changes the request
  // and so re-runs the loader.
  private readonly loadCount = signal(0);

  private readonly resource = rxResource({
    params: () => (this.loadCount() === 0 ? undefined : this.loadCount()),
    stream: () => this.settingsApi.getSettings(),
  });

  /** Canonical settings signal: `null` until first successful load (or after `clear()`). */
  readonly settingsSignal = computed<AppSettings | null>(() => this.resource.value() ?? null);

  /** True while a settings fetch is in flight. */
  readonly isLoading = this.resource.isLoading;

  /** The error from the last failed fetch, if any. */
  readonly error = this.resource.error;

  constructor() {
    // Reflect document-level effects of settings whenever the resolved value
    // changes (load, update, or clear). Currently this mirrors the "Show
    // Animations" toggle onto `<html>` as the `animations-off` class, which the
    // global stylesheet and `prefersReducedMotion()` both honor so the one
    // setting silences every decorative animation at once.
    effect(() => {
      const settings = this.settingsSignal();
      if (typeof document !== 'undefined') {
        const animationsOff = settings?.show_animations === false;
        document.documentElement.classList.toggle(ANIMATIONS_OFF_CLASS, animationsOff);
      }
    });
  }

  /** Fetch (or refetch) settings from the server. No-op while a fetch is in flight. */
  load(): void {
    if (this.resource.isLoading()) return;
    this.loadCount.update((n) => n + 1);
  }

  update(changes: SettingsUpdate): Observable<AppSettings> {
    return this.settingsApi
      .updateSettings(changes)
      .pipe(tap((updated) => this.resource.set(updated)));
  }

  clear(): void {
    this.resource.set(undefined);
  }
}
