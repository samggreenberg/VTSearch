import { Injectable, Signal, computed, effect, inject, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { SettingsApiService } from './settings-api.service';
import { ANIMATIONS_OFF_CLASS, ANIMATIONS_ON_CLASS } from '../utils/reduced-motion';

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
/**
 * A single `{media_type: value}` settings preference, resolved for one
 * media type.
 *
 * Handed out by {@link SettingsStateService.perMediaType}. `value` is a
 * `computed` over the settings signal and the caller's media-type signal, so it
 * is a first-class reactive dependency: a template that reads it repaints when
 * either changes, with no shadow dict to hydrate and no `effect()` to mirror it
 * (see `docs/FRONTEND.md` section 5). `set` writes one media type's entry back,
 * merging into the live dict so sibling media types survive.
 */
export interface PerMediaTypePref<T> {
  /** The preference for the active media type, or the default when unset. */
  readonly value: Signal<T>;
  /** The whole `{media_type: value}` dict as currently loaded. */
  readonly dict: Signal<Record<string, T>>;
  /**
   * Persist `next` for the active media type, preserving every other type's
   * entry. No-op when the media type is empty (nothing to key the write on).
   *
   * The merge is taken from the live dict, so it depends on the settings signal
   * never reading as empty while the values are merely in flight — see
   * {@link SettingsStateService.load}.
   */
  set(next: T): Observable<AppSettings> | null;
}

/** Options for {@link SettingsStateService.perMediaType}. */
export interface PerMediaTypeOptions<T> {
  /** Value to report when the dict has no entry for the active media type. */
  fallback: T;
  /**
   * Optional normalizer applied to a stored value. Return `undefined` to reject
   * it and fall back — that is how a clamp, an enum-membership check, or a
   * type guard against a hand-edited settings file is expressed.
   */
  coerce?: (raw: unknown) => T | undefined;
}

@Injectable({ providedIn: 'root' })
export class SettingsStateService {
  private readonly settingsApi = inject(SettingsApiService);

  // A monotonic load counter doubles as the resource request. `0` means
  // "not requested yet" -> `params()` returns `undefined` -> the resource stays
  // idle (no fetch). The FIRST `load()` bumps the counter, which changes the
  // request and so runs the loader; later ones reload in place instead, which
  // keeps the last value while the refetch is in flight (see `load()`).
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
    // changes (load, update, or clear). This mirrors the "Show Animations"
    // pulldown onto `<html>`: "Hide" adds the `animations-off` class and "Show"
    // adds the `animations-on` class (which forces motion on even against an OS
    // reduce-motion request); "OS Setting" leaves both off so the platform
    // preference governs. The global stylesheet and `prefersReducedMotion()`
    // both honor these classes so the one setting governs every decorative
    // animation at once.
    effect(() => {
      const settings = this.settingsSignal();
      if (typeof document !== 'undefined') {
        const mode = settings?.show_animations;
        const root = document.documentElement.classList;
        root.toggle(ANIMATIONS_OFF_CLASS, mode === 'hide');
        root.toggle(ANIMATIONS_ON_CLASS, mode === 'show');
      }
    });
  }

  /**
   * Fetch (or refetch) settings from the server. No-op while a fetch is in
   * flight.
   *
   * A **refetch goes through `reload()`**, not through another bump of the
   * request counter. Both re-run the loader, but changing the request throws
   * the resource's stream away, so `value()` — and with it `settingsSignal()` —
   * reverts to `null` for the length of the round-trip; `reload()` keeps the
   * last value and only flips `isLoading()`. That window is not cosmetic:
   * `load()` is called from half a dozen components' init hooks, so it lands
   * regularly on a mounted view, and while it is open every consumer reads
   * "no settings yet". Two things go wrong there.
   *
   * - **Values flicker.** A preference resolved through the null reports its
   *   default and then snaps back — panel widths, thumbnail sizes, focus modes.
   * - **A concurrent write destroys data.** A `{media_type: value}` write
   *   merges into the dict it can see; against a null that merge yields
   *   `{[mediaType]: value}` alone, and `PUT /api/settings` *replaces* the
   *   stored dict — so every other media type's entry is silently dropped.
   *   {@link perMediaType} is exactly this shape.
   *
   * The first load still bumps the counter: there is nothing to reload yet.
   */
  load(): void {
    if (this.resource.isLoading()) return;
    if (this.loadCount() > 0 && this.resource.reload()) return;
    this.loadCount.update((n) => n + 1);
  }

  update(changes: SettingsUpdate): Observable<AppSettings> {
    return this.settingsApi
      .updateSettings(changes)
      .pipe(tap((updated) => this.resource.set(updated)));
  }

  /**
   * Bind one `{media_type: value}` settings key to a media-type signal.
   *
   * This is the sanctioned shape for a per-media-type preference. It replaces
   * the read/coerce/merge-write dance that used to be written longhand at every
   * consumer — a shadow `Record` field, an `effect()` mirroring the settings
   * signal into it, a second `effect()` re-deriving the value on a media-type
   * switch, and a bespoke spread on the way back out.
   *
   * Two things fall out of collapsing that into a `computed`:
   *
   * - **Reactivity is structural.** The returned `value` is a real signal, so a
   *   template binding tracks it and repaints under zoneless change detection
   *   on its own, rather than depending on some co-located `effect()` in the
   *   same component happening to dirty the view.
   * - **There is no mirror to go stale.** The old shadow dicts were hydrated
   *   behind an `if (dict && typeof dict === 'object')` guard, so a key that
   *   went absent server-side left the last-seen copy in place forever. Reading
   *   through a `computed` has no such state.
   *
   * @param key       the `AppSettings` key holding the dict.
   * @param mediaType signal carrying the active media type (`''` when unknown).
   * @param options   `fallback`, plus an optional `coerce` that rejects an
   *                  out-of-range or unrecognized stored value by returning
   *                  `undefined`.
   */
  perMediaType<T>(
    key: keyof AppSettings & string,
    mediaType: Signal<string>,
    options: PerMediaTypeOptions<T>,
  ): PerMediaTypePref<T> {
    const dict = computed<Record<string, T>>(() => {
      const raw = this.settingsSignal()?.[key];
      return raw && typeof raw === 'object' && !Array.isArray(raw)
        ? (raw as Record<string, T>)
        : {};
    });

    const value = computed<T>(() => {
      const mt = mediaType();
      if (!mt) return options.fallback;
      const raw = dict()[mt];
      if (raw === undefined) return options.fallback;
      if (!options.coerce) return raw;
      const coerced = options.coerce(raw);
      return coerced === undefined ? options.fallback : coerced;
    });

    return {
      value,
      dict,
      set: (next: T) => {
        const mt = mediaType();
        if (!mt) return null;
        // Merge, never replace: dropping the other media types' entries here is
        // the one way this helper could silently destroy user data.
        const merged = { ...dict(), [mt]: next };
        return this.update({ [key]: merged } as unknown as SettingsUpdate);
      },
    };
  }

  clear(): void {
    this.resource.set(undefined);
  }
}
