import { WritableSignal, signal } from '@angular/core';
import { ActiveContextService } from '../services/active-context.service';
import { SettingsStateService } from '../services/settings-state.service';
import type { AppSettings } from '../generated/api-client/models/app-settings';

/**
 * Shared service stubs for the frontend spec suite.
 *
 * Deliberately small: most component/service specs tailor their stubs to what
 * the individual test drives (different subjects, method subsets, and return
 * values), so only genuinely-identical, recurring stubs are centralized here —
 * pushing the bespoke ones in would hurt readability more than it saves. Sits
 * next to `zoneless-testbed.ts` (`provideZoneless()`) and `test-providers.ts`
 * (`provideHttpTesting()`); the three compose in a `TestBed` module definition.
 */

/** No-argument no-op, for void-returning stub methods. */
export const noop = (): void => {};

/**
 * `ActiveContextService` stub whose `mediaUrl` is the identity function — the
 * shape the browse-* specs share. Pass overrides to add the fields a given spec
 * drives (`pair$`, `datasetId`, `setActive`, …).
 */
export function makeActiveContextStub(
  overrides: Partial<ActiveContextService> = {},
): Partial<ActiveContextService> {
  return { mediaUrl: (p: string) => p, ...overrides };
}

/**
 * `SettingsStateService` stub for specs that must not hit HTTP but whose
 * component reads per-media-type preferences.
 *
 * `perMediaType` is **borrowed from the real prototype** rather than
 * reimplemented: it only touches `settingsSignal()` and `update()`, both of
 * which the stub supplies, so the spec exercises the shipped resolution and
 * merge logic instead of a copy that could quietly drift from it.
 *
 * Returns the stub together with the writable signal behind `settingsSignal`,
 * so a spec can push a settings value the way a load would.
 */
export function makeSettingsStateStub(overrides: Partial<SettingsStateService> = {}): {
  stub: Partial<SettingsStateService>;
  settings: WritableSignal<AppSettings | null>;
} {
  const settings = signal<AppSettings | null>(null);
  const stub: Partial<SettingsStateService> = {
    settingsSignal: settings as SettingsStateService['settingsSignal'],
    load: noop,
    perMediaType: SettingsStateService.prototype.perMediaType,
    ...overrides,
  };
  return { stub, settings };
}
