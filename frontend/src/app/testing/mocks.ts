import { ActiveContextService } from '../services/active-context.service';

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
