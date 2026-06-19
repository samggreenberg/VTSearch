import { TestBed, ComponentFixture } from '@angular/core/testing';

/**
 * Drain the asynchrony an `rxResource` introduces in Vitest specs.
 *
 * An `rxResource` loader is promise-based and runs in a root effect, so its
 * stream value commits on a microtask rather than synchronously. After flushing
 * the backing HTTP request (`httpMock.flush(...)`), `await settleResource()`
 * lets that microtask run and then ticks the TestBed so the resource's signal —
 * and any `computed`/`effect` reading it — updates before the assertion.
 *
 * Pair it with a `TestBed.tick()` *before* `httpMock.expectOne(...)` to issue
 * the GET (the loader effect doesn't run during `fixture.detectChanges()`).
 */
export async function settleResource(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve));
  TestBed.tick();
}

/**
 * Drain scheduled change detection in a zoneless spec (see
 * `docs/plans/zoneless-migration.md`, Phase 0.3 and `zoneless-testbed.ts`).
 *
 * After driving a component's state through the production channel (a service
 * signal/subject write, a dispatched bound event), `await settleZoneless(fixture)`
 * runs a macrotask — letting promise microtasks and `setTimeout`-based resource
 * resolution land — then awaits `fixture.whenStable()`, which flushes only the CD
 * that was actually *scheduled*. Crucially it does NOT call `detectChanges()`, so
 * a missing notification (a forgotten signal write / `markForCheck`) shows up as
 * stale DOM rather than being papered over by a forced render.
 *
 * Assert on the rendered DOM (`fixture.nativeElement.querySelector(...)`) after
 * this resolves, not on component fields.
 */
export async function settleZoneless(fixture: ComponentFixture<unknown>): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve));
  await fixture.whenStable();
}
