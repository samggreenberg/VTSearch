import { TestBed } from '@angular/core/testing';

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
