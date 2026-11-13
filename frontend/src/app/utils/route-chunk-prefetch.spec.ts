import { Component, Type } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router, Routes, provideRouter } from '@angular/router';
import type { Mock } from 'vitest';
import { configureZoneless } from '../testing/zoneless-testbed';
import { provideRouteChunkPrefetch } from './route-chunk-prefetch';

@Component({ selector: 'vt-prefetch-view', template: '' })
class ViewComponent {}

/**
 * The router loads a lazy route's component only after its guards pass. These
 * pin that the prefetch starts the load while a guard is still waiting (#5657),
 * and the router behaviour that makes the prefetch worth having.
 */
describe('provideRouteChunkPrefetch', () => {
  let loader: Mock<() => Promise<Type<unknown>>>;
  /** Resolves once the router has called the guard. */
  let guardEntered: Promise<void>;
  /** Lets the waiting guard answer. */
  let answerGuard: (allow: boolean) => void;
  let guard: () => Promise<boolean>;

  function configure(routes: Routes, prefetch: boolean): Router {
    configureZoneless({
      providers: [provideRouter(routes), ...(prefetch ? [provideRouteChunkPrefetch()] : [])],
    });
    return TestBed.inject(Router);
  }

  beforeEach(() => {
    loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    let entered!: () => void;
    guardEntered = new Promise<void>((resolve) => (entered = resolve));
    const answer = new Promise<boolean>((resolve) => (answerGuard = resolve));
    guard = () => {
      entered();
      return answer;
    };
  });

  const guarded = (): Routes => [{ path: 'view', canActivate: [() => guard()], loadComponent: loader }];

  it('starts the lazy load while the guard is still waiting', async () => {
    const router = configure(guarded(), true);
    const navigated = router.navigateByUrl('/view');
    await guardEntered;

    expect(loader).toHaveBeenCalledTimes(1);

    answerGuard(true);
    expect(await navigated).toBe(true);
    expect(router.url).toBe('/view');
    // The router makes its own call once the guard passes; in a browser both
    // `import()`s resolve to one module record, so the chunk is fetched once.
    expect(loader).toHaveBeenCalledTimes(2);
  });

  it('is needed: without it the router waits for the guard before loading', async () => {
    const router = configure(guarded(), false);
    const navigated = router.navigateByUrl('/view');
    await guardEntered;

    expect(loader).not.toHaveBeenCalled();

    answerGuard(true);
    expect(await navigated).toBe(true);
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('prefetches a child route as well as its parent', async () => {
    const child = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    const router = configure(
      [
        {
          path: 'parent',
          canActivate: [() => guard()],
          loadComponent: loader,
          children: [{ path: 'child', loadComponent: child }],
        },
      ],
      true,
    );
    const navigated = router.navigateByUrl('/parent/child');
    await guardEntered;

    expect(loader).toHaveBeenCalledTimes(1);
    expect(child).toHaveBeenCalledTimes(1);

    answerGuard(false);
    expect(await navigated).toBe(false);
  });

  it('drops a failed prefetch when the guard turns the navigation away', async () => {
    // The router never loads this chunk itself, so only the prefetch sees the
    // rejection; an unhandled one would fail the run.
    const failing = vi.fn((): Promise<Type<unknown>> => Promise.reject(new Error('chunk failed')));
    const router = configure([{ path: 'view', canActivate: [() => guard()], loadComponent: failing }], true);
    const navigated = router.navigateByUrl('/view');
    await guardEntered;
    expect(failing).toHaveBeenCalledTimes(1);

    answerGuard(false);
    expect(await navigated).toBe(false);
  });
});
