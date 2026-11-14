import { Component, PendingTasks, Type } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router, Routes, provideRouter } from '@angular/router';
import type { Mock } from 'vitest';
import { configureZoneless } from '../testing/zoneless-testbed';
import { provideIdleRoutePreload } from './idle-route-preload';

@Component({ selector: 'vt-preload-home', template: '' })
class HomeComponent {}

@Component({ selector: 'vt-preload-view', template: '' })
class ViewComponent {}

type Loader = Mock<() => Promise<Type<unknown>>>;

/** A loader whose chunk "arrives" when the test says so. */
function heldLoader(): { loader: Loader; arrive: () => void; fail: () => void } {
  let arrive!: () => void;
  let fail!: () => void;
  const chunk = new Promise<Type<unknown>>((resolve, reject) => {
    arrive = () => resolve(ViewComponent);
    fail = () => reject(new Error('chunk failed'));
  });
  return { loader: vi.fn(() => chunk), arrive, fail };
}

/** Let the strategy's idle wait (a timer here: jsdom has no idle callback) pass. */
async function idle(): Promise<void> {
  for (let i = 0; i < 5; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

/**
 * The other views' chunks are fetched ahead of the click, but only while the
 * app has nothing else to fetch, and one view at a time (#5663).
 */
describe('provideIdleRoutePreload', () => {
  /** Configure the router; the caller navigates. */
  function configure(routes: Routes): Router {
    configureZoneless({
      providers: [provideRouter([{ path: 'home', component: HomeComponent }, ...routes]), provideIdleRoutePreload()],
    });
    return TestBed.inject(Router);
  }

  async function open(routes: Routes): Promise<Router> {
    const router = configure(routes);
    await router.navigateByUrl('/home');
    return router;
  }

  afterEach(() => {
    Reflect.deleteProperty(navigator, 'connection');
  });

  it('loads a view nobody has opened once the app is idle', async () => {
    const loader: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    await open([{ path: 'view', loadComponent: loader }]);
    await idle();

    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('waits while a request is in flight', async () => {
    const loader: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    const router = configure([{ path: 'view', loadComponent: loader }]);
    // What HttpClient does for the length of a request.
    const finishRequest = TestBed.inject(PendingTasks).add();
    await router.navigateByUrl('/home');
    await idle();

    expect(loader).not.toHaveBeenCalled();

    finishRequest();
    await idle();
    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('loads one view at a time', async () => {
    const first = heldLoader();
    const second = heldLoader();
    await open([
      { path: 'first', loadComponent: first.loader },
      { path: 'second', loadComponent: second.loader },
    ]);
    await idle();

    expect(first.loader).toHaveBeenCalledTimes(1);
    expect(second.loader).not.toHaveBeenCalled();

    first.arrive();
    await idle();
    expect(second.loader).toHaveBeenCalledTimes(1);
    second.arrive();
    await idle();
  });

  it('waits for a request that starts between two views', async () => {
    const first = heldLoader();
    const second = heldLoader();
    await open([
      { path: 'first', loadComponent: first.loader },
      { path: 'second', loadComponent: second.loader },
    ]);
    await idle();
    const finishRequest = TestBed.inject(PendingTasks).add();
    first.arrive();
    await idle();

    expect(second.loader).not.toHaveBeenCalled();

    finishRequest();
    await idle();
    expect(second.loader).toHaveBeenCalledTimes(1);
    second.arrive();
    await idle();
  });

  it('drops a failed preload and carries on with the next view', async () => {
    const failing = heldLoader();
    const next: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    await open([
      { path: 'failing', loadComponent: failing.loader },
      { path: 'next', loadComponent: next },
    ]);
    await idle();
    failing.fail();
    await idle();

    expect(next).toHaveBeenCalledTimes(1);
  });

  it('preloads a child route as well as its parent', async () => {
    const parent: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    const child: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    await open([{ path: 'parent', loadComponent: parent, children: [{ path: 'child', loadComponent: child }] }]);
    await idle();

    expect(parent).toHaveBeenCalledTimes(1);
    expect(child).toHaveBeenCalledTimes(1);
  });

  it('runs once, not after every navigation', async () => {
    const loader: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    const router = await open([{ path: 'view', loadComponent: loader }]);
    await idle();
    await router.navigateByUrl('/home?again=1');
    await idle();

    expect(loader).toHaveBeenCalledTimes(1);
  });

  it('preloads nothing when the browser asks to save data', async () => {
    Object.defineProperty(navigator, 'connection', { configurable: true, value: { saveData: true } });
    const loader: Loader = vi.fn(() => Promise.resolve<Type<unknown>>(ViewComponent));
    await open([{ path: 'view', loadComponent: loader }]);
    await idle();

    expect(loader).not.toHaveBeenCalled();
  });
});
