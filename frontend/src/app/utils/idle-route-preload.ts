import {
  ApplicationRef,
  EnvironmentInjector,
  EnvironmentProviders,
  inject,
  provideEnvironmentInitializer,
  runInInjectionContext,
} from '@angular/core';
import { NavigationEnd, Route, Router } from '@angular/router';
import { filter, firstValueFrom, isObservable, take } from 'rxjs';

/** How long the browser may put the idle callback off before it runs anyway. */
const IDLE_TIMEOUT_MS = 2000;

/**
 * Fetch every lazy view's chunks, one view at a time, while the app has
 * nothing else to do (#5663).
 *
 * A lazy view's chunk imports shared-module chunks the browser discovers only
 * once that chunk is parsed (Browse: 16), and an HTTP/1.1 browser fetches
 * them about six at a time. Opening Browse from the Dashboard spent 0.3 s of
 * 0.48 s on that at 100 ms of latency. `provideRouteChunkPrefetch()` cannot
 * help there: it overlaps the chunks with the guard's wait, and inside the
 * app the guard returns at once. So the chunks are fetched ahead of the
 * click instead.
 *
 * It starts after the first navigation ends, and before each view waits
 * until:
 *
 * - **the app is stable**: no `HttpClient` request, `resource` load or
 *   navigation is pending, so the view on screen has its data;
 * - **the browser is idle** (`requestIdleCallback`): nothing is rendering.
 *
 * Waiting again before every view keeps a preload from running into requests
 * the user starts meanwhile, and one view at a time leaves most of the
 * browser's connections free. The router's own `import()` of a view lands on
 * the module record the preload fetched, or joins a fetch still under way.
 *
 * This is the router's `withPreloading()` by hand: `RouterPreloader` costs
 * 2.5 kB of the eager bundle, and it asks its strategy about every route at
 * once, so the queue would be ours either way.
 *
 * Nothing is preloaded when the browser asks to save data. A failed preload
 * is dropped: the router loads the view itself when it is opened and reports
 * a failure then.
 */
export function provideIdleRoutePreload(): EnvironmentProviders {
  return provideEnvironmentInitializer(() => {
    if (savesData()) return;
    const router = inject(Router);
    const appRef = inject(ApplicationRef);
    const injector = inject(EnvironmentInjector);
    router.events
      .pipe(
        filter((e) => e instanceof NavigationEnd),
        take(1),
      )
      .subscribe(() => void preloadWhenIdle(lazyRoutes(router.config), appRef, injector));
  });
}

/** Every route under *routes* that loads its component lazily. */
function lazyRoutes(routes: readonly Route[]): Route[] {
  return routes.flatMap((route) => [...(route.loadComponent ? [route] : []), ...lazyRoutes(route.children ?? [])]);
}

async function preloadWhenIdle(routes: Route[], appRef: ApplicationRef, injector: EnvironmentInjector): Promise<void> {
  for (const route of routes) {
    await appRef.whenStable();
    await browserIdle();
    // Rendering during the idle wait may have started new requests.
    await appRef.whenStable();
    if (injector.destroyed) return;
    try {
      // The router runs the loader in an injection context too.
      const loaded = runInInjectionContext(injector, () => route.loadComponent?.());
      await (isObservable(loaded) ? firstValueFrom(loaded) : loaded);
    } catch {
      // Dropped: see above.
    }
  }
}

function savesData(): boolean {
  const connection = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection;
  return connection?.saveData === true;
}

function browserIdle(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestIdleCallback === 'function') {
      requestIdleCallback(() => resolve(), { timeout: IDLE_TIMEOUT_MS });
    } else {
      setTimeout(resolve, 0);
    }
  });
}
