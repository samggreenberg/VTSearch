import {
  EnvironmentInjector,
  EnvironmentProviders,
  inject,
  provideEnvironmentInitializer,
  runInInjectionContext,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRouteSnapshot, Router, RoutesRecognized } from '@angular/router';
import { filter, isObservable } from 'rxjs';

/**
 * Start a lazy route's chunk download as soon as the router has matched the
 * route, instead of once its guards have passed (#5657).
 *
 * The router calls a route's `loadComponent` only after `canActivate` and the
 * resolvers. A guard that waits on the network holds the view's chunk back
 * for the whole wait: Browse's and People's guards await both registries on
 * a fresh load, and the pair guard may load the pair. The chunk's own static
 * imports then wait as well, since the browser finds them only once the
 * chunk is parsed. Browse's are 16 more files, which an HTTP/1.1 browser
 * fetches in waves. Calling the loader at `RoutesRecognized` overlaps all of
 * that with the guard.
 *
 * The router's own call lands later on the same module record, so nothing is
 * fetched twice. A guard that redirects leaves the chunk in the cache, unused
 * until that view is opened. A failed prefetch is dropped: the router's own
 * load reports the failure as it always has.
 */
export function provideRouteChunkPrefetch(): EnvironmentProviders {
  return provideEnvironmentInitializer(() => {
    const injector = inject(EnvironmentInjector);
    inject(Router)
      .events.pipe(
        filter((e): e is RoutesRecognized => e instanceof RoutesRecognized),
        takeUntilDestroyed(),
      )
      .subscribe((e) => prefetchRouteChunks(e.state.root, injector));
  });
}

/** Call every `loadComponent` in the matched tree under *root*. */
export function prefetchRouteChunks(root: ActivatedRouteSnapshot, injector: EnvironmentInjector): void {
  const pending = [root];
  for (let snapshot = pending.pop(); snapshot; snapshot = pending.pop()) {
    const load = snapshot.routeConfig?.loadComponent;
    if (load) {
      // The router runs the loader in an injection context too.
      const loaded = runInInjectionContext(injector, load);
      if (isObservable(loaded)) {
        loaded.subscribe({ error: () => undefined });
      } else if (loaded instanceof Promise) {
        loaded.catch(() => undefined);
      }
    }
    pending.push(...snapshot.children);
  }
}
