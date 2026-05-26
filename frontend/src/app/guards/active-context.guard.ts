import { inject } from '@angular/core';
import { CanActivateFn, Router, UrlTree } from '@angular/router';
import { Observable, combineLatest, of } from 'rxjs';
import { filter, map, switchMap, take, tap } from 'rxjs/operators';
import { ActiveContextService } from '../services/active-context.service';
import { ContextSwitchService } from '../services/context-switch.service';
import { DatasetStateService } from '../services/dataset-state.service';
import { RecentSessionsService } from '../services/recent-sessions.service';
import { ToastService } from '../services/toast.service';

/**
 * Guard for the URL-driven active-context routes
 * (`/label/:datasetId/:detectorId`, `/find/:datasetId/:detectorId`).
 *
 * Resolves the URL pair into the active context, holding the route's
 * activation until any required dataset/detector load completes so the
 * view never renders against a half-prepared backend. Invalid ids
 * redirect to the Dashboard with a toast.
 *
 * - **Both halves present and exist** → flip `ActiveContextService` via
 *   `ContextSwitchService.applyActivePair(...)`, wait for loads, allow.
 * - **Either half missing from the URL** → redirect to `/dashboard`.
 * - **Either id not in the registry** → toast + redirect to
 *   `/dashboard`. The registry fetch is awaited (deep-link cold-starts
 *   may arrive before the AppComponent-triggered refresh completes).
 * - **Pair is incompatible (different media types)** → allow; the
 *   `<vt-incompatible-pair-explainer>` overlay takes over the view.
 */
export const activeContextGuard: CanActivateFn = (route) => {
  const router = inject(Router);
  const activeContext = inject(ActiveContextService);
  const contextSwitch = inject(ContextSwitchService);
  const datasetState = inject(DatasetStateService);
  const recentSessions = inject(RecentSessionsService);
  const toast = inject(ToastService);

  const datasetId = route.paramMap.get('datasetId') || '';
  const detectorId = route.paramMap.get('detectorId') || '';

  if (!datasetId || !detectorId) {
    return router.parseUrl('/dashboard');
  }

  // Trigger the registry fetch if it hasn't already happened (e.g.
  // direct deep-link load before AppComponent initialised). `refresh()`
  // is debounced internally via `switchMap`.
  if (!datasetState.loaded) {
    datasetState.refresh();
  }

  return datasetState.loaded$.pipe(
    filter((loaded) => loaded),
    take(1),
    switchMap((): Observable<true | UrlTree> => {
      const dataset = datasetState.datasets.find((d) => d.id === datasetId);
      const detector = datasetState.detectors.find((d) => d.id === detectorId);

      if (!dataset) {
        toast.error({
          message: `Dataset '${datasetId}' is not available.`,
          dedupKey: `active-context-guard:dataset:${datasetId}`,
        });
        return of(router.parseUrl('/dashboard'));
      }
      if (!detector) {
        toast.error({
          message: `Detector '${detectorId}' is not available.`,
          dedupKey: `active-context-guard:detector:${detectorId}`,
        });
        return of(router.parseUrl('/dashboard'));
      }

      // If the pair already matches (and nothing's loading), skip the
      // wait entirely; saves a microtask on intra-view navigation.
      if (
        activeContext.datasetId === datasetId &&
        activeContext.modelId === detectorId &&
        !contextSwitch.switching
      ) {
        recentSessions.bump(datasetId, detectorId);
        return of(true);
      }

      // Otherwise hold the route until prep completes. Incompatibility
      // is a valid UI state (the explainer renders against the new
      // pair), so we don't fast-fail on it here.
      return contextSwitch.applyActivePair(datasetId, detectorId).pipe(
        // `applyActivePair` returns an Observable that completes (no
        // value) on success. `combineLatest`-ing with an of(true) ensures
        // the guard emits `true` once prep settles.
        tap(() => recentSessions.bump(datasetId, detectorId)),
        switchMap(() => of(true as const)),
      );
    }),
  );
};
