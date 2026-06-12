import { inject } from '@angular/core';
import { CanActivateFn, Router, UrlTree } from '@angular/router';
import { Observable, of } from 'rxjs';
import { filter, switchMap, take, tap } from 'rxjs/operators';
import { ActiveContextService } from '../services/active-context.service';
import { ContextSwitchService } from '../services/context-switch.service';
import { DatasetStateService } from '../services/dataset-state.service';
import { ToastService } from '../services/toast.service';

export const browseContextGuard: CanActivateFn = (route) => {
  const router = inject(Router);
  const activeContext = inject(ActiveContextService);
  const contextSwitch = inject(ContextSwitchService);
  const datasetState = inject(DatasetStateService);
  const toast = inject(ToastService);

  const datasetId = route.paramMap.get('datasetId') || '';

  if (!datasetId) {
    return router.parseUrl('/dashboard');
  }

  // Ephemeral detector-positives browse contexts (`__detpos__<id>`) are built
  // server-side and reached via the X-Dataset-Id header; they are deliberately
  // absent from the dataset registry, so skip the registry/load checks. The
  // dashboard's Browse button already set the active context before navigating.
  if (datasetId.startsWith('__detpos__')) {
    return true;
  }

  if (!datasetState.loaded) {
    datasetState.refresh();
  }

  return datasetState.loaded$.pipe(
    filter((loaded) => loaded),
    take(1),
    switchMap((): Observable<true | UrlTree> => {
      const dataset = datasetState.datasets.find((d) => d.id === datasetId);

      if (!dataset) {
        toast.error({
          message: `Dataset '${datasetId}' is not available.`,
          dedupKey: `browse-context-guard:dataset:${datasetId}`,
        });
        return of(router.parseUrl('/dashboard'));
      }

      // Fast-path only when the dataset is genuinely loaded in the
      // backend. A matching *active* id is not sufficient: the dataset
      // may have been unloaded/evicted since it was last active, in
      // which case short-circuiting here would let the browse view fire
      // requests that 409 `dataset_not_loaded`. When `loaded` is false
      // we fall through to `applyActivePair`, which loads it first.
      if (
        activeContext.datasetId === datasetId &&
        dataset.loaded &&
        !contextSwitch.switching
      ) {
        return of(true);
      }

      return contextSwitch.applyActivePair(datasetId, activeContext.modelId || '').pipe(
        switchMap(() => of(true as const)),
      );
    }),
  );
};
