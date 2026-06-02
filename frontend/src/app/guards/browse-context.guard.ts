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

      if (
        activeContext.datasetId === datasetId &&
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
