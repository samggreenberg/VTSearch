import { Routes } from '@angular/router';
import { activeContextGuard } from './guards/active-context.guard';

/**
 * Routes. The label/find views encode the active (dataset, detector)
 * pair in the URL so reload, share-link, and browser back/forward all
 * carry the pair correctly. The bare `/label` and `/find` paths are
 * legacy redirects — they have no pair to encode and would land on a
 * broken view, so we bounce them back to the Dashboard.
 *
 * See `docs/plans/active-context-switcher.md` § Phase 2.
 */
export const routes: Routes = [
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./components/dashboard/dashboard.component').then(
        (m) => m.DashboardComponent,
      ),
  },
  {
    path: 'label/:datasetId/:detectorId',
    canActivate: [activeContextGuard],
    loadComponent: () =>
      import('./components/label-view/label-view.component').then(
        (m) => m.LabelViewComponent,
      ),
  },
  {
    path: 'find/:datasetId/:detectorId',
    canActivate: [activeContextGuard],
    loadComponent: () =>
      import('./components/find-view/find-view.component').then(
        (m) => m.FindViewComponent,
      ),
  },
  // Legacy / malformed paths: bounce to dashboard rather than render a
  // half-pair view.
  { path: 'label', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'find', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: '**', redirectTo: 'dashboard' },
];
