import { Routes } from '@angular/router';
import { activeContextGuard } from './guards/active-context.guard';
import { browseContextGuard } from './guards/browse-context.guard';

/**
 * Routes. The label/tune views encode the active (dataset, detector)
 * pair in the URL so reload, share-link, and browser back/forward all
 * carry the pair correctly. The bare `/label` and `/tune` paths are
 * legacy redirects (they have no pair to encode and would land on a
 * broken view), so we bounce them back to the Dashboard.
 *
 * The Tune view was called Find until #4525 and Test until #5098, and its code
 * is still Find: the `/tune` route renders `FindViewComponent`, and the API keeps `find`
 * throughout (`/api/find-label`, `find_mode`, the `find` SSE channel).
 *
 * Browse is dataset-only (no detector required): `/browse/:datasetId`.
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
    path: 'tune/:datasetId/:detectorId',
    canActivate: [activeContextGuard],
    loadComponent: () =>
      import('./components/find-view/find-view.component').then(
        (m) => m.FindViewComponent,
      ),
  },
  {
    path: 'browse/:datasetId',
    canActivate: [browseContextGuard],
    loadComponent: () =>
      import('./components/browse-view/browse-view.component').then(
        (m) => m.BrowseViewComponent,
      ),
  },
  // Legacy / malformed paths: bounce to dashboard rather than render a
  // half-pair view.
  { path: 'label', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'tune', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'browse', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: '**', redirectTo: 'dashboard' },
];
