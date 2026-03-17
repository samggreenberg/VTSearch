import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./components/dashboard/dashboard.component').then(
        (m) => m.DashboardComponent,
      ),
  },
  {
    path: 'label',
    loadComponent: () =>
      import('./components/label-view/label-view.component').then(
        (m) => m.LabelViewComponent,
      ),
  },
  {
    path: 'find',
    loadComponent: () =>
      import('./components/find-view/find-view.component').then(
        (m) => m.FindViewComponent,
      ),
  },
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
];
