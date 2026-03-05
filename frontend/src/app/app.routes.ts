import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { LabelViewComponent } from './components/label-view/label-view.component';

export const routes: Routes = [
  { path: 'dashboard', component: DashboardComponent },
  { path: 'label', component: LabelViewComponent },
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
];
