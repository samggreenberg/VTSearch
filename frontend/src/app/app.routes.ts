import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { LabelComponent } from './components/label/label.component';

export const routes: Routes = [
  { path: 'dashboard', component: DashboardComponent },
  { path: 'label', component: LabelComponent },
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
];
