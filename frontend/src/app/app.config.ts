import { ApplicationConfig, provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';

import { routes } from './app.routes';
import { activeContextInterceptor } from './interceptors/active-context.interceptor';
import { achievementsRefreshInterceptor } from './interceptors/achievements-refresh.interceptor';
import { errorInterceptor } from './interceptors/error.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    provideHttpClient(
      withInterceptors([
        activeContextInterceptor,
        achievementsRefreshInterceptor,
        errorInterceptor,
      ]),
    ),
  ],
};
