import { ApplicationConfig, provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';

import { routes } from './app.routes';
import { activeContextInterceptor } from './interceptors/active-context.interceptor';
import { achievementsRefreshInterceptor } from './interceptors/achievements-refresh.interceptor';
import { errorInterceptor } from './interceptors/error.interceptor';
import { metricsInterceptor } from './interceptors/metrics.interceptor';
import { timezoneInterceptor } from './interceptors/timezone.interceptor';
import { provideRouteChunkPrefetch } from './utils/route-chunk-prefetch';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes),
    // Fetch a lazy view's chunks while its guard waits, not after (#5657).
    provideRouteChunkPrefetch(),
    provideHttpClient(
      withInterceptors([
        timezoneInterceptor,
        activeContextInterceptor,
        achievementsRefreshInterceptor,
        metricsInterceptor,
        errorInterceptor,
      ]),
    ),
  ],
};
