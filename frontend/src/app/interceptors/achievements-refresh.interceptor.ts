import { HttpInterceptorFn, HttpResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { tap } from 'rxjs/operators';
import { AchievementsService } from '../services/achievements.service';

/**
 * After any action endpoint that could unlock an achievement, schedule a
 * refresh of the AchievementsService.  The refresh itself coalesces
 * concurrent calls so bursts of votes don't fan out to N requests.
 *
 * Endpoints watched:
 * - POST /api/medias/:id/vote                 (votes_cast, detectors_trained)
 * - POST /api/detectors/:name/labels/:id/vote (votes_cast, detectors_trained)
 * - POST /api/find-label                      (find_media)
 * - POST /api/auto-detect                     (find_media)
 * - POST /api/label-importers/import/:name    (detectors_imported)
 * - POST /api/datasets/registry/:id/load      (datasets_loaded — fires when the load completes)
 */
const WATCHED_PATTERNS: RegExp[] = [
  /^\/api\/medias\/[^/]+\/vote$/,
  /^\/api\/detectors\/[^/]+\/labels\/[^/]+\/vote$/,
  /^\/api\/find-label$/,
  /^\/api\/auto-detect$/,
  /^\/api\/label-importers\/import\/[^/]+$/,
  /^\/api\/datasets\/registry\/[^/]+\/load$/,
];

export const achievementsRefreshInterceptor: HttpInterceptorFn = (req, next) => {
  if (req.method !== 'POST') {
    return next(req);
  }
  const path = req.url.split('?')[0];
  const watch = WATCHED_PATTERNS.some((re) => re.test(path));
  if (!watch) {
    return next(req);
  }

  const achievements = inject(AchievementsService);
  return next(req).pipe(
    tap({
      next: (event) => {
        if (event instanceof HttpResponse && event.status >= 200 && event.status < 300) {
          achievements.refresh();
        }
      },
    }),
  );
};
