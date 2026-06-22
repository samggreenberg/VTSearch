import { HttpInterceptorFn } from '@angular/common/http';

/**
 * Attaches an `X-Timezone-Offset` header to every outgoing HTTP request,
 * carrying the browser's `Date.prototype.getTimezoneOffset()` value (the
 * difference, in minutes, between UTC and local time: positive west of UTC,
 * negative east of it).
 *
 * The backend uses it to bucket the "Around the Clock" hours and "Your Days
 * are Numbered" days by the user's local wall-clock time rather than UTC, so
 * those milestones reflect the clock the user actually saw on their screen.
 */
export const timezoneInterceptor: HttpInterceptorFn = (req, next) => {
  const offset = new Date().getTimezoneOffset();
  return next(req.clone({ headers: req.headers.set('X-Timezone-Offset', String(offset)) }));
};
