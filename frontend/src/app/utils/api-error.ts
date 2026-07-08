/**
 * Extract a human-readable message from an HTTP error response body.
 *
 * The backend deliberately emits two error envelope shapes:
 *
 *  - `error_response()` (vtsearch/routes/_shared.py):
 *    `{ "error": ..., "detail": ..., "request_id": ... }`
 *  - `flask_smorest.abort()`:
 *    `{ "code", "status", "message", "errors" }`
 *
 * The global error toast interceptor already reads both keys; inline
 * component handlers must too, or every `abort(400, message=...)` route
 * degrades to the component's generic fallback string.  Funnel all inline
 * error-message extraction through here.
 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  const body = (err as { error?: { error?: unknown; message?: unknown } } | null | undefined)
    ?.error;
  const message = body?.error ?? body?.message;
  return typeof message === 'string' && message ? message : fallback;
}
