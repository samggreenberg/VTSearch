/**
 * Extract a human-readable message from an HTTP error response body.
 *
 * Every JSON error the backend emits carries one envelope
 * (`vtsearch/errors.py`):
 *
 * ```json
 * { "code": 404, "status": "Not Found", "message": "...", "request_id": "..." }
 * ```
 *
 * This used to have to read two spellings: a hand-rolled `{error, detail,
 * request_id}` envelope coexisted with flask-smorest's `{code, status,
 * message, errors}`, so an inline handler that read only one key degraded
 * every route on the other envelope to its generic fallback string. The
 * backend now emits `message` everywhere, so this is a single key read --
 * kept as a helper rather than inlined at each call site so the envelope
 * stays named in one place if it ever moves again.
 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  const body = (err as { error?: { message?: unknown } } | null | undefined)?.error;
  const message = body?.message;
  return typeof message === 'string' && message ? message : fallback;
}
