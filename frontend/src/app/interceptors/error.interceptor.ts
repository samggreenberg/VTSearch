import { HttpContextToken, HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { ActiveContextService } from '../services/active-context.service';
import { ErrorContext, ToastService } from '../services/toast.service';

/**
 * Opt out of the global error toast for a single request.
 *
 * Use when the caller handles the failure itself (e.g. a probe that
 * expects 404, a retry loop, or a form that renders inline validation).
 * The error still propagates through ``catchError``/``error:`` handlers
 * exactly as before; only the global UI toast is suppressed.
 *
 * Example::
 *
 *     http.get('/api/foo', {
 *       context: new HttpContext().set(SKIP_ERROR_TOAST, true),
 *     });
 */
export const SKIP_ERROR_TOAST = new HttpContextToken<boolean>(() => false);

/**
 * Captures every failed HTTP response and pushes a structured
 * ``ErrorContext`` to ``ToastService`` as an error toast. Enriches the
 * response with the active dataset/detector IDs and the server-side
 * request_id (from the response body and/or ``X-Request-Id`` header).
 *
 * Repeating the same endpoint+status (e.g. a retry loop hammering a
 * broken backend) replaces the existing toast via dedup key rather
 * than stacking duplicates.
 */
export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const toast = inject(ToastService);
  const ctx = inject(ActiveContextService);

  return next(req).pipe(
    catchError((err: unknown) => {
      if (!(err instanceof HttpErrorResponse)) {
        return throwError(() => err);
      }
      if (req.context.get(SKIP_ERROR_TOAST)) {
        return throwError(() => err);
      }
      const parsed = parseErrorBody(err);
      const requestId =
        parsed.requestId || err.headers?.get('X-Request-Id') || undefined;
      const datasetId = req.headers.get('X-Dataset-Id') || ctx.datasetId || undefined;
      const detectorId = req.headers.get('X-Detector-Id') || ctx.modelId || undefined;
      const url = stripOrigin(req.url);
      const errorCtx: ErrorContext = {
        message: parsed.message || defaultMessage(err),
        detail: parsed.detail,
        status: err.status,
        statusText: err.statusText,
        method: req.method,
        url,
        requestId,
        datasetId,
        detectorId,
        rawBody: parsed.rawBody,
        extra: parsed.extra,
        timestamp: new Date().toISOString(),
      };
      toast.error({
        message: errorCtx.message,
        detail: errorCtx.detail,
        errorContext: errorCtx,
        dedupKey: `http:${req.method}:${url}:${err.status}`,
      });
      return throwError(() => err);
    }),
  );
};

interface ParsedBody {
  message?: string;
  detail?: string;
  requestId?: string;
  rawBody?: string;
  extra?: Record<string, unknown>;
}

function parseErrorBody(err: HttpErrorResponse): ParsedBody {
  const body = err.error;
  // String body: usually a non-JSON error page (HTML or plain text).
  if (typeof body === 'string') {
    return { rawBody: body.length > 4000 ? body.slice(0, 4000) + '…' : body };
  }
  if (body && typeof body === 'object') {
    const obj = body as Record<string, unknown>;
    // Standard {error, detail, request_id} shape from the backend.
    const message =
      pickString(obj, 'error') ||
      pickString(obj, 'message') ||
      undefined;
    const detail = pickString(obj, 'detail');
    const requestId = pickString(obj, 'request_id');
    // Surface any other top-level fields (e.g. missing_fields, available)
    // so they show up in the "extra" section.
    const known = new Set(['error', 'message', 'detail', 'request_id']);
    const extra: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      if (!known.has(k)) extra[k] = v;
    }
    let rawBody: string | undefined;
    try {
      rawBody = JSON.stringify(obj, null, 2);
    } catch {
      rawBody = undefined;
    }
    return {
      message,
      detail,
      requestId,
      rawBody,
      extra: Object.keys(extra).length > 0 ? extra : undefined,
    };
  }
  return {};
}

function pickString(obj: Record<string, unknown>, key: string): string | undefined {
  const v = obj[key];
  return typeof v === 'string' && v.length > 0 ? v : undefined;
}

function defaultMessage(err: HttpErrorResponse): string {
  if (err.status === 0) return 'Network error: could not reach the server.';
  return `Request failed (${err.status}${err.statusText ? ` ${err.statusText}` : ''}).`;
}

function stripOrigin(url: string): string {
  // Show paths as ``/api/foo``, not the full origin.
  try {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      const u = new URL(url);
      return u.pathname + u.search;
    }
  } catch {
    // fall through
  }
  return url;
}
