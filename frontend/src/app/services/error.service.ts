import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

/**
 * Structured error context surfaced by the global error banner.
 *
 * Populated by the ``errorInterceptor`` whenever an HTTP request fails.
 * Carries everything a user needs to file a useful bug report: the
 * endpoint, status, server message, request_id, and the active dataset
 * and detector at the time of failure.
 */
export interface ErrorContext {
  /** Short human-readable headline. */
  message: string;
  /** Optional longer detail (server-provided ``detail`` field or traceback summary). */
  detail?: string;
  /** HTTP status code (0 when the request never reached the server). */
  status: number;
  /** HTTP status text from the response, when available. */
  statusText?: string;
  /** Request method (GET/POST/...). */
  method: string;
  /** Request URL. */
  url: string;
  /** ``X-Request-Id`` echoed back by the server, when present. */
  requestId?: string;
  /** Active ``X-Dataset-Id`` at request time. */
  datasetId?: string;
  /** Active ``X-Detector-Id`` at request time. */
  detectorId?: string;
  /** Raw server response body, JSON-stringified for the copy bundle. */
  rawBody?: string;
  /** Any additional structured fields returned by the server (e.g. ``missing_fields``). */
  extra?: Record<string, unknown>;
  /** ISO-8601 timestamp captured client-side when the error fired. */
  timestamp: string;
}

/**
 * Central error sink for the frontend.
 *
 * The HTTP error interceptor pushes errors here; the global
 * ``ErrorBannerComponent`` (mounted in ``AppComponent``) subscribes and
 * renders them. There is at most one active error at a time — a new
 * error replaces any previous one (the user sees the most recent
 * failure, which is almost always what they want).
 */
@Injectable({ providedIn: 'root' })
export class ErrorService {
  private readonly errorSubject = new BehaviorSubject<ErrorContext | null>(null);
  readonly error$ = this.errorSubject.asObservable();

  get current(): ErrorContext | null {
    return this.errorSubject.value;
  }

  show(ctx: ErrorContext): void {
    this.errorSubject.next(ctx);
  }

  dismiss(): void {
    this.errorSubject.next(null);
  }

  /**
   * Format an error as a markdown block suitable for pasting into an
   * issue tracker or chat. Excludes empty fields so the output stays
   * compact.
   */
  formatForClipboard(ctx: ErrorContext): string {
    const lines: string[] = ['**VTSearch error**', ''];
    lines.push(`- **Message:** ${ctx.message}`);
    if (ctx.detail) lines.push(`- **Detail:** ${ctx.detail}`);
    lines.push(`- **Status:** ${ctx.status}${ctx.statusText ? ` ${ctx.statusText}` : ''}`);
    lines.push(`- **Endpoint:** \`${ctx.method} ${ctx.url}\``);
    if (ctx.requestId) lines.push(`- **Request ID:** \`${ctx.requestId}\``);
    if (ctx.datasetId) lines.push(`- **Dataset:** \`${ctx.datasetId}\``);
    if (ctx.detectorId) lines.push(`- **Detector:** \`${ctx.detectorId}\``);
    lines.push(`- **Timestamp:** ${ctx.timestamp}`);
    if (ctx.extra && Object.keys(ctx.extra).length > 0) {
      lines.push('');
      lines.push('**Extra fields:**');
      lines.push('```json');
      lines.push(JSON.stringify(ctx.extra, null, 2));
      lines.push('```');
    }
    if (ctx.rawBody) {
      lines.push('');
      lines.push('**Raw response body:**');
      lines.push('```');
      lines.push(ctx.rawBody);
      lines.push('```');
    }
    return lines.join('\n');
  }
}
