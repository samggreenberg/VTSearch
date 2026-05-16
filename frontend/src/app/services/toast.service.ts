import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { LoadingTask } from '../models/api.models';
import { ProgressEventsService } from './progress-events.service';

/**
 * Rich error context attached to HTTP-failure toasts. Carries everything
 * a user needs to file a useful bug report — the endpoint, status,
 * request_id, and the active dataset / detector at the time of failure.
 */
export interface ErrorContext {
  message: string;
  detail?: string;
  status: number;
  statusText?: string;
  method: string;
  url: string;
  requestId?: string;
  datasetId?: string;
  detectorId?: string;
  rawBody?: string;
  extra?: Record<string, unknown>;
  timestamp: string;
}

export interface Toast {
  id: number;
  message: string;
  detail?: string;
  /** Optional rich HTTP error context — enables the Details / Copy debug info actions. */
  errorContext?: ErrorContext;
  /**
   * Optional dedup key. Pushing a new toast with the same key replaces
   * the existing one (resetting its auto-dismiss timer). Used to avoid
   * piling up the same error during retry loops or repeating SSE events.
   */
  dedupKey?: string;
  timestamp: string;
}

interface ShowOptions {
  message: string;
  detail?: string;
  errorContext?: ErrorContext;
  dedupKey?: string;
}

const MAX_TOASTS = 5;

/**
 * Central toast sink for the frontend. Every error surface routes
 * through this service:
 *
 *  - HTTP failures (via ``errorInterceptor``) — emitted with full
 *    ``ErrorContext`` for the Details / Copy debug info actions.
 *  - SSE LoadingTask failures (via ``SseErrorRouterService``) — emitted
 *    deduped per task_id.
 *
 * Toasts stack (newest at the bottom) and stay until the user
 * dismisses them. Replaces the old single-banner ``ErrorService`` so a
 * second error doesn't silently clobber a first one the user hasn't
 * read yet.
 */
@Injectable({ providedIn: 'root' })
export class ToastService {
  private nextId = 1;
  private readonly toastsSubject = new BehaviorSubject<Toast[]>([]);
  readonly toasts$ = this.toastsSubject.asObservable();
  private readonly seenTaskKeys = new Set<string>();

  constructor(progressEvents: ProgressEventsService) {
    progressEvents.loadingTasks$.subscribe((tasks) => this.routeTaskErrors(tasks, 'dataset'));
    progressEvents.detectorLoadingTasks$.subscribe((tasks) => this.routeTaskErrors(tasks, 'detector'));
  }

  get toasts(): Toast[] {
    return this.toastsSubject.value;
  }

  private routeTaskErrors(tasks: LoadingTask[], kind: 'dataset' | 'detector'): void {
    for (const t of tasks) {
      if (t.status !== 'idle' || !t.error || t.error === 'Cancelled') continue;
      const key = `sse:${kind}:${t.task_id}`;
      if (this.seenTaskKeys.has(key)) continue;
      this.seenTaskKeys.add(key);
      this.error({
        message: kind === 'dataset' ? 'Dataset load failed' : 'Detector load failed',
        detail: `${t.name}: ${t.error}`,
        dedupKey: key,
      });
    }
  }

  error(opts: ShowOptions): number {
    const toast: Toast = {
      id: this.nextId++,
      message: opts.message,
      detail: opts.detail,
      errorContext: opts.errorContext,
      dedupKey: opts.dedupKey,
      timestamp: new Date().toISOString(),
    };

    let next = this.toastsSubject.value.slice();
    if (opts.dedupKey) {
      const existing = next.findIndex((t) => t.dedupKey === opts.dedupKey);
      if (existing >= 0) next.splice(existing, 1);
    }
    next.push(toast);
    while (next.length > MAX_TOASTS) next.shift();
    this.toastsSubject.next(next);
    return toast.id;
  }

  dismiss(id: number): void {
    const next = this.toastsSubject.value.filter((t) => t.id !== id);
    if (next.length !== this.toastsSubject.value.length) {
      this.toastsSubject.next(next);
    }
  }

  dismissAll(): void {
    this.toastsSubject.next([]);
  }

  /**
   * Format a toast as a markdown block suitable for pasting into an
   * issue tracker or chat. Toasts with an attached ``ErrorContext``
   * include the full HTTP debug bundle (endpoint, request_id, dataset
   * / detector, raw body, extra fields); plain toasts include just the
   * headline.
   */
  formatForClipboard(toast: Toast): string {
    const ctx = toast.errorContext;
    const lines: string[] = ['**VTSearch error**', ''];
    lines.push(`- **Message:** ${toast.message}`);
    if (toast.detail) lines.push(`- **Detail:** ${toast.detail}`);
    if (ctx) {
      lines.push(`- **Status:** ${ctx.status}${ctx.statusText ? ` ${ctx.statusText}` : ''}`);
      lines.push(`- **Endpoint:** \`${ctx.method} ${ctx.url}\``);
      if (ctx.requestId) lines.push(`- **Request ID:** \`${ctx.requestId}\``);
      if (ctx.datasetId) lines.push(`- **Dataset:** \`${ctx.datasetId}\``);
      if (ctx.detectorId) lines.push(`- **Detector:** \`${ctx.detectorId}\``);
    }
    lines.push(`- **Timestamp:** ${toast.timestamp}`);
    if (ctx?.extra && Object.keys(ctx.extra).length > 0) {
      lines.push('', '**Extra fields:**', '```json', JSON.stringify(ctx.extra, null, 2), '```');
    }
    if (ctx?.rawBody) {
      lines.push('', '**Raw response body:**', '```', ctx.rawBody, '```');
    }
    return lines.join('\n');
  }
}
