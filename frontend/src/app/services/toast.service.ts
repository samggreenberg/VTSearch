import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { LoadingTask } from '../models/api.models';
import { ProgressEventsService } from './progress-events.service';

/**
 * Rich error context attached to HTTP-failure toasts. Carries everything
 * a user needs to file a useful bug report: the endpoint, status,
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

export type ToastLevel = 'error' | 'success';

/**
 * Optional action button rendered next to the toast's dismiss. Used by
 * non-error toasts to offer a one-click follow-up ("Save as default",
 * "Undo", etc). Clicking the button auto-dismisses the toast.
 */
export interface ToastAction {
  label: string;
  /** Tooltip / aria-label for the button. */
  title?: string;
  onClick: () => void;
}

export interface Toast {
  id: number;
  level: ToastLevel;
  message: string;
  detail?: string;
  /** Optional rich HTTP error context; enables the Details / Copy debug info actions. */
  errorContext?: ErrorContext;
  /** Optional follow-up action button (see :class:`ToastAction`). */
  action?: ToastAction;
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
  action?: ToastAction;
  dedupKey?: string;
}

const MAX_TOASTS = 5;
const SUCCESS_AUTO_DISMISS_MS = 5000;

/**
 * Central toast sink for the frontend. Every error surface routes
 * through this service:
 *
 *  - HTTP failures (via ``errorInterceptor``): emitted with full
 *    ``ErrorContext`` for the Details / Copy debug info actions.
 *  - SSE LoadingTask failures (via ``SseErrorRouterService``): emitted
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
  private readonly autoDismissTimers = new Map<number, ReturnType<typeof setTimeout>>();

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
    return this.push('error', opts);
  }

  /**
   * Non-blocking success notification. Auto-dismisses after
   * ``SUCCESS_AUTO_DISMISS_MS`` so the user is not forced to click
   * through a modal for "X is done" style messages.
   */
  success(opts: ShowOptions): number {
    return this.push('success', opts, SUCCESS_AUTO_DISMISS_MS);
  }

  private push(level: ToastLevel, opts: ShowOptions, autoDismissMs = 0): number {
    const toast: Toast = {
      id: this.nextId++,
      level,
      message: opts.message,
      detail: opts.detail,
      errorContext: opts.errorContext,
      action: opts.action,
      dedupKey: opts.dedupKey,
      timestamp: new Date().toISOString(),
    };

    let next = this.toastsSubject.value.slice();
    if (opts.dedupKey) {
      const existing = next.findIndex((t) => t.dedupKey === opts.dedupKey);
      if (existing >= 0) {
        const evicted = next[existing];
        this.clearTimer(evicted.id);
        next.splice(existing, 1);
      }
    }
    next.push(toast);
    while (next.length > MAX_TOASTS) {
      const dropped = next.shift();
      if (dropped) this.clearTimer(dropped.id);
    }
    this.toastsSubject.next(next);

    if (autoDismissMs > 0) {
      this.autoDismissTimers.set(
        toast.id,
        setTimeout(() => this.dismiss(toast.id), autoDismissMs),
      );
    }
    return toast.id;
  }

  dismiss(id: number): void {
    this.clearTimer(id);
    const next = this.toastsSubject.value.filter((t) => t.id !== id);
    if (next.length !== this.toastsSubject.value.length) {
      this.toastsSubject.next(next);
    }
  }

  dismissAll(): void {
    for (const id of this.autoDismissTimers.keys()) this.clearTimer(id);
    this.toastsSubject.next([]);
  }

  private clearTimer(id: number): void {
    const t = this.autoDismissTimers.get(id);
    if (t !== undefined) {
      clearTimeout(t);
      this.autoDismissTimers.delete(id);
    }
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
