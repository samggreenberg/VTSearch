import { Injectable, inject } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { LoadingTask, ServerNotification } from '../models/api.models';
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

/**
 * Toast severities, in descending order of how hard they insist.
 *
 * `error` and `warning` stay on screen until the user dismisses them: both
 * report that something did not go as asked, and a message the user missed is
 * the same as a message never sent. `success` and `info` auto-dismiss — they
 * confirm or narrate, and nothing is lost if they scroll past unread.
 */
export type ToastLevel = 'error' | 'warning' | 'success' | 'info';

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

/**
 * A visible "this is about to happen" timer on a toast. The toast renders
 * ``<label> <remaining>…`` and ticks once a second; on reaching zero it
 * dismisses itself and runs ``onExpire``.
 *
 * Dismissing the toast first — via its action button, its close button, or
 * ``dismiss``/``dismissAll`` — cancels the countdown, and ``onExpire`` never
 * runs. That is what makes the countdown escapable: every way of getting rid
 * of the toast is also a way of calling off the pending action.
 */
export interface ToastCountdown {
  /** Lead-in text, e.g. ``'Taking you back to the Dashboard in'``. */
  label: string;
  /** Seconds left. Ticks down to zero. */
  remaining: number;
  /** Seconds the countdown started from (for progress rendering). */
  total: number;
  /** Ran once when the countdown reaches zero. */
  onExpire: () => void;
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
  /** Optional escapable countdown (see :class:`ToastCountdown`). */
  countdown?: ToastCountdown;
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
  /**
   * Attach an escapable countdown. Supplying this suppresses the plain
   * auto-dismiss timer — the countdown owns the toast's lifetime instead, so
   * the seconds the user sees are the seconds they actually have.
   */
  countdown?: { label: string; seconds: number; onExpire: () => void };
  /**
   * Override how long a success toast stays up, in milliseconds; ``0`` keeps
   * it until the user dismisses it. Left unset, success toasts auto-dismiss
   * after ``SUCCESS_AUTO_DISMISS_MS``. Set this when the toast's
   * {@link ToastAction} is the user's only route to something they asked for
   * (e.g. a browser tab the popup blocker refused), since a timed-out toast
   * takes that route with it.
   */
  autoDismissMs?: number;
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
 *  - Backend notifications (the ``notification`` SSE channel): one-off
 *    messages any server-side code — most often a plugin that hit a
 *    recoverable problem and chose to continue — pushed with ``notify()``.
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
  private readonly countdownTimers = new Map<number, ReturnType<typeof setInterval>>();

  constructor() {
    const progressEvents = inject(ProgressEventsService);

    progressEvents.loadingTasks$.subscribe((tasks) => this.routeTaskErrors(tasks, 'dataset'));
    progressEvents.detectorLoadingTasks$.subscribe((tasks) => this.routeTaskErrors(tasks, 'detector'));
    progressEvents.notifications$.subscribe((note) => this.routeServerNotification(note));
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

  /**
   * Render a notification pushed by the backend (`notify()` in
   * `vtscore/concurrency/notifications.py`), typically from a plugin that hit
   * a recoverable problem and kept going.
   *
   * The `source` is folded into the detail line rather than the headline: the
   * headline is the plugin's own sentence and should read as written, while
   * "which part of the app is telling me this" is context.
   */
  private routeServerNotification(note: ServerNotification): void {
    const detail = [note.source, note.detail].filter(Boolean).join(' — ') || undefined;
    this.show(note.level, { message: note.message, detail, dedupKey: `server:${note.id}` });
  }

  error(opts: ShowOptions): number {
    return this.push('error', opts);
  }

  /**
   * Something went wrong but the operation carried on with a partial or
   * degraded result. Stays up until dismissed, like {@link error}: the user
   * is holding a result that is not quite what they asked for, and needs to
   * know before they act on it.
   */
  warning(opts: ShowOptions): number {
    return this.push('warning', opts);
  }

  /**
   * Neutral news — "this happened, nothing is wrong". Auto-dismisses on the
   * same timer as {@link success}.
   */
  info(opts: ShowOptions): number {
    const autoDismissMs = opts.autoDismissMs ?? SUCCESS_AUTO_DISMISS_MS;
    return this.push('info', opts, opts.countdown ? 0 : autoDismissMs);
  }

  /** Dispatch to the per-level method for a level only known at runtime. */
  show(level: ToastLevel, opts: ShowOptions): number {
    switch (level) {
      case 'error':
        return this.error(opts);
      case 'warning':
        return this.warning(opts);
      case 'success':
        return this.success(opts);
      default:
        return this.info(opts);
    }
  }

  /**
   * Non-blocking success notification. Auto-dismisses after
   * ``SUCCESS_AUTO_DISMISS_MS`` so the user is not forced to click
   * through a modal for "X is done" style messages, unless the caller
   * overrides that with ``autoDismissMs`` (``0`` = stays until dismissed).
   */
  success(opts: ShowOptions): number {
    const autoDismissMs = opts.autoDismissMs ?? SUCCESS_AUTO_DISMISS_MS;
    return this.push('success', opts, opts.countdown ? 0 : autoDismissMs);
  }

  private push(level: ToastLevel, opts: ShowOptions, autoDismissMs = 0): number {
    const cd = opts.countdown;
    const toast: Toast = {
      id: this.nextId++,
      level,
      message: opts.message,
      detail: opts.detail,
      errorContext: opts.errorContext,
      action: opts.action,
      countdown: cd
        ? { label: cd.label, remaining: cd.seconds, total: cd.seconds, onExpire: cd.onExpire }
        : undefined,
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
    if (toast.countdown) {
      this.countdownTimers.set(
        toast.id,
        setInterval(() => this.tickCountdown(toast.id), 1000),
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
    for (const id of this.countdownTimers.keys()) this.clearTimer(id);
    this.toastsSubject.next([]);
  }

  /**
   * Call off a toast's pending countdown *without* dismissing the toast.
   *
   * The countdown line and the escape action disappear and ``onExpire`` never
   * runs; what stays is an ordinary notification carrying the original
   * headline, which then auto-dismisses on the usual timer. Pass ``note`` to
   * replace the detail line with an explanation of why the pending action was
   * called off.
   *
   * This is the hook for cancelling from *outside* the toast: the countdown's
   * own buttons already cancel by dismissing, but a caller that watches the
   * page behind the toast (e.g. "the user started clicking things, so don't
   * navigate away from under them") wants to drop the timer while leaving the
   * message the user may not have read yet.
   */
  cancelCountdown(id: number, note?: string): void {
    const list = this.toastsSubject.value;
    const idx = list.findIndex((t) => t.id === id);
    if (idx < 0 || !list[idx].countdown) return;
    this.clearTimer(id);
    const next = list.slice();
    next[idx] = {
      ...list[idx],
      countdown: undefined,
      action: undefined,
      detail: note ?? list[idx].detail,
    };
    this.toastsSubject.next(next);
    this.autoDismissTimers.set(
      id,
      setTimeout(() => this.dismiss(id), SUCCESS_AUTO_DISMISS_MS),
    );
  }

  /**
   * Advance one countdown by a second. Emits a replacement toast object each
   * tick so ``OnPush`` renderers repaint the remaining seconds. At zero the
   * toast is dismissed *before* ``onExpire`` runs, so a handler that navigates
   * away never leaves a stale countdown on screen.
   */
  private tickCountdown(id: number): void {
    const list = this.toastsSubject.value;
    const idx = list.findIndex((t) => t.id === id);
    const countdown = idx >= 0 ? list[idx].countdown : undefined;
    if (!countdown) {
      // Toast already gone (dismissed by the user, or evicted by MAX_TOASTS):
      // the pending action is cancelled along with it.
      this.clearTimer(id);
      return;
    }

    const remaining = countdown.remaining - 1;
    if (remaining > 0) {
      const next = list.slice();
      next[idx] = { ...list[idx], countdown: { ...countdown, remaining } };
      this.toastsSubject.next(next);
      return;
    }

    this.dismiss(id);
    countdown.onExpire();
  }

  private clearTimer(id: number): void {
    const t = this.autoDismissTimers.get(id);
    if (t !== undefined) {
      clearTimeout(t);
      this.autoDismissTimers.delete(id);
    }
    const c = this.countdownTimers.get(id);
    if (c !== undefined) {
      clearInterval(c);
      this.countdownTimers.delete(id);
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
