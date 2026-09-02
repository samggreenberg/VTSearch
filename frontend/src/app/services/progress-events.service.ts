import { Injectable, OnDestroy, computed, effect, inject, signal } from '@angular/core';
import { toObservable } from '@angular/core/rxjs-interop';
import { Observable, Subject } from 'rxjs';
import { filter, map, takeWhile } from 'rxjs/operators';
import {
  LoadingTask,
  ProgressEvent,
  ServerNotification,
  VotingIterationsResponse,
} from '../models/api.models';
import { ConnectionStateService } from './connection-state.service';

/**
 * Single EventSource subscription that fans out progress updates to every
 * consumer in the app. Replaces the old REST polling (timer(0, 500) +
 * /api/dataset/progress, etc.) with a server-push stream.
 *
 * Every channel carries the same `ProgressEvent` shape (see
 * `models/api.models.ts`) so any consumer can render any of them with the
 * shared `formatProgressMessage` helper in `utils/format-progress.ts`.
 *
 * Channels carried over `/api/events`:
 *
 *  - `server` -> { boot_id } (per-connect identity frame; a change in
 *    `boot_id` between connects means the backend restarted, which fires
 *    `serverReset$` so consumers can drop any state keyed on stale
 *    `task_id`s from the previous process)
 *  - `loading-tasks` -> LoadingTask[] (per-task progress for dataset loads)
 *  - `detector-loading-tasks` -> LoadingTask[]
 *  - `sort` -> ProgressEvent (text-sort)
 *  - `find` -> ProgressEvent (multi-dataset×detector /api/find)
 *  - `eval` -> ProgressEvent (train-and-score; consumers derive `done` themselves)
 *  - `notification` -> ServerNotification (one-off message from server-side
 *    code, typically a plugin reporting something it decided not to fail on;
 *    surfaced as a toast by `ToastService`)
 *
 * Every channel above except `notification` carries *state*: a snapshot the
 * backend re-sends on connect and on every heartbeat, so a dropped frame
 * heals itself. `notification` carries *events*, so it is exposed as a
 * `Subject` rather than a signal — there is no "current notification" to
 * read, only ones that arrive while you are listening.
 */
@Injectable({ providedIn: 'root' })
export class ProgressEventsService implements OnDestroy {
  private connection = inject(ConnectionStateService);

  // Canonical channel state as signals. A signal write inside the EventSource
  // callback notifies Angular's scheduler directly, so the SSE pump triggers
  // change detection on bound templates with no NgZone re-entry — correct under
  // both zone-based and zoneless change detection.
  private readonly _loadingTasks = signal<LoadingTask[]>([]);
  private readonly _detectorLoadingTasks = signal<LoadingTask[]>([]);
  private readonly _sort = signal<ProgressEvent>({});
  private readonly _find = signal<ProgressEvent>({});
  private readonly _eval = signal<ProgressEvent>({});

  // Read-only signal views: the canonical reads (also the synchronous
  // latest-value accessors — call them, e.g. `loadingTasks()`).
  readonly loadingTasks = this._loadingTasks.asReadonly();
  readonly detectorLoadingTasks = this._detectorLoadingTasks.asReadonly();
  readonly sort = this._sort.asReadonly();
  readonly find = this._find.asReadonly();

  // Observable bridges for the consumers that compose channel updates with
  // RxJS operators (takeUntil/filter/take/…). Each is a `toObservable` view of
  // the backing signal, so a signal write still drives them.
  readonly loadingTasks$ = toObservable(this._loadingTasks);
  readonly detectorLoadingTasks$ = toObservable(this._detectorLoadingTasks);
  readonly sort$ = toObservable(this._sort);
  readonly find$ = toObservable(this._find);
  readonly eval$ = toObservable(this._eval);

  /**
   * Fires whenever the backend's `boot_id` changes between successive
   * SSE connects (i.e. the backend restarted). Consumers that hold
   * `task_id`-keyed bookkeeping should drop it on this signal — those
   * ids no longer exist on the restarted backend. Does NOT fire on the
   * very first connect.
   */
  private readonly serverResetSubject = new Subject<void>();
  readonly serverReset$ = this.serverResetSubject.asObservable();

  /**
   * One-off messages pushed by server-side code (see `ServerNotification`).
   * A `Subject`, not a signal: these are events with no resting value, and a
   * late subscriber must not be handed a stale one to re-toast.
   */
  private readonly notificationsSubject = new Subject<ServerNotification>();
  readonly notifications$ = this.notificationsSubject.asObservable();

  private source: EventSource | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private lastBootId: string | null = null;

  constructor() {
    // Track the circuit breaker: stay connected while online, and tear the
    // EventSource down when the breaker trips so its native auto-reconnect
    // stops hammering a dead backend. An effect on the status signal performs
    // the initial connect (status starts `online`) and reacts to every later
    // flip; signals are distinct by default, so no `distinctUntilChanged` is
    // needed.
    effect(() => {
      if (this.connection.status() === 'offline') this.disconnect();
      else this.connect();
    });
  }

  ngOnDestroy(): void {
    this.disconnect();
  }

  // -------------------------------------------------------------------------
  // Derived eval progress. Components that briefly need a value outside an
  // RxJS pipeline read the signals/computeds directly (e.g. `loadingTasks()`).
  // -------------------------------------------------------------------------

  /** Derive the {progress,total,done} shape the eval modal cares about. */
  readonly votingIterations = computed<VotingIterationsResponse>(() => {
    const prog = this._eval();
    const current = Number(prog.current ?? 0);
    const total = Number(prog.total ?? 0);
    const status = String(prog.status ?? 'idle');
    const done = status === 'idle' && total > 0 && current >= total;
    return { progress: current, total, done };
  });

  readonly votingIterations$ = toObservable(this.votingIterations);

  /**
   * Snapshots of one detector-loading task, completing once it finishes.
   *
   * Emits every snapshot of `taskId` seen on the `detector-loading-tasks`
   * channel and completes on the terminal one (`status === 'idle'`, which
   * carries the task's `error` / `ingest_result`), so a caller can render a
   * live bar in `next` and take the follow-up step in `complete`. A task that
   * vanishes from the feed after being seen — the tracker prunes finished
   * entries — also completes the stream.
   *
   * Snapshots *before* the task appears are ignored rather than treated as
   * completion: the backend registers the task before answering the request
   * that returned its id, so the row is either already in the current
   * snapshot or one frame away.
   */
  detectorTaskUntilDone$(taskId: string): Observable<LoadingTask> {
    let seen = false;
    return this.detectorLoadingTasks$.pipe(
      map((tasks) => tasks.find((t) => t.task_id === taskId) ?? null),
      filter((task) => {
        if (task) seen = true;
        return seen;
      }),
      map((task) => task ?? ({ task_id: taskId, status: 'idle' } as LoadingTask)),
      takeWhile((task) => task.status !== 'idle', true),
    );
  }

  // -------------------------------------------------------------------------
  // EventSource lifecycle.
  // -------------------------------------------------------------------------

  private connect(): void {
    if (this.source) return;
    // EventSource fires events outside Angular's NgZone, but each handler
    // writes a signal (or calls a signalized service), which notifies the
    // change-detection scheduler directly — no zone re-entry needed.
    const es = new EventSource('/api/events');
    this.source = es;

    this.listen(es, 'server', (e) => this.handleServerFrame(e));
    this.listen(es, 'loading-tasks', (e) =>
      this._loadingTasks.set(this.parse<LoadingTask[]>(e, [])),
    );
    this.listen(es, 'detector-loading-tasks', (e) =>
      this._detectorLoadingTasks.set(this.parse<LoadingTask[]>(e, [])),
    );
    this.listen(es, 'sort', (e) => this._sort.set(this.parse<ProgressEvent>(e, {})));
    this.listen(es, 'find', (e) => this._find.set(this.parse<ProgressEvent>(e, {})));
    this.listen(es, 'eval', (e) => this._eval.set(this.parse<ProgressEvent>(e, {})));
    this.listen(es, 'notification', (e) => {
      const note = this.parse<ServerNotification | null>(e, null);
      // A frame that failed to parse, or one with nothing to say, would render
      // as an empty toast the user cannot act on — drop it instead.
      if (note?.message) this.notificationsSubject.next(note);
    });
    // The periodic `heartbeat` frame carries no payload we render; its sole
    // job is to keep the circuit breaker online (handled by `listen`'s
    // recordSuccess) during long, busy operations that aren't emitting
    // progress frames, so the backend never looks dead while it's merely busy.
    this.listen(es, 'heartbeat', () => {});

    es.onopen = () => this.connection.recordSuccess();

    es.onerror = () => {
      // Feed the circuit breaker indirectly: EventSource hides the HTTP
      // status, so this error may be a dead backend or a 503 slot-cap
      // rejection from a perfectly healthy one. recordStreamFailure()
      // probes /healthz to tell the two apart — a dashboard sitting on only
      // the stream still trips offline (via failed probes) when the backend
      // is really gone, but a cap rejection no longer locks the app (#2816).
      this.connection.recordStreamFailure();
      // EventSource reconnects on its own for transient failures, but
      // schedules an extra reconnect in case the server closed the stream
      // permanently. Idempotent: if `source` is already non-null the next
      // connect() call returns immediately.
      if (es.readyState === EventSource.CLOSED) {
        this.source = null;
        this.scheduleReconnect();
      }
    };
  }

  private disconnect(): void {
    if (this.reconnectTimer != null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer != null) return;
    // Don't queue a reconnect while the breaker is tripped — the status-signal
    // effect owns reconnection once we're back online.
    if (this.connection.isOffline) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  /**
   * Register a named-event listener that records the frame as proof the
   * backend is alive and then dispatches it. Every frame that arrives over the
   * stream — progress data or a bare `heartbeat` — resets the connection
   * circuit breaker (a signal write), so it only trips offline when the stream
   * genuinely goes silent (a dead backend), not when a busy backend is slow to
   * answer unrelated background pollers.
   */
  private listen(es: EventSource, name: string, handler: (e: Event) => void): void {
    es.addEventListener(name, (e) => {
      this.connection.recordSuccess();
      handler(e);
    });
  }

  private parse<T>(e: Event, fallback: T): T {
    const msg = e as MessageEvent<string>;
    try {
      return JSON.parse(msg.data) as T;
    } catch {
      return fallback;
    }
  }

  private handleServerFrame(e: Event): void {
    const { boot_id } = this.parse<{ boot_id?: string }>(e, {});
    if (!boot_id) return;
    if (this.lastBootId !== null && this.lastBootId !== boot_id) {
      this.serverResetSubject.next();
    }
    this.lastBootId = boot_id;
  }
}
