import { Injectable, OnDestroy, NgZone } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { distinctUntilChanged } from 'rxjs/operators';
import {
  LoadingTask,
  ProgressEvent,
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
 *  - `dataset` -> ProgressEvent (singleton tracker; staging operations)
 *  - `loading-tasks` -> LoadingTask[] (per-task progress for dataset loads)
 *  - `detector-loading-tasks` -> LoadingTask[]
 *  - `sort` -> ProgressEvent (text-sort)
 *  - `find` -> ProgressEvent (multi-dataset×detector /api/find)
 *  - `eval` -> ProgressEvent (train-and-score; consumers derive `done` themselves)
 */
@Injectable({ providedIn: 'root' })
export class ProgressEventsService implements OnDestroy {
  private readonly datasetSubject = new BehaviorSubject<ProgressEvent>({});
  private readonly loadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly detectorLoadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly sortSubject = new BehaviorSubject<ProgressEvent>({});
  private readonly findSubject = new BehaviorSubject<ProgressEvent>({});
  private readonly evalSubject = new BehaviorSubject<ProgressEvent>({});

  readonly dataset$ = this.datasetSubject.asObservable();
  readonly loadingTasks$ = this.loadingTasksSubject.asObservable();
  readonly detectorLoadingTasks$ = this.detectorLoadingTasksSubject.asObservable();
  readonly sort$ = this.sortSubject.asObservable();
  readonly find$ = this.findSubject.asObservable();
  readonly eval$ = this.evalSubject.asObservable();

  /**
   * Fires whenever the backend's `boot_id` changes between successive
   * SSE connects (i.e. the backend restarted). Consumers that hold
   * `task_id`-keyed bookkeeping should drop it on this signal — those
   * ids no longer exist on the restarted backend. Does NOT fire on the
   * very first connect.
   */
  private readonly serverResetSubject = new Subject<void>();
  readonly serverReset$ = this.serverResetSubject.asObservable();

  private source: EventSource | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private lastBootId: string | null = null;

  constructor(
    private zone: NgZone,
    private connection: ConnectionStateService,
  ) {
    // Track the circuit breaker: stay connected while online, and tear the
    // EventSource down when the breaker trips so its native auto-reconnect
    // stops hammering a dead backend. The BehaviorSubject replays the current
    // status, so this also performs the initial connect.
    this.connection.status$
      .pipe(distinctUntilChanged())
      .subscribe((status) => {
        if (status === 'offline') this.disconnect();
        else this.connect();
      });
  }

  ngOnDestroy(): void {
    this.disconnect();
  }

  // -------------------------------------------------------------------------
  // Latest-value accessors. Components that briefly need a value outside an
  // RxJS pipeline (e.g. inside an imperative callback after a POST) use
  // these instead of re-subscribing.
  // -------------------------------------------------------------------------

  get loadingTasks(): LoadingTask[] {
    return this.loadingTasksSubject.value;
  }

  get detectorLoadingTasks(): LoadingTask[] {
    return this.detectorLoadingTasksSubject.value;
  }

  get find(): ProgressEvent {
    return this.findSubject.value;
  }

  /** Derive the {progress,total,done} shape the eval modal cares about. */
  get votingIterations(): VotingIterationsResponse {
    const prog = this.evalSubject.value;
    const current = Number(prog.current ?? 0);
    const total = Number(prog.total ?? 0);
    const status = String(prog.status ?? 'idle');
    const done = status === 'idle' && total > 0 && current >= total;
    return { progress: current, total, done };
  }

  readonly votingIterations$ = new Observable<VotingIterationsResponse>((sub) => {
    const inner = this.eval$.subscribe(() => sub.next(this.votingIterations));
    return () => inner.unsubscribe();
  });

  // -------------------------------------------------------------------------
  // EventSource lifecycle.
  // -------------------------------------------------------------------------

  private connect(): void {
    if (this.source) return;
    // EventSource fires events outside Angular's NgZone, so manually
    // re-enter the zone for every subject.next() to trigger change
    // detection on bound templates.
    const es = new EventSource('/api/events');
    this.source = es;

    es.addEventListener('server', (e) =>
      this.zone.run(() => this.handleServerFrame(e)),
    );
    es.addEventListener('dataset', (e) =>
      this.zone.run(() => this.datasetSubject.next(this.parse<ProgressEvent>(e, {}))),
    );
    es.addEventListener('loading-tasks', (e) =>
      this.zone.run(() => this.loadingTasksSubject.next(this.parse<LoadingTask[]>(e, []))),
    );
    es.addEventListener('detector-loading-tasks', (e) =>
      this.zone.run(() => this.detectorLoadingTasksSubject.next(this.parse<LoadingTask[]>(e, []))),
    );
    es.addEventListener('sort', (e) =>
      this.zone.run(() => this.sortSubject.next(this.parse<ProgressEvent>(e, {}))),
    );
    es.addEventListener('find', (e) =>
      this.zone.run(() => this.findSubject.next(this.parse<ProgressEvent>(e, {}))),
    );
    es.addEventListener('eval', (e) =>
      this.zone.run(() => this.evalSubject.next(this.parse<ProgressEvent>(e, {}))),
    );

    es.onopen = () => this.zone.run(() => this.connection.recordSuccess());

    es.onerror = () => {
      // Feed the circuit breaker: an SSE error is a connectivity signal too,
      // so a dashboard sitting on only the stream still trips offline.
      this.connection.recordNetworkFailure();
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
    // Don't queue a reconnect while the breaker is tripped — the status$
    // subscription owns reconnection once we're back online.
    if (this.connection.isOffline) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
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
