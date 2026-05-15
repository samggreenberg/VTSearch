import { Injectable, OnDestroy, NgZone } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import {
  DatasetProgress,
  LoadingTask,
  SortProgressResponse,
  VotingIterationsResponse,
} from '../models/api.models';

/**
 * Single EventSource subscription that fans out progress updates to every
 * consumer in the app. Replaces the old REST polling (timer(0, 500) +
 * /api/dataset/progress, etc.) with a server-push stream — see
 * `docs/plans/feature-brainstorm.md` §12.5.
 *
 * Channels carried over `/api/events`:
 *
 *  - `dataset` -> DatasetProgress (singleton tracker; staging operations)
 *  - `loading-tasks` -> LoadingTask[]
 *  - `detector-loading-tasks` -> LoadingTask[]
 *  - `sort` -> SortProgressResponse
 *  - `find` -> generic progress dict (used by /api/find)
 *  - `eval` -> singleton tracker dict (raw fields, not the
 *    {progress,total,done} shape the old `/api/eval/voting-iterations`
 *    endpoint returned — consumers derive `done` themselves)
 */
@Injectable({ providedIn: 'root' })
export class ProgressEventsService implements OnDestroy {
  private readonly datasetSubject = new BehaviorSubject<DatasetProgress>({});
  private readonly loadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly detectorLoadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly sortSubject = new BehaviorSubject<SortProgressResponse>({});
  private readonly findSubject = new BehaviorSubject<Record<string, unknown>>({});
  private readonly evalSubject = new BehaviorSubject<Record<string, unknown>>({});

  readonly dataset$ = this.datasetSubject.asObservable();
  readonly loadingTasks$ = this.loadingTasksSubject.asObservable();
  readonly detectorLoadingTasks$ = this.detectorLoadingTasksSubject.asObservable();
  readonly sort$ = this.sortSubject.asObservable();
  readonly find$ = this.findSubject.asObservable();
  readonly eval$ = this.evalSubject.asObservable();

  private source: EventSource | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private zone: NgZone) {
    this.connect();
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

  get find(): Record<string, unknown> {
    return this.findSubject.value;
  }

  /** Derive the {progress,total,done} shape the eval modal cares about. */
  get votingIterations(): VotingIterationsResponse {
    const prog = this.evalSubject.value;
    const current = Number(prog['current'] ?? 0);
    const total = Number(prog['total'] ?? 0);
    const status = String(prog['status'] ?? 'idle');
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

    es.addEventListener('dataset', (e) =>
      this.zone.run(() => this.datasetSubject.next(this.parse<DatasetProgress>(e, {}))),
    );
    es.addEventListener('loading-tasks', (e) =>
      this.zone.run(() => this.loadingTasksSubject.next(this.parse<LoadingTask[]>(e, []))),
    );
    es.addEventListener('detector-loading-tasks', (e) =>
      this.zone.run(() => this.detectorLoadingTasksSubject.next(this.parse<LoadingTask[]>(e, []))),
    );
    es.addEventListener('sort', (e) =>
      this.zone.run(() => this.sortSubject.next(this.parse<SortProgressResponse>(e, {}))),
    );
    es.addEventListener('find', (e) =>
      this.zone.run(() => this.findSubject.next(this.parse<Record<string, unknown>>(e, {}))),
    );
    es.addEventListener('eval', (e) =>
      this.zone.run(() => this.evalSubject.next(this.parse<Record<string, unknown>>(e, {}))),
    );

    es.onerror = () => {
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
}
