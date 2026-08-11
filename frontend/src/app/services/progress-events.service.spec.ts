import { Component, WritableSignal, inject, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ProgressEventsService } from './progress-events.service';
import { ConnectionStateService, ConnectionStatus } from './connection-state.service';
import type { LoadingTask } from '../models/api.models';
import { configureZoneless } from '../testing/zoneless-testbed';
import { settleZoneless } from '../testing/settle-resource';

/**
 * Minimal stand-in for the browser `EventSource` (jsdom has none). Captures
 * the named-event listeners the service registers so the test can synthesise
 * server frames and assert how the service reacts.
 */
class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static readonly CLOSED = 2;

  readyState = 0;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private readonly listeners = new Map<string, ((e: MessageEvent) => void)[]>();

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(name: string, fn: (e: MessageEvent) => void): void {
    const list = this.listeners.get(name) ?? [];
    list.push(fn);
    this.listeners.set(name, list);
  }

  close(): void {
    this.readyState = FakeEventSource.CLOSED;
  }

  /** Test helper: deliver a frame on a named channel. */
  emit(name: string, data: unknown): void {
    const event = { data: JSON.stringify(data) } as MessageEvent;
    for (const fn of this.listeners.get(name) ?? []) fn(event);
  }
}

describe('ProgressEventsService liveness wiring', () => {
  let recordSuccess: ReturnType<typeof vi.fn>;
  let recordNetworkFailure: ReturnType<typeof vi.fn>;
  let recordStreamFailure: ReturnType<typeof vi.fn>;
  let status: WritableSignal<ConnectionStatus>;
  let originalEventSource: typeof EventSource;

  beforeEach(() => {
    FakeEventSource.instances = [];
    originalEventSource = globalThis.EventSource;
    globalThis.EventSource = FakeEventSource as unknown as typeof EventSource;

    recordSuccess = vi.fn();
    recordNetworkFailure = vi.fn();
    recordStreamFailure = vi.fn();
    status = signal<ConnectionStatus>('online');
    const connectionStub = {
      status,
      recordSuccess,
      recordNetworkFailure,
      recordStreamFailure,
      get isOffline() {
        return status() === 'offline';
      },
    };

    TestBed.configureTestingModule({
      providers: [
        { provide: ConnectionStateService, useValue: connectionStub as unknown as ConnectionStateService },
      ],
    });
  });

  afterEach(() => {
    globalThis.EventSource = originalEventSource;
  });

  function connectedSource(): FakeEventSource {
    TestBed.inject(ProgressEventsService);
    // The connect() side effect now runs inside an effect on the status signal,
    // so flush it before reading the EventSource the service opened.
    TestBed.tick();
    return FakeEventSource.instances[0];
  }

  it('connects to /api/events while online', () => {
    const es = connectedSource();
    expect(es.url).toBe('/api/events');
  });

  it('treats a heartbeat frame as proof the backend is alive', () => {
    const es = connectedSource();
    recordSuccess.mockClear();
    es.emit('heartbeat', { ts: 123 });
    expect(recordSuccess).toHaveBeenCalledTimes(1);
  });

  it('treats a real progress frame as proof of life too', () => {
    const es = connectedSource();
    recordSuccess.mockClear();
    es.emit('dataset', { status: 'loading', current: 5, total: 10 });
    expect(recordSuccess).toHaveBeenCalledTimes(1);
  });

  it('asks the connection service to classify an SSE error, never counting it as a network failure outright (#2816)', () => {
    // EventSource hides HTTP statuses: the error may be a 503 slot-cap
    // rejection from a live backend. recordStreamFailure() probes /healthz
    // to tell; a direct recordNetworkFailure() here is exactly the bug that
    // locked the app offline behind a manual Retry.
    const es = connectedSource();
    es.readyState = FakeEventSource.CLOSED;
    es.onerror!();
    expect(recordStreamFailure).toHaveBeenCalledTimes(1);
    expect(recordNetworkFailure).not.toHaveBeenCalled();
    // Tear down the 2s reconnect timer the error handler scheduled.
    status.set('offline');
    TestBed.tick();
  });

  // --- detectorTaskUntilDone$: awaiting one task's terminal frame (#2703) ---

  /** Subscribe to `detectorTaskUntilDone$` and record what it emits. */
  function watchTask(taskId: string) {
    const seen: LoadingTask[] = [];
    const state = { completed: false };
    TestBed.inject(ProgressEventsService)
      .detectorTaskUntilDone$(taskId)
      .subscribe({
        next: (task) => seen.push(task),
        complete: () => {
          state.completed = true;
        },
      });
    return { seen, state };
  }

  it('emits only the named task and completes on its terminal frame', () => {
    const es = connectedSource();
    const { seen, state } = watchTask('_detingest_d9');
    TestBed.tick();

    es.emit('detector-loading-tasks', [{ task_id: '_detload_other', status: 'loading' }]);
    TestBed.tick();
    expect(seen).toEqual([]);

    es.emit('detector-loading-tasks', [
      { task_id: '_detingest_d9', status: 'loading', current: 2, total: 5 },
    ]);
    TestBed.tick();
    expect(seen.map((t) => t.current)).toEqual([2]);
    expect(state.completed).toBe(false);

    es.emit('detector-loading-tasks', [
      { task_id: '_detingest_d9', status: 'idle', ingest_result: { ingested: 5 } },
    ]);
    TestBed.tick();
    expect(seen.at(-1)?.ingest_result).toEqual({ ingested: 5 });
    expect(state.completed).toBe(true);
  });

  it('completes when a task it has seen is pruned from the feed', () => {
    const es = connectedSource();
    const { state } = watchTask('_detingest_d9');
    TestBed.tick();

    es.emit('detector-loading-tasks', [{ task_id: '_detingest_d9', status: 'loading' }]);
    TestBed.tick();
    expect(state.completed).toBe(false);

    // The tracker drops finished tasks after their stale-prune window.
    es.emit('detector-loading-tasks', []);
    TestBed.tick();
    expect(state.completed).toBe(true);
  });

  it('keeps waiting while the task has not appeared yet', () => {
    const es = connectedSource();
    const { seen, state } = watchTask('_detingest_d9');
    TestBed.tick();

    // The backend registers the task before answering the request that hands
    // back its id, so an empty feed here just means the frame is in flight.
    es.emit('detector-loading-tasks', []);
    TestBed.tick();
    expect(seen).toEqual([]);
    expect(state.completed).toBe(false);
  });
});

/**
 * Zoneless staleness canary. The six SSE channels are signals, so a frame
 * arriving on the EventSource (which fires *outside* Angular's NgZone)
 * schedules change detection via the signal write — no `zone.run` re-entry.
 * This component reads the `dataset` channel signal in its template; driving
 * a frame through the fake EventSource and asserting the rendered DOM after
 * `settleZoneless()`, with no manual `detectChanges()`, proves the SSE pump
 * repaints bound views under zoneless.
 */
@Component({
  selector: 'app-progress-canary',
  standalone: true,
  template: `<span class="status">{{ progress.dataset().status ?? 'none' }}</span>`,
})
class ProgressCanaryComponent {
  readonly progress = inject(ProgressEventsService);
}

describe('ProgressEventsService (zoneless view canary)', () => {
  let fixture: ComponentFixture<ProgressCanaryComponent>;
  let originalEventSource: typeof EventSource;
  let status: WritableSignal<ConnectionStatus>;

  beforeEach(async () => {
    FakeEventSource.instances = [];
    originalEventSource = globalThis.EventSource;
    globalThis.EventSource = FakeEventSource as unknown as typeof EventSource;

    status = signal<ConnectionStatus>('online');
    const connectionStub = {
      status,
      recordSuccess: vi.fn(),
      recordNetworkFailure: vi.fn(),
      recordStreamFailure: vi.fn(),
      get isOffline() {
        return status() === 'offline';
      },
    };

    configureZoneless({
      imports: [ProgressCanaryComponent],
      providers: [
        ProgressEventsService,
        { provide: ConnectionStateService, useValue: connectionStub as unknown as ConnectionStateService },
      ],
    });
    fixture = TestBed.createComponent(ProgressCanaryComponent);
    // Constructing the component injects the service; its status-signal effect
    // opens the EventSource on settle.
    await settleZoneless(fixture);
  });

  afterEach(() => {
    globalThis.EventSource = originalEventSource;
  });

  function statusText(): string | null {
    return fixture.nativeElement.querySelector('.status')?.textContent?.trim() ?? null;
  }

  it('renders the empty initial channel state', () => {
    expect(statusText()).toBe('none');
  });

  it('repaints when an SSE frame lands, with no manual detectChanges', async () => {
    const es = FakeEventSource.instances[0];
    es.emit('dataset', { status: 'loading', current: 5, total: 10 });
    await settleZoneless(fixture);
    expect(statusText()).toBe('loading');

    es.emit('dataset', { status: 'idle' });
    await settleZoneless(fixture);
    expect(statusText()).toBe('idle');
  });
});
