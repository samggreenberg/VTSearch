import { WritableSignal, signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ProgressEventsService } from './progress-events.service';
import { ConnectionStateService, ConnectionStatus } from './connection-state.service';

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
  let status: WritableSignal<ConnectionStatus>;
  let originalEventSource: typeof EventSource;

  beforeEach(() => {
    FakeEventSource.instances = [];
    originalEventSource = globalThis.EventSource;
    globalThis.EventSource = FakeEventSource as unknown as typeof EventSource;

    recordSuccess = vi.fn();
    status = signal<ConnectionStatus>('online');
    const connectionStub = {
      status,
      recordSuccess,
      recordNetworkFailure: vi.fn(),
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
});
