import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { ErrorContext, Toast, ToastService } from './toast.service';
import { ProgressEventsService } from './progress-events.service';
import { LoadingTask, ServerNotification } from '../models/api.models';

describe('ToastService', () => {
  let service: ToastService;
  let loadingTasks$: Subject<LoadingTask[]>;
  let detectorLoadingTasks$: Subject<LoadingTask[]>;
  let notifications$: Subject<ServerNotification>;

  function task(overrides: Partial<LoadingTask>): LoadingTask {
    return {
      status: 'idle',
      message: '',
      current: 0,
      total: 0,
      task_id: 't',
      name: 'n',
      created_at: 0,
      ...overrides,
    };
  }

  beforeEach(() => {
    loadingTasks$ = new Subject<LoadingTask[]>();
    detectorLoadingTasks$ = new Subject<LoadingTask[]>();
    notifications$ = new Subject<ServerNotification>();
    TestBed.configureTestingModule({
      providers: [
        ToastService,
        {
          provide: ProgressEventsService,
          useValue: { loadingTasks$, detectorLoadingTasks$, notifications$ },
        },
      ],
    });
    service = TestBed.inject(ToastService);
  });

  it('starts with no toasts', () => {
    expect(service.toasts).toEqual([]);
  });

  it('error() pushes an error toast and returns its id', () => {
    const id = service.error({ message: 'boom' });
    expect(service.toasts.length).toBe(1);
    const toast = service.toasts[0];
    expect(toast.id).toBe(id);
    expect(toast.level).toBe('error');
    expect(toast.message).toBe('boom');
    expect(toast.timestamp).toBeTruthy();
  });

  it('assigns increasing ids to successive toasts', () => {
    const a = service.error({ message: 'a' });
    const b = service.error({ message: 'b' });
    expect(b).toBeGreaterThan(a);
  });

  it('success() auto-dismisses after the timeout', async () => {
    vi.useFakeTimers();
    try {
      service.success({ message: 'saved' });
      expect(service.toasts.length).toBe(1);

      await vi.advanceTimersByTimeAsync(4999);
      expect(service.toasts.length).toBe(1);

      await vi.advanceTimersByTimeAsync(1);
      expect(service.toasts.length).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('error() toasts persist past the success auto-dismiss window', async () => {
    vi.useFakeTimers();
    try {
      service.error({ message: 'stays' });
      await vi.advanceTimersByTimeAsync(60000);
      expect(service.toasts.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('dedupKey replaces the existing toast instead of stacking', () => {
    service.error({ message: 'first', dedupKey: 'k' });
    service.error({ message: 'second', dedupKey: 'k' });
    expect(service.toasts.length).toBe(1);
    expect(service.toasts[0].message).toBe('second');
  });

  it('caps the stack at MAX_TOASTS, dropping the oldest', () => {
    for (let i = 0; i < 7; i++) service.error({ message: `m${i}` });
    expect(service.toasts.length).toBe(5);
    // m0 and m1 were shifted off the front; m2..m6 remain.
    expect(service.toasts.map((t) => t.message)).toEqual(['m2', 'm3', 'm4', 'm5', 'm6']);
  });

  it('dismiss() removes only the matching toast', () => {
    const a = service.error({ message: 'a' });
    service.error({ message: 'b' });
    service.dismiss(a);
    expect(service.toasts.map((t) => t.message)).toEqual(['b']);
  });

  it('dismiss() of an unknown id is a no-op (no re-emit)', () => {
    service.error({ message: 'a' });
    let emissions = 0;
    service.toasts$.subscribe(() => emissions++);
    emissions = 0; // ignore the initial BehaviorSubject replay
    service.dismiss(9999);
    expect(emissions).toBe(0);
    expect(service.toasts.length).toBe(1);
  });

  it('dismissAll() clears everything', () => {
    service.error({ message: 'a' });
    service.error({ message: 'b' });
    service.dismissAll();
    expect(service.toasts).toEqual([]);
  });

  it('routes an SSE dataset-load failure to a deduped error toast', () => {
    loadingTasks$.next([task({ task_id: 'x', status: 'idle', error: 'disk full', name: 'MyData' })]);
    expect(service.toasts.length).toBe(1);
    expect(service.toasts[0].message).toBe('Dataset load failed');
    expect(service.toasts[0].detail).toBe('MyData: disk full');
  });

  it('routes an SSE detector-load failure with the detector wording', () => {
    detectorLoadingTasks$.next([task({ task_id: 'y', status: 'idle', error: 'oom', name: 'Det' })]);
    expect(service.toasts.length).toBe(1);
    expect(service.toasts[0].message).toBe('Detector load failed');
  });

  it('does not re-toast the same failed task_id twice', () => {
    loadingTasks$.next([task({ task_id: 'x', error: 'e1' })]);
    loadingTasks$.next([task({ task_id: 'x', error: 'e1' })]);
    expect(service.toasts.length).toBe(1);
  });

  it('ignores cancelled and non-idle tasks', () => {
    loadingTasks$.next([task({ task_id: 'a', status: 'running', error: 'e' })]);
    loadingTasks$.next([task({ task_id: 'b', status: 'idle', error: 'Cancelled' })]);
    loadingTasks$.next([task({ task_id: 'c', status: 'idle', error: undefined })]);
    expect(service.toasts.length).toBe(0);
  });

  it('formatForClipboard includes headline and full HTTP context', () => {
    const errorContext: ErrorContext = {
      message: 'Request failed',
      status: 500,
      statusText: 'Server Error',
      method: 'POST',
      url: '/api/thing',
      requestId: 'req-1',
      datasetId: 'ds-1',
      detectorId: 'det-1',
      extra: { foo: 'bar' },
      rawBody: '{"err":true}',
      timestamp: '2026-01-01T00:00:00Z',
    };
    const toast: Toast = {
      id: 1,
      level: 'error',
      message: 'Request failed',
      detail: 'something broke',
      errorContext,
      timestamp: '2026-01-01T00:00:00Z',
    };
    const md = service.formatForClipboard(toast);
    expect(md).toContain('**VTSearch error**');
    expect(md).toContain('**Message:** Request failed');
    expect(md).toContain('**Detail:** something broke');
    expect(md).toContain('**Status:** 500 Server Error');
    expect(md).toContain('**Endpoint:** `POST /api/thing`');
    expect(md).toContain('**Request ID:** `req-1`');
    expect(md).toContain('**Dataset:** `ds-1`');
    expect(md).toContain('**Detector:** `det-1`');
    expect(md).toContain('"foo": "bar"');
    expect(md).toContain('{"err":true}');
  });

  it('formatForClipboard for a plain toast carries only the headline', () => {
    const toast: Toast = { id: 1, level: 'error', message: 'plain', timestamp: '2026-01-01T00:00:00Z' };
    const md = service.formatForClipboard(toast);
    expect(md).toContain('**Message:** plain');
    expect(md).not.toContain('**Status:**');
    expect(md).not.toContain('**Endpoint:**');
  });

  // --- levels ------------------------------------------------------------

  it('warning() stays up until dismissed, like error()', async () => {
    vi.useFakeTimers();
    try {
      service.warning({ message: 'partial results' });
      expect(service.toasts[0].level).toBe('warning');
      await vi.advanceTimersByTimeAsync(60_000);
      expect(service.toasts.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('info() auto-dismisses on the success timer', async () => {
    vi.useFakeTimers();
    try {
      service.info({ message: 'heads up' });
      expect(service.toasts[0].level).toBe('info');
      await vi.advanceTimersByTimeAsync(5000);
      expect(service.toasts).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('show() dispatches to the level it is handed', () => {
    for (const level of ['error', 'warning', 'success', 'info'] as const) {
      service.dismissAll();
      service.show(level, { message: level });
      expect(service.toasts[0].level).toBe(level);
    }
  });

  // --- backend notifications (#3132) --------------------------------------

  function notification(overrides: Partial<ServerNotification> = {}): ServerNotification {
    return { id: 'note_ab_1', level: 'warning', message: 'Skipped 3 files', ...overrides };
  }

  it('renders a backend notification as a toast of the same level', () => {
    notifications$.next(notification());

    expect(service.toasts.length).toBe(1);
    expect(service.toasts[0].level).toBe('warning');
    expect(service.toasts[0].message).toBe('Skipped 3 files');
  });

  it('folds the source into the detail line, ahead of the detail', () => {
    // The headline is the plugin's own sentence and reads as written; which
    // part of the app spoke is context and belongs on the secondary line.
    notifications$.next(notification({ source: 'Server Folder', detail: 'a, b, c' }));
    expect(service.toasts[0].detail).toBe('Server Folder — a, b, c');
  });

  it('uses the source alone as the detail when there is no detail', () => {
    notifications$.next(notification({ source: 'Server Folder' }));
    expect(service.toasts[0].detail).toBe('Server Folder');
  });

  it('leaves the detail unset when neither source nor detail is given', () => {
    notifications$.next(notification());
    expect(service.toasts[0].detail).toBeUndefined();
  });

  it('dedups on the backend notification id, so a redelivered frame does not stack', () => {
    notifications$.next(notification());
    notifications$.next(notification());
    expect(service.toasts.length).toBe(1);
  });

  it('stacks distinct notifications', () => {
    notifications$.next(notification({ id: 'note_ab_1', message: 'first' }));
    notifications$.next(notification({ id: 'note_ab_2', message: 'second' }));
    expect(service.toasts.map((t) => t.message)).toEqual(['first', 'second']);
  });

  it('auto-dismisses an info notification but keeps an error one', async () => {
    vi.useFakeTimers();
    try {
      notifications$.next(notification({ id: 'note_ab_1', level: 'info', message: 'fyi' }));
      notifications$.next(notification({ id: 'note_ab_2', level: 'error', message: 'bad' }));
      await vi.advanceTimersByTimeAsync(5000);
      expect(service.toasts.map((t) => t.message)).toEqual(['bad']);
    } finally {
      vi.useRealTimers();
    }
  });
});
