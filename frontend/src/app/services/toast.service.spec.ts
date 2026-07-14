import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { ErrorContext, Toast, ToastService } from './toast.service';
import { ProgressEventsService } from './progress-events.service';
import { LoadingTask } from '../models/api.models';

describe('ToastService', () => {
  let service: ToastService;
  let loadingTasks$: Subject<LoadingTask[]>;
  let detectorLoadingTasks$: Subject<LoadingTask[]>;

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
    TestBed.configureTestingModule({
      providers: [
        ToastService,
        { provide: ProgressEventsService, useValue: { loadingTasks$, detectorLoadingTasks$ } },
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
});
