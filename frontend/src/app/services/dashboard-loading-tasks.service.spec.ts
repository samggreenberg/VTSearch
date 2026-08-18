import { TestBed } from '@angular/core/testing';
import { of, Subject, throwError } from 'rxjs';

import { DashboardLoadingTasksService } from './dashboard-loading-tasks.service';
import { AchievementsService } from './achievements.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { ProgressEventsService } from './progress-events.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { LoadingTask } from '../models/api.models';

/**
 * The dashboard promotes a loaded dataset/detector to the active pair via the
 * `onComplete` callback it hands to `startProgressPolling` /
 * `startDetectorProgressPolling`. These tests pin the callback contract:
 * a callback registered while the polling loop is ALREADY active (an import
 * or another load in flight - the constructor auto-starts polling on any
 * non-idle SSE snapshot) must still fire when the loop settles, and only a
 * failure of the callback's own task suppresses it.
 */
describe('DashboardLoadingTasksService', () => {
  let service: DashboardLoadingTasksService;
  let loadingTasks$: Subject<LoadingTask[]>;
  let detectorLoadingTasks$: Subject<LoadingTask[]>;
  let serverReset$: Subject<void>;
  let datasetsRegistryApiStub: { cancelTask: ReturnType<typeof vi.fn> };

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
    serverReset$ = new Subject<void>();

    const progressEventsStub = {
      loadingTasks$,
      detectorLoadingTasks$,
      serverReset$,
    };
    const datasetStateStub = {
      datasets: [],
      refresh: vi.fn(),
      setLoading: vi.fn(),
    };
    const achievementsStub = { refresh: vi.fn() };
    datasetsRegistryApiStub = { cancelTask: vi.fn(() => of(null)) };
    const detectorsRegistryApiStub = { cancelDetectorLoadingTask: vi.fn(() => of(null)) };

    configureZoneless({
      providers: [
        { provide: ProgressEventsService, useValue: progressEventsStub },
        { provide: DatasetStateService, useValue: datasetStateStub },
        { provide: AchievementsService, useValue: achievementsStub },
        { provide: DatasetsRegistryApiService, useValue: datasetsRegistryApiStub },
        { provide: DetectorsRegistryApiService, useValue: detectorsRegistryApiStub },
      ],
    });

    service = TestBed.inject(DashboardLoadingTasksService);
  });

  it('fires onComplete registered while polling is already active (regression)', () => {
    // An unrelated import is running → the constructor subscription
    // auto-starts the polling loop with NO callback.
    loadingTasks$.next([task({ task_id: 'import-1', status: 'running' })]);

    // The user clicks Load on a dataset; polling is already active, so the
    // early-return branch is taken. The callback used to be silently
    // dropped here, leaving setActivePair never called.
    const onComplete = vi.fn();
    service.startProgressPolling('load-1', onComplete);

    // The awaited load shows up in the stream, then everything settles.
    loadingTasks$.next([
      task({ task_id: 'import-1', status: 'running' }),
      task({ task_id: 'load-1', status: 'running' }),
    ]);
    expect(onComplete).not.toHaveBeenCalled();

    loadingTasks$.next([task({ task_id: 'import-1' }), task({ task_id: 'load-1' })]);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('suppresses onComplete when its own task failed', () => {
    const onComplete = vi.fn();
    service.startProgressPolling('load-1', onComplete);

    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);
    loadingTasks$.next([task({ task_id: 'load-1', error: 'exploded' })]);

    expect(onComplete).not.toHaveBeenCalled();
  });

  it('fires onComplete when only an unrelated task failed', () => {
    loadingTasks$.next([task({ task_id: 'import-1', status: 'running' })]);

    const onComplete = vi.fn();
    service.startProgressPolling('load-1', onComplete);

    loadingTasks$.next([
      task({ task_id: 'import-1', status: 'running' }),
      task({ task_id: 'load-1', status: 'running' }),
    ]);
    // The unrelated import fails; the dataset load succeeds. Promotion of
    // the successfully loaded dataset must not be held hostage.
    loadingTasks$.next([task({ task_id: 'import-1', error: 'disk full' }), task({ task_id: 'load-1' })]);

    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('fires each registered dataset callback exactly once', () => {
    const first = vi.fn();
    const second = vi.fn();
    service.startProgressPolling('load-1', first);
    service.startProgressPolling('load-2', second);

    loadingTasks$.next([
      task({ task_id: 'load-1', status: 'running' }),
      task({ task_id: 'load-2', status: 'running' }),
    ]);
    loadingTasks$.next([task({ task_id: 'load-1' }), task({ task_id: 'load-2' })]);

    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);

    // A later settle must not re-fire consumed callbacks.
    loadingTasks$.next([task({ task_id: 'other', status: 'running' })]);
    loadingTasks$.next([task({ task_id: 'other' })]);
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('fires detector onComplete registered while detector polling is already active (regression)', () => {
    // A detector load is in flight → constructor auto-starts the loop.
    detectorLoadingTasks$.next([task({ task_id: 'det-other', status: 'running' })]);

    const onComplete = vi.fn();
    service.startDetectorProgressPolling(onComplete);

    detectorLoadingTasks$.next([task({ task_id: 'det-other' })]);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('flags a task as cancelling on cancel and clears it once it leaves the active list', () => {
    service.startProgressPolling();
    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);
    expect(service.isCancelling('load-1')).toBe(false);

    service.cancelLoadingTask('load-1');
    expect(service.isCancelling('load-1')).toBe(true);

    // Still unwinding: the task is reported active, so the flag stays set.
    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);
    expect(service.isCancelling('load-1')).toBe(true);

    // Backend finished cancelling: the task goes idle (with the Cancelled
    // sentinel that gets filtered out), leaving the active list → flag clears.
    loadingTasks$.next([task({ task_id: 'load-1', error: 'Cancelled' })]);
    expect(service.isCancelling('load-1')).toBe(false);
  });

  it('clears the cancelling flag when the backend refuses the cancel', () => {
    // POST /api/dataset/cancel/<id> answers 409 when the cancel reached
    // nothing — the task had already finished, or its progress was stale with
    // no worker behind it. The row is not cancelling, so it must not be left
    // sitting on "Cancelling…".
    datasetsRegistryApiStub.cancelTask.mockReturnValueOnce(throwError(() => ({ status: 409 })));
    service.startProgressPolling();
    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);

    service.cancelLoadingTask('load-1');

    expect(service.isCancelling('load-1')).toBe(false);
  });

  it('flags a detector task as cancelling and clears it when it settles', () => {
    service.startDetectorProgressPolling();
    detectorLoadingTasks$.next([task({ task_id: 'det-1', status: 'running' })]);

    service.cancelDetectorLoadingTask('det-1');
    expect(service.isCancelling('det-1')).toBe(true);

    detectorLoadingTasks$.next([task({ task_id: 'det-1', error: 'Cancelled' })]);
    expect(service.isCancelling('det-1')).toBe(false);
  });

  it('drops cancelling flags on backend restart', () => {
    service.startProgressPolling();
    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);
    service.cancelLoadingTask('load-1');
    expect(service.isCancelling('load-1')).toBe(true);

    serverReset$.next();
    expect(service.isCancelling('load-1')).toBe(false);
  });

  it('clears pending callbacks on backend restart', () => {
    const onComplete = vi.fn();
    service.startProgressPolling('load-1', onComplete);
    loadingTasks$.next([task({ task_id: 'load-1', status: 'running' })]);

    serverReset$.next();

    // The restarted backend knows nothing about load-1; a fresh settle
    // must not fire the stale callback.
    loadingTasks$.next([task({ task_id: 'new-task', status: 'running' })]);
    loadingTasks$.next([task({ task_id: 'new-task' })]);
    expect(onComplete).not.toHaveBeenCalled();
  });
});
