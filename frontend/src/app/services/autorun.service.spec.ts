import { TestBed } from '@angular/core/testing';
import { BehaviorSubject, of, Subject, throwError } from 'rxjs';

import { AutoRunService } from './autorun.service';
import { AuthService } from './auth.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ProgressEventsService } from './progress-events.service';
import { ToastService } from './toast.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { AutoRunTaskInfo, LoadingTask } from '../models/api.models';

/**
 * `AutoRunService` owns the end of a background AutoRun (#4252): a run this
 * tab started from the dataset ⋯ menu opens its results; any other run of the
 * user's (an import's) is announced with a toast that opens them. Other
 * users' runs, failed runs, and runs already handled are left alone.
 */
describe('AutoRunService', () => {
  let service: AutoRunService;
  let loadingTasks$: Subject<LoadingTask[]>;
  let serverReset$: Subject<void>;
  let registryApi: { runAutorun: ReturnType<typeof vi.fn> };
  let findApi: { getAutorunRun: ReturnType<typeof vi.fn> };
  let toast: { success: ReturnType<typeof vi.fn>; warning: ReturnType<typeof vi.fn> };

  const run = { dataset_id: 'ds1', dataset_name: 'Birds', media_type: 'audio', detectors_run: 1, results: {} };

  function info(overrides: Partial<AutoRunTaskInfo> = {}): AutoRunTaskInfo {
    return {
      run_id: '_autorun_1',
      owner: 'alice',
      trigger: 'import',
      dataset_id: 'ds1',
      dataset_name: 'Birds',
      detectors_run: 2,
      total_hits: 5,
      ...overrides,
    };
  }

  function finished(overrides: Partial<LoadingTask> = {}, autorun = info()): LoadingTask {
    return {
      status: 'idle',
      message: '',
      current: 2,
      total: 2,
      task_id: autorun.run_id,
      name: `AutoRun: ${autorun.dataset_name}`,
      created_at: 0,
      dataset_id: autorun.dataset_id,
      autorun,
      ...overrides,
    };
  }

  beforeEach(() => {
    loadingTasks$ = new Subject<LoadingTask[]>();
    serverReset$ = new Subject<void>();
    registryApi = { runAutorun: vi.fn(() => of({ ok: true, message: 'AutoRun started', task_id: '_autorun_1' })) };
    findApi = { getAutorunRun: vi.fn(() => of(run)) };
    toast = { success: vi.fn(), warning: vi.fn() };

    configureZoneless({
      providers: [
        { provide: ProgressEventsService, useValue: { loadingTasks$, serverReset$ } },
        { provide: AuthService, useValue: { status$: new BehaviorSubject({ user: 'alice' }) } },
        { provide: DatasetsRegistryApiService, useValue: registryApi },
        { provide: DetectorsFindApiService, useValue: findApi },
        { provide: ToastService, useValue: toast },
      ],
    });
    service = TestBed.inject(AutoRunService);
  });

  it('opens the results of a run this tab started, once it finishes', () => {
    service.run('ds1');
    expect(registryApi.runAutorun).toHaveBeenCalledWith('ds1');

    loadingTasks$.next([finished({ status: 'loading' }, info({ trigger: 'manual' }))]);
    expect(findApi.getAutorunRun).not.toHaveBeenCalled();

    loadingTasks$.next([finished({}, info({ trigger: 'manual' }))]);
    expect(findApi.getAutorunRun).toHaveBeenCalledWith('_autorun_1', expect.anything());
    expect(service.results()).toEqual(run);
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('announces an import-triggered run with a toast that opens its results', () => {
    loadingTasks$.next([finished()]);

    expect(toast.success).toHaveBeenCalledTimes(1);
    const opts = toast.success.mock.calls[0][0];
    expect(opts.message).toBe('AutoRun found 5 hits in "Birds"');
    expect(opts.autoDismissMs).toBe(0);
    expect(service.results()).toBeNull();

    opts.action.onClick();
    expect(service.results()).toEqual(run);
  });

  it('announces each run once, however often the row is re-sent', () => {
    loadingTasks$.next([finished()]);
    loadingTasks$.next([finished()]);
    expect(toast.success).toHaveBeenCalledTimes(1);
  });

  it("ignores another user's run", () => {
    loadingTasks$.next([finished({}, info({ owner: 'bob' }))]);
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('ignores a failed or cancelled run (ToastService reports failures)', () => {
    service.run('ds1');
    loadingTasks$.next([finished({ error: 'Cancelled' }, info({ trigger: 'manual' }))]);
    expect(findApi.getAutorunRun).not.toHaveBeenCalled();
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('warns rather than congratulates when the auto-export failed', () => {
    loadingTasks$.next([finished({}, info({ auto_export: { exporter: 'email_smtp', success: false, error: 'no MX' } }))]);
    expect(toast.success).not.toHaveBeenCalled();
    expect(toast.warning.mock.calls[0][0].detail).toContain('no MX');
  });

  it('says so when the results have expired', () => {
    findApi.getAutorunRun.mockReturnValue(throwError(() => new Error('404')));
    service.openResults('_autorun_old');
    expect(toast.warning).toHaveBeenCalled();
    expect(service.results()).toBeNull();
  });

  it('forgets pending runs when the backend restarts', () => {
    service.run('ds1');
    serverReset$.next();
    loadingTasks$.next([finished({}, info({ trigger: 'manual' }))]);
    // No longer "ours to open", so it is announced like any other run.
    expect(findApi.getAutorunRun).not.toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledTimes(1);
  });

  it('closes the dialog', () => {
    service.openResults('_autorun_1');
    service.closeResults();
    expect(service.results()).toBeNull();
  });
});
