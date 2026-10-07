import { TestBed } from '@angular/core/testing';
import { BehaviorSubject, of, Subject, throwError } from 'rxjs';

import { AutoFindService } from './autofind.service';
import { AuthService } from './auth.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ProgressEventsService } from './progress-events.service';
import { ToastService } from './toast.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { AutoFindTaskInfo, LoadingTask } from '../models/api.models';

/**
 * `AutoFindService` owns the end of a background AutoFind (#4252): a Find this
 * tab started (the Dashboard's big Find button, #4529) opens its results unless
 * the dialog already shows another run; any other run of the user's (an
 * import's, or the dataset ⋯ menu's Run AutoFind, #4615) is announced with a
 * toast that opens them. Other users' runs, failed runs, and runs already
 * handled are left alone.
 */
describe('AutoFindService', () => {
  let service: AutoFindService;
  let loadingTasks$: Subject<LoadingTask[]>;
  let serverReset$: Subject<void>;
  let registryApi: { runAutofind: ReturnType<typeof vi.fn> };
  let findApi: { getAutofindRun: ReturnType<typeof vi.fn> };
  let toast: { success: ReturnType<typeof vi.fn>; warning: ReturnType<typeof vi.fn>; info: ReturnType<typeof vi.fn> };

  const run = { dataset_id: 'ds1', dataset_name: 'Birds', media_type: 'audio', detectors_run: 1, results: {} };

  function info(overrides: Partial<AutoFindTaskInfo> = {}): AutoFindTaskInfo {
    return {
      run_id: '_autofind_1',
      owner: 'alice',
      trigger: 'import',
      dataset_id: 'ds1',
      dataset_name: 'Birds',
      detectors_run: 2,
      total_hits: 5,
      ...overrides,
    };
  }

  function finished(overrides: Partial<LoadingTask> = {}, autofind = info()): LoadingTask {
    return {
      status: 'idle',
      message: '',
      current: 2,
      total: 2,
      task_id: autofind.run_id,
      name: `AutoFind: ${autofind.dataset_name}`,
      created_at: 0,
      dataset_id: autofind.dataset_id,
      autofind,
      ...overrides,
    };
  }

  beforeEach(() => {
    loadingTasks$ = new Subject<LoadingTask[]>();
    serverReset$ = new Subject<void>();
    registryApi = { runAutofind: vi.fn(() => of({ ok: true, message: 'AutoFind started', task_id: '_autofind_1' })) };
    findApi = { getAutofindRun: vi.fn(() => of(run)) };
    toast = { success: vi.fn(), warning: vi.fn(), info: vi.fn() };

    configureZoneless({
      providers: [
        { provide: ProgressEventsService, useValue: { loadingTasks$, serverReset$ } },
        { provide: AuthService, useValue: { status$: new BehaviorSubject({ user: 'alice' }) } },
        { provide: DatasetsRegistryApiService, useValue: registryApi },
        { provide: DetectorsFindApiService, useValue: findApi },
        { provide: ToastService, useValue: toast },
      ],
    });
    service = TestBed.inject(AutoFindService);
  });

  it('opens the results of a Find this tab started, once it finishes', () => {
    service.find('ds1', ['m1', 'm2']);
    expect(registryApi.runAutofind).toHaveBeenCalledWith('ds1', ['m1', 'm2']);

    loadingTasks$.next([finished({ status: 'loading' }, info({ trigger: 'find' }))]);
    expect(findApi.getAutofindRun).not.toHaveBeenCalled();

    loadingTasks$.next([finished({}, info({ trigger: 'find' }))]);
    expect(findApi.getAutofindRun).toHaveBeenCalledWith('_autofind_1', expect.anything());
    expect(service.results()).toEqual(run);
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("ends the ⋯ menu's Run AutoFind with a toast, not the dialog (#4615)", () => {
    service.run('ds1');
    expect(registryApi.runAutofind).toHaveBeenCalledWith('ds1');

    loadingTasks$.next([finished({}, info({ trigger: 'manual' }))]);
    expect(findApi.getAutofindRun).not.toHaveBeenCalled();
    expect(service.results()).toBeNull();
    expect(toast.success).toHaveBeenCalledTimes(1);
    const opts = toast.success.mock.calls[0][0];
    expect(opts.message).toBe('AutoFind finished on "Birds": 5 hits');
    expect(opts.action.label).toBe('View results');

    opts.action.onClick();
    expect(service.results()).toEqual(run);
  });

  it("announces a requested run instead of replacing another run's open results", () => {
    registryApi.runAutofind
      .mockReturnValueOnce(of({ ok: true, message: 'AutoFind started', task_id: '_autofind_1' }))
      .mockReturnValueOnce(of({ ok: true, message: 'AutoFind started', task_id: '_autofind_2' }));
    service.find('ds1', ['m1']);
    service.find('ds2', ['m1']);

    loadingTasks$.next([finished({}, info({ trigger: 'find' }))]);
    expect(service.results()).toEqual(run);
    expect(findApi.getAutofindRun).toHaveBeenCalledTimes(1);

    const second = info({ run_id: '_autofind_2', trigger: 'find', dataset_id: 'ds2', dataset_name: 'Frogs' });
    loadingTasks$.next([finished({}, second)]);
    expect(findApi.getAutofindRun).toHaveBeenCalledTimes(1);
    expect(toast.success).toHaveBeenCalledTimes(1);
    expect(toast.success.mock.calls[0][0].message).toBe('Find finished on "Frogs": 5 hits');
  });

  it('opens only the first of two requested runs that land together', () => {
    const pending = new Subject<typeof run>();
    findApi.getAutofindRun.mockReturnValue(pending);
    registryApi.runAutofind
      .mockReturnValueOnce(of({ ok: true, message: 'AutoFind started', task_id: '_autofind_1' }))
      .mockReturnValueOnce(of({ ok: true, message: 'AutoFind started', task_id: '_autofind_2' }));
    service.find('ds1', ['m1']);
    service.find('ds2', ['m1']);

    // The first fetch is still in flight when the second run is seen.
    loadingTasks$.next([
      finished({}, info({ trigger: 'find' })),
      finished({}, info({ run_id: '_autofind_2', trigger: 'find', dataset_id: 'ds2', dataset_name: 'Frogs' })),
    ]);
    expect(findApi.getAutofindRun).toHaveBeenCalledTimes(1);
    expect(findApi.getAutofindRun).toHaveBeenCalledWith('_autofind_1', expect.anything());
    expect(toast.success).toHaveBeenCalledTimes(1);

    pending.next(run);
    expect(service.results()).toEqual(run);
  });

  it('opens a requested run again once the dialog is closed', () => {
    service.openResults('_autofind_0');
    service.closeResults();
    service.find('ds1', ['m1']);
    loadingTasks$.next([finished({}, info({ trigger: 'find' }))]);
    expect(findApi.getAutofindRun).toHaveBeenLastCalledWith('_autofind_1', expect.anything());
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('announces an import-triggered run with a toast that opens its results', () => {
    loadingTasks$.next([finished()]);

    expect(toast.success).toHaveBeenCalledTimes(1);
    const opts = toast.success.mock.calls[0][0];
    expect(opts.message).toBe('AutoFind finished on "Birds": 5 hits');
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
    service.find('ds1', ['m1']);
    loadingTasks$.next([finished({ error: 'Cancelled' }, info({ trigger: 'find' }))]);
    expect(findApi.getAutofindRun).not.toHaveBeenCalled();
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("explains an import's skipped run instead of announcing results", () => {
    loadingTasks$.next([finished({}, info({ skipped: 'None of your AutoFind detectors are for audio datasets.' }))]);
    expect(toast.success).not.toHaveBeenCalled();
    expect(toast.info).toHaveBeenCalledTimes(1);
    expect(toast.info.mock.calls[0][0].message).toBe('AutoFind didn\'t run on "Birds"');
    expect(toast.info.mock.calls[0][0].detail).toContain('audio datasets');
  });

  it('warns rather than congratulates when the auto-export failed', () => {
    loadingTasks$.next([finished({}, info({ auto_export: { exporter: 'email_smtp', success: false, error: 'no MX' } }))]);
    expect(toast.success).not.toHaveBeenCalled();
    expect(toast.warning.mock.calls[0][0].detail).toContain('no MX');
  });

  it('says so when the results have expired', () => {
    findApi.getAutofindRun.mockReturnValue(throwError(() => new Error('404')));
    service.openResults('_autofind_old');
    expect(toast.warning).toHaveBeenCalled();
    expect(service.results()).toBeNull();
  });

  it('forgets pending runs when the backend restarts', () => {
    service.find('ds1', ['m1']);
    serverReset$.next();
    loadingTasks$.next([finished({}, info({ trigger: 'find' }))]);
    // No longer "ours to open", so it is announced like any other run.
    expect(findApi.getAutofindRun).not.toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledTimes(1);
  });

  it('closes the dialog', () => {
    service.openResults('_autofind_1');
    service.closeResults();
    expect(service.results()).toBeNull();
  });
});
