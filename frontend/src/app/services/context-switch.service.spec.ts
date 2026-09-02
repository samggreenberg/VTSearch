import { TestBed } from '@angular/core/testing';

import { provideRouter } from '@angular/router';
import { BehaviorSubject, Subject } from 'rxjs';

import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { ProgressEventsService } from './progress-events.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { DatasetRegistryEntry, LoadingTask } from '../models/api.models';
import { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * Regression spec for logical-bug-audit H25: "Active dataset pair is set
 * before load completes".
 *
 * The fix split `ActiveContextService` into two layers:
 *   - **intent**: flips immediately on switch entry (for UI affordances)
 *   - **active**: flips only after the dataset / detector load resolves
 *     AND the corresponding `loadingTasks` SSE channel goes idle.
 *
 * The HTTP interceptor (`activeContextInterceptor`) reads the *active*
 * layer.  These tests pin down that contract so a future refactor can't
 * silently regress to flipping active up-front and re-introducing the
 * 409 `dataset_not_loaded` cascade.
 */
describe('ContextSwitchService: H25 active/intent layering', () => {
  let switcher: ContextSwitchService;
  let activeContext: ActiveContextService;

  // Stub state: we drive these directly from each test instead of
  // letting the real services do HTTP / SSE work.
  let datasets: DatasetRegistryEntry[];
  let detectors: DetectorRegistryEntry[];
  let loadRegisteredSubjects: Map<string, Subject<unknown>>;
  let loadDetectorSubjects: Map<string, Subject<unknown>>;
  let loadingTasks$: BehaviorSubject<LoadingTask[]>;
  let detectorLoadingTasks$: BehaviorSubject<LoadingTask[]>;
  let cancelledTaskIds: string[];
  let cancelledDetectorTaskIds: string[];

  function makeDataset(
    id: string,
    overrides: Partial<DatasetRegistryEntry> = {},
  ): DatasetRegistryEntry {
    return { id, name: id, media_type: 'audio', loaded: false, ...overrides };
  }

  function makeDetector(
    id: string,
    overrides: Partial<DetectorRegistryEntry> = {},
  ): DetectorRegistryEntry {
    return {
      id,
      name: id,
      media_type: 'audio',
      detector_loaded: false,
      ...overrides,
    };
  }

  function makeLoadingTask(overrides: Partial<LoadingTask>): LoadingTask {
    return {
      status: 'loading',
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
    datasets = [];
    detectors = [];
    loadRegisteredSubjects = new Map();
    loadDetectorSubjects = new Map();
    loadingTasks$ = new BehaviorSubject<LoadingTask[]>([]);
    detectorLoadingTasks$ = new BehaviorSubject<LoadingTask[]>([]);
    cancelledTaskIds = [];
    cancelledDetectorTaskIds = [];

    const datasetStateStub = {
      get datasets(): DatasetRegistryEntry[] {
        return datasets;
      },
      get detectors(): DetectorRegistryEntry[] {
        return detectors;
      },
      // Computed on call, like the real service's `computed` Maps: the
      // arrays above are reassigned per test.
      datasetById: (): ReadonlyMap<string, DatasetRegistryEntry> =>
        new Map(datasets.map((d) => [d.id, d])),
      detectorById: (): ReadonlyMap<string, DetectorRegistryEntry> =>
        new Map(detectors.map((d) => [d.id, d])),
      setLoading: (_loading: boolean): void => {
        /* no-op for tests */
      },
      refresh: (): void => {
        /* no-op for tests */
      },
    };

    const datasetsRegistryApiStub = {
      loadRegistered: (id: string): Subject<unknown> => {
        const s = new Subject<unknown>();
        loadRegisteredSubjects.set(id, s);
        return s;
      },
      cancelTask: (taskId: string): Subject<unknown> => {
        cancelledTaskIds.push(taskId);
        return new Subject<unknown>();
      },
    };

    const detectorsRegistryApiStub = {
      loadDetector: (id: string): Subject<unknown> => {
        const s = new Subject<unknown>();
        loadDetectorSubjects.set(id, s);
        return s;
      },
      cancelDetectorLoadingTask: (taskId: string): Subject<unknown> => {
        cancelledDetectorTaskIds.push(taskId);
        return new Subject<unknown>();
      },
    };

    const progressEventsStub = {
      loadingTasks$: loadingTasks$.asObservable(),
      detectorLoadingTasks$: detectorLoadingTasks$.asObservable(),
      loadingTasks: (): LoadingTask[] => loadingTasks$.value,
      detectorLoadingTasks: (): LoadingTask[] => detectorLoadingTasks$.value,
    };

    configureZoneless({
      providers: [
        ...provideHttpTesting(),
        provideRouter([]),
        { provide: DatasetStateService, useValue: datasetStateStub },
        { provide: DatasetsRegistryApiService, useValue: datasetsRegistryApiStub },
        { provide: DetectorsRegistryApiService, useValue: detectorsRegistryApiStub },
        { provide: ProgressEventsService, useValue: progressEventsStub },
      ],
    });

    switcher = TestBed.inject(ContextSwitchService);
    activeContext = TestBed.inject(ActiveContextService);
  });

  it('flips intent and active synchronously when nothing needs to load', () => {
    datasets = [makeDataset('d1', { loaded: true })];
    detectors = [makeDetector('m1', { detector_loaded: true })];

    switcher.applyActivePair('d1', 'm1').subscribe();

    expect(activeContext.intentDatasetId).toBe('d1');
    expect(activeContext.intentModelId).toBe('m1');
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('flips intent immediately but keeps active pinned while a dataset load is in flight', () => {
    // Start from a known active pair so we can prove it does NOT change.
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1', { loaded: false })]; // needs load
    detectors = [makeDetector('m1', { detector_loaded: true })];
    // Pre-populate a non-idle task so `waitForDatasetLoad`'s filter
    // does NOT pass on its first emission; otherwise the empty initial
    // BehaviorSubject value would satisfy "no task non-idle" trivially.
    loadingTasks$.next([makeLoadingTask({ dataset_id: 'd1', status: 'loading' })]);

    switcher.applyActivePair('d1', 'm1').subscribe();

    // Intent reflects the user's pick immediately…
    expect(activeContext.intentDatasetId).toBe('d1');
    expect(activeContext.intentModelId).toBe('m1');
    // …but the HTTP interceptor still reads the previous pair until
    // the load completes (the whole point of H25).
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    // HTTP load endpoint resolves (with the background task's id), but the
    // loading task is still non-idle, so active must stay pinned.
    loadRegisteredSubjects.get('d1')!.next({ ok: true, message: '', task_id: 't' });
    loadRegisteredSubjects.get('d1')!.complete();
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    // SSE channel goes idle; only now does active flip.
    loadingTasks$.next([]);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('does NOT flip active before the SSE load task appears (browse-on-unloaded race)', () => {
    // Reproduces the production race the Browse button hit: the HTTP load
    // response lands BEFORE the `loading-tasks` SSE frame for that load
    // arrives, so the channel is momentarily empty. The waiter must not
    // read that empty snapshot as "load already finished" — otherwise it
    // promotes active and the caller fires requests (e.g. the projection
    // build) against a still-loading dataset, which 409s. Unlike the H25
    // tests below, we deliberately do NOT pre-seed a non-idle task.
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1', { loaded: false })]; // needs load
    detectors = [makeDetector('m1', { detector_loaded: true })];

    switcher.applyActivePair('d1', 'm1').subscribe();

    // HTTP load endpoint resolves with the background task's id, but the
    // SSE channel has not reported the task yet (still empty).
    loadRegisteredSubjects.get('d1')!.next({ ok: true, message: '', task_id: 'task-d1' });
    loadRegisteredSubjects.get('d1')!.complete();
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    // The task now appears, running. Active must still stay pinned.
    loadingTasks$.next([
      makeLoadingTask({ dataset_id: 'd1', task_id: 'task-d1', status: 'loading' }),
    ]);
    expect(activeContext.datasetId).toBe('old-ds');

    // Task goes idle: only now does active flip.
    loadingTasks$.next([
      makeLoadingTask({ dataset_id: 'd1', task_id: 'task-d1', status: 'idle' }),
    ]);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('keeps active pinned while a detector load is in flight', () => {
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1', { loaded: true })];
    detectors = [makeDetector('m1', { detector_loaded: false })]; // needs load
    detectorLoadingTasks$.next([
      makeLoadingTask({ detector_id: 'm1', status: 'loading' }),
    ]);

    switcher.applyActivePair('d1', 'm1').subscribe();

    expect(activeContext.intentDatasetId).toBe('d1');
    expect(activeContext.intentModelId).toBe('m1');
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    loadDetectorSubjects.get('m1')!.next({ ok: true, task_id: 't' });
    loadDetectorSubjects.get('m1')!.complete();
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    detectorLoadingTasks$.next([]);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('re-embeds labels when dataset and detector embedders differ, holding active until detector reloads', () => {
    // detector_loaded is true but the embedders disagree → still needs
    // a detector load to re-embed labels, and active must wait.
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [
      makeDataset('d1', { loaded: true, embedder: 'clap' }),
    ];
    detectors = [
      makeDetector('m1', { detector_loaded: true, embedder: 'xclip' }),
    ];
    detectorLoadingTasks$.next([
      makeLoadingTask({ detector_id: 'm1', status: 'loading' }),
    ]);

    switcher.applyActivePair('d1', 'm1').subscribe();

    expect(loadDetectorSubjects.has('m1')).toBe(true);
    expect(activeContext.datasetId).toBe('old-ds');

    loadDetectorSubjects.get('m1')!.next({});
    detectorLoadingTasks$.next([]);

    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('discards a stale switch via the request-id check when a second switch starts first', () => {
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1'), makeDataset('d2', { loaded: true })];
    detectors = [
      makeDetector('m1', { detector_loaded: true }),
      makeDetector('m2', { detector_loaded: true }),
    ];
    loadingTasks$.next([makeLoadingTask({ dataset_id: 'd1', status: 'loading' })]);

    // First switch: d1/m1; needs a dataset load that we'll never finish.
    switcher.applyActivePair('d1', 'm1').subscribe();
    expect(activeContext.intentDatasetId).toBe('d1');

    // Second switch starts before d1 finishes loading; d2/m2 is already
    // loaded so no awaiting is needed.  This must replace the in-flight
    // switch: intent and active both end up on d2/m2, and a *late*
    // completion of the d1 load (next + idle tasks) must NOT roll
    // active back to d1.
    switcher.applyActivePair('d2', 'm2').subscribe();
    expect(activeContext.intentDatasetId).toBe('d2');
    expect(activeContext.datasetId).toBe('d2');
    expect(activeContext.modelId).toBe('m2');

    // Late arrival of the d1 load's tail (HTTP resolve + idle SSE).
    // The request-id captured by the d1 ActiveSwitch is now stale, so
    // finishIfCurrent() must short-circuit and leave active on d2/m2.
    loadRegisteredSubjects.get('d1')!.next({});
    loadingTasks$.next([]);

    expect(activeContext.datasetId).toBe('d2');
    expect(activeContext.modelId).toBe('m2');
  });

  it('does not promote the pair when the dataset load kick-off fails', () => {
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1', { loaded: false })];
    detectors = [makeDetector('m1', { detector_loaded: true })];

    let emitted = false;
    let completed = false;
    switcher.applyActivePair('d1', 'm1').subscribe({
      next: () => (emitted = true),
      complete: () => (completed = true),
    });

    // The load POST fails (e.g. 500). Regression: this used to count as a
    // completed load and promote the unloaded pair to active, re-opening
    // the H25 cascade of 409 dataset_not_loaded from the very view the
    // guard was about to activate.
    loadRegisteredSubjects.get('d1')!.error(new Error('500'));

    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');
    // The pulldown highlight rolls back to the still-active pair.
    expect(activeContext.intentDatasetId).toBe('old-ds');
    // Completion completes WITHOUT emitting → the guard's defaultIfEmpty
    // denies navigation cleanly.
    expect(emitted).toBe(false);
    expect(completed).toBe(true);
    expect(switcher.switching).toBe(false);
  });

  it('does not promote the pair when the background load task errors', () => {
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1', { loaded: false })];
    detectors = [makeDetector('m1', { detector_loaded: true })];

    let emitted = false;
    switcher.applyActivePair('d1', 'm1').subscribe({ next: () => (emitted = true) });

    loadRegisteredSubjects.get('d1')!.next({ ok: true, message: '', task_id: 'task-d1' });
    loadRegisteredSubjects.get('d1')!.complete();
    loadingTasks$.next([
      makeLoadingTask({ dataset_id: 'd1', task_id: 'task-d1', status: 'loading' }),
    ]);

    // The background load dies (OOM, bad pickle, …): the task settles
    // idle WITH an error. The pair must not be promoted.
    loadingTasks$.next([
      makeLoadingTask({ dataset_id: 'd1', task_id: 'task-d1', status: 'idle', error: 'boom' }),
    ]);

    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');
    expect(emitted).toBe(false);
    expect(switcher.switching).toBe(false);
  });

  it('cancels only the superseded switch\'s own loading tasks on cancel-and-replace', () => {
    activeContext.setActivePair('old-ds', 'old-det');
    datasets = [makeDataset('d1'), makeDataset('d2', { loaded: true }), makeDataset('d3')];
    detectors = [
      makeDetector('m1', { detector_loaded: true }),
      makeDetector('m2', { detector_loaded: true }),
    ];
    // d1's load task (owned by the switch about to be superseded) plus an
    // unrelated in-flight load of d3 (e.g. a dashboard-initiated import).
    loadingTasks$.next([
      makeLoadingTask({ dataset_id: 'd1', task_id: 'task-d1', status: 'loading' }),
      makeLoadingTask({ dataset_id: 'd3', task_id: 'task-d3', status: 'loading' }),
    ]);

    switcher.applyActivePair('d1', 'm1').subscribe();
    // Rapid re-click: replace d1/m1 with d2/m2.
    switcher.applyActivePair('d2', 'm2').subscribe();

    // Regression: cancel-and-replace used to POST a cancel for EVERY
    // non-idle task, killing the unrelated d3 load mid-flight.
    expect(cancelledTaskIds).toEqual(['task-d1']);
    expect(cancelledDetectorTaskIds).toEqual([]);
  });

  it('promotes intent to active via the completion observable returned to the route guard', () => {
    datasets = [makeDataset('d1', { loaded: true })];
    detectors = [makeDetector('m1', { detector_loaded: true })];

    let completed = false;
    switcher.applyActivePair('d1', 'm1').subscribe({ complete: () => (completed = true) });

    expect(completed).toBe(true);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });
});
