import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { BehaviorSubject, Subject } from 'rxjs';

import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsApiService } from './datasets-api.service';
import { DetectorsApiService } from './detectors-api.service';
import { ProgressEventsService } from './progress-events.service';
import {
  DatasetRegistryEntry,
  DetectorRegistryEntry,
  LoadingTask,
} from '../models/api.models';

/**
 * Regression spec for H25 — "Active dataset pair is set before load
 * completes" (see `docs/plans/logical-bug-audit.md`).
 *
 * The fix split `ActiveContextService` into two layers:
 *   - **intent** — flips immediately on switch entry (for UI affordances)
 *   - **active** — flips only after the dataset / detector load resolves
 *     AND the corresponding `loadingTasks` SSE channel goes idle.
 *
 * The HTTP interceptor (`activeContextInterceptor`) reads the *active*
 * layer.  These tests pin down that contract so a future refactor can't
 * silently regress to flipping active up-front and re-introducing the
 * 409 `dataset_not_loaded` cascade.
 */
describe('ContextSwitchService — H25 active/intent layering', () => {
  let switcher: ContextSwitchService;
  let activeContext: ActiveContextService;

  // Stub state — we drive these directly from each test instead of
  // letting the real services do HTTP / SSE work.
  let datasets: DatasetRegistryEntry[];
  let detectors: DetectorRegistryEntry[];
  let loadRegisteredSubjects: Map<string, Subject<unknown>>;
  let loadDetectorSubjects: Map<string, Subject<unknown>>;
  let loadingTasks$: BehaviorSubject<LoadingTask[]>;
  let detectorLoadingTasks$: BehaviorSubject<LoadingTask[]>;

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

    const datasetStateStub = {
      get datasets(): DatasetRegistryEntry[] {
        return datasets;
      },
      get detectors(): DetectorRegistryEntry[] {
        return detectors;
      },
      setLoading: (_loading: boolean): void => {
        /* no-op for tests */
      },
      refresh: (): void => {
        /* no-op for tests */
      },
    };

    const datasetsApiStub = {
      loadRegistered: (id: string): Subject<unknown> => {
        const s = new Subject<unknown>();
        loadRegisteredSubjects.set(id, s);
        return s;
      },
      cancelTask: (_taskId: string): Subject<unknown> => new Subject<unknown>(),
    };

    const detectorsApiStub = {
      loadDetector: (id: string): Subject<unknown> => {
        const s = new Subject<unknown>();
        loadDetectorSubjects.set(id, s);
        return s;
      },
      cancelDetectorLoadingTask: (_taskId: string): Subject<unknown> =>
        new Subject<unknown>(),
    };

    const progressEventsStub = {
      loadingTasks$: loadingTasks$.asObservable(),
      detectorLoadingTasks$: detectorLoadingTasks$.asObservable(),
      get loadingTasks(): LoadingTask[] {
        return loadingTasks$.value;
      },
      get detectorLoadingTasks(): LoadingTask[] {
        return detectorLoadingTasks$.value;
      },
    };

    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        { provide: DatasetStateService, useValue: datasetStateStub },
        { provide: DatasetsApiService, useValue: datasetsApiStub },
        { provide: DetectorsApiService, useValue: detectorsApiStub },
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
    // does NOT pass on its first emission — otherwise the empty initial
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

    // HTTP load endpoint resolves — but the loading task is still
    // non-idle, so active must stay pinned.
    loadRegisteredSubjects.get('d1')!.next({});
    loadRegisteredSubjects.get('d1')!.complete();
    expect(activeContext.datasetId).toBe('old-ds');
    expect(activeContext.modelId).toBe('old-det');

    // SSE channel goes idle — only now does active flip.
    loadingTasks$.next([]);
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

    loadDetectorSubjects.get('m1')!.next({});
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

    expect(loadDetectorSubjects.has('m1')).toBeTrue();
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

    // First switch: d1/m1 — needs a dataset load that we'll never finish.
    switcher.applyActivePair('d1', 'm1').subscribe();
    expect(activeContext.intentDatasetId).toBe('d1');

    // Second switch starts before d1 finishes loading — d2/m2 is already
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

  it('promotes intent to active via the completion observable returned to the route guard', () => {
    datasets = [makeDataset('d1', { loaded: true })];
    detectors = [makeDetector('m1', { detector_loaded: true })];

    let completed = false;
    switcher.applyActivePair('d1', 'm1').subscribe({ complete: () => (completed = true) });

    expect(completed).toBeTrue();
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });
});
