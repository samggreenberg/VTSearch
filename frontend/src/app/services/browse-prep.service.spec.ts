import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { Observable, Subject, of, throwError } from 'rxjs';
import { HttpErrorResponse } from '@angular/common/http';

import { BrowsePrepService } from './browse-prep.service';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { ProgressEventsService } from './progress-events.service';
import { ProjectionApiService } from './projection-api.service';
import { LoadingTask } from '../models/api.models';
import type { ProjectionBuildResponse, ProjectionMeta } from '../models/projection.models';

/**
 * The Browse button must raise missing-load / missing-projection problems on
 * the dashboard: it loads AND projects a dataset (showing progress on the
 * row) before navigating to the browse view. These tests pin that contract:
 * navigation happens only once both phases settle, and failures surface inline
 * instead of leaking into the browse window.
 */
describe('BrowsePrepService', () => {
  let service: BrowsePrepService;
  let router: Router;

  let applyPair$: Subject<void>;
  let loadingTasks: LoadingTask[];
  // Reassigned per test to script the projection meta/build responses.
  let metaProvider: () => Observable<ProjectionMeta>;
  let buildProvider: () => Observable<ProjectionBuildResponse>;

  const DS = 'ds1';

  function meta(overrides: Partial<ProjectionMeta>): ProjectionMeta {
    return {
      projection_id: 'p',
      bounds: [0, 0, 1, 1],
      base_radius: 1,
      tile_span: 1,
      point_count: 0,
      levels: [],
      ...overrides,
    };
  }

  function erroredTask(): LoadingTask {
    return {
      status: 'idle',
      message: '',
      current: 0,
      total: 0,
      task_id: 't',
      name: 'n',
      created_at: 0,
      dataset_id: DS,
      error: 'boom',
    };
  }

  beforeEach(() => {
    applyPair$ = new Subject<void>();
    loadingTasks = [];
    metaProvider = () => of(meta({ status: 'idle' }));
    buildProvider = () => of({ status: 'building' });

    const contextSwitchStub = {
      applyActivePair: (): Observable<void> => applyPair$,
    };
    const activeContextStub = { modelId: '' };
    const progressEventsStub = {
      get loadingTasks(): LoadingTask[] {
        return loadingTasks;
      },
    };
    const projectionApiStub = {
      getMeta: (): Observable<ProjectionMeta> => metaProvider(),
      build: (): Observable<ProjectionBuildResponse> => buildProvider(),
    };

    TestBed.configureTestingModule({
      providers: [
        provideRouter([]),
        { provide: ContextSwitchService, useValue: contextSwitchStub },
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: ProgressEventsService, useValue: progressEventsStub },
        { provide: ProjectionApiService, useValue: projectionApiStub },
      ],
    });

    service = TestBed.inject(BrowsePrepService);
    router = TestBed.inject(Router);
    spyOn(router, 'navigate').and.resolveTo(true);
  });

  it('navigates once the projection is already built', () => {
    metaProvider = () => of(meta({ status: 'ready', point_count: 42 }));

    service.prepareAndBrowse(DS);
    expect(service.preparing).toBeTrue();
    applyPair$.next(); // load completes

    expect(router.navigate).toHaveBeenCalledWith(['/browse', DS]);
    expect(service.preparing).toBeFalse();
  });

  it('builds the projection, polls until ready, then navigates', fakeAsync(() => {
    // idle → build (building) → poll → ready
    metaProvider = () => of(meta({ status: 'idle' }));
    buildProvider = () => of({ status: 'building' });

    service.prepareAndBrowse(DS);
    applyPair$.next();

    // After the build kicks off we are projecting and have not navigated.
    expect(service.taskKind(DS)).toBe('projection');
    expect(service.displayTask(DS)?.status).toBe('building');
    expect(router.navigate).not.toHaveBeenCalled();

    // Next poll reports the layout is ready.
    metaProvider = () => of(meta({ status: 'ready', point_count: 7 }));
    tick(1000);

    expect(router.navigate).toHaveBeenCalledWith(['/browse', DS]);
  }));

  it('surfaces a projection build failure inline without navigating', () => {
    metaProvider = () => of(meta({ status: 'idle' }));
    buildProvider = () =>
      throwError(() => new HttpErrorResponse({ status: 409, error: { message: 'no embeddings' } }));

    service.prepareAndBrowse(DS);
    applyPair$.next();

    const task = service.displayTask(DS);
    expect(task?.error).toBe('no embeddings');
    expect(service.ownsTask(task!.task_id)).toBeTrue();
    expect(router.navigate).not.toHaveBeenCalled();
    // Error is terminal-but-dismissable, so the dashboard isn't held loading.
    expect(service.preparing).toBeFalse();
  });

  it('bails out silently when the dataset load failed (SSE shows the error)', () => {
    loadingTasks = [erroredTask()];
    let metaCalled = false;
    metaProvider = () => {
      metaCalled = true;
      return of(meta({}));
    };

    service.prepareAndBrowse(DS);
    applyPair$.next();

    expect(metaCalled).toBeFalse();
    expect(service.displayTask(DS)).toBeNull();
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('defers the load-phase row to the real SSE task', () => {
    service.prepareAndBrowse(DS);
    // No real task yet → placeholder so the row gives immediate feedback.
    expect(service.displayTask(DS)?.status).toBe('loading');
    // Once the SSE task appears, defer to it.
    loadingTasks = [{ ...erroredTask(), status: 'loading', error: '' }];
    expect(service.displayTask(DS)).toBeNull();
  });
});
