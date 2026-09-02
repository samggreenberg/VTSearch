import { Component, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { Observable, Subject, of, throwError } from 'rxjs';
import { HttpErrorResponse } from '@angular/common/http';

import { BrowsePrepService } from './browse-prep.service';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { ProgressEventsService } from './progress-events.service';
import { ProjectionApiService } from './projection-api.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { LoadingTask } from '../models/api.models';
import type { ProjectionBuildResponse, ProjectionMeta } from '../models/projection.models';

/**
 * Stands in for the dashboard's dataset row, which reads this service's state
 * through plain method calls in its template — `[loadingTask]="browsePrep
 * .displayTask(dataset.id) ?? …"` — with no `AsyncPipe` or `toSignal` bridge
 * anywhere. That is the shape the repaint test below needs to exercise.
 */
@Component({
  selector: 'app-browse-prep-row',
  standalone: true,
  template: `<span class="msg">{{ prep.displayTask('ds1')?.message }}</span>`,
})
class BrowsePrepRowComponent {
  readonly prep = inject(BrowsePrepService);
}

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
      loadingTasks: (): LoadingTask[] => loadingTasks,
    };
    const projectionApiStub = {
      getMeta: (): Observable<ProjectionMeta> => metaProvider(),
      build: (): Observable<ProjectionBuildResponse> => buildProvider(),
    };

    configureZoneless({
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
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
  });

  it('navigates once the projection is already built', () => {
    metaProvider = () => of(meta({ status: 'ready', point_count: 42 }));

    service.prepareAndBrowse(DS);
    expect(service.preparing).toBe(true);
    applyPair$.next(); // load completes

    expect(router.navigate).toHaveBeenCalledWith(['/browse', DS]);
    expect(service.preparing).toBe(false);
  });

  it('builds the projection, polls until ready, then navigates', async () => {
    vi.useFakeTimers();
    try {
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
      await vi.advanceTimersByTimeAsync(1000);

      expect(router.navigate).toHaveBeenCalledWith(['/browse', DS]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('surfaces a projection build failure inline without navigating', () => {
    metaProvider = () => of(meta({ status: 'idle' }));
    buildProvider = () =>
      throwError(() => new HttpErrorResponse({ status: 409, error: { message: 'no embeddings' } }));

    service.prepareAndBrowse(DS);
    applyPair$.next();

    const task = service.displayTask(DS);
    expect(task?.error).toBe('no embeddings');
    expect(service.ownsTask(task!.task_id)).toBe(true);
    expect(router.navigate).not.toHaveBeenCalled();
    // Error is terminal-but-dismissable, so the dashboard isn't held loading.
    expect(service.preparing).toBe(false);
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

    expect(metaCalled).toBe(false);
    expect(service.displayTask(DS)).toBeNull();
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('carries overall_step_end onto the synthetic row, bounding a count-less step', async () => {
    vi.useFakeTimers();
    try {
      metaProvider = () => of(meta({ status: 'idle' }));
      buildProvider = () => of({ status: 'building' });
      service.prepareAndBrowse(DS);
      applyPair$.next();

      // A count-less phase: the backend reports where its slice ends but has no
      // within-phase total to count against. `dataset-card` feeds this row
      // straight to `progressBarState`, which needs `overall_step_end` to render
      // the bounded sweep the Find-side overlay already gets — without it the
      // bar shimmers the parked fill instead.
      metaProvider = () =>
        of(meta({ status: 'building', overall: 0.25, overall_step_end: 0.6, total: 0 }));
      await vi.advanceTimersByTimeAsync(1000);

      expect(service.displayTask(DS)?.overall_step_end).toBe(0.6);
    } finally {
      vi.useRealTimers();
    }
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

/**
 * Regression guard for the zoneless notification path (#3446). The state this
 * service publishes reaches the dashboard *only* through `displayTask()` /
 * `taskKind()` / `preparing` — value-returning accessors, never an `AsyncPipe`
 * or `toSignal` bridge. While that state lived in a `BehaviorSubject`, a poll
 * writing it notified nobody (docs/FRONTEND.md §5: "a missed bridge is a
 * stale-view bug"), and the projection row repainted only when some unrelated
 * signal happened to dirty the view — in practice the 5s SSE heartbeat, against
 * a 1s poll cadence. Backing it with a signal is what makes the read tracked.
 *
 * This spec deliberately holds every other signal still, so the only thing that
 * can repaint the row is the poll itself.
 */
describe('BrowsePrepService: zoneless repaint of the projection row', () => {
  let applyPair$: Subject<void>;
  let metaProvider: () => Observable<ProjectionMeta>;

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

  function msgText(fixture: ComponentFixture<BrowsePrepRowComponent>): string {
    return fixture.nativeElement.querySelector('.msg')?.textContent?.trim() ?? '';
  }

  /**
   * Wait for the row to render *expected*, on real timers. Fake timers are not
   * an option here: `fixture.whenStable()` resolves off a real macrotask, so
   * faking the clock deadlocks the settle rather than speeding it up. Never
   * calls `detectChanges()` — a missing notification stays stale DOM and this
   * fails on the deadline, which is exactly the regression being guarded.
   */
  async function waitForMessage(
    fixture: ComponentFixture<BrowsePrepRowComponent>,
    expected: string,
  ): Promise<void> {
    const deadline = Date.now() + 4000;
    while (Date.now() < deadline) {
      await fixture.whenStable();
      if (msgText(fixture) === expected) return;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    expect(msgText(fixture)).toBe(expected);
  }

  it('repaints on every poll, with no other signal changing', async () => {
    applyPair$ = new Subject<void>();
    metaProvider = () => of(meta({ status: 'building', message: 'Arranging the items…' }));

    configureZoneless({
      imports: [BrowsePrepRowComponent],
      providers: [
        provideRouter([]),
        { provide: ContextSwitchService, useValue: { applyActivePair: () => applyPair$ } },
        { provide: ActiveContextService, useValue: { modelId: '' } },
        // Held constant for the whole test: the SSE heartbeat is exactly the
        // incidental repaint this regression used to hide behind.
        { provide: ProgressEventsService, useValue: { loadingTasks: (): LoadingTask[] => [] } },
        {
          provide: ProjectionApiService,
          useValue: {
            getMeta: (): Observable<ProjectionMeta> => metaProvider(),
            build: (): Observable<ProjectionBuildResponse> => of({ status: 'building' }),
          },
        },
      ],
    });
    vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);

    const fixture = TestBed.createComponent(BrowsePrepRowComponent);
    const service = TestBed.inject(BrowsePrepService);
    service.prepareAndBrowse('ds1');
    applyPair$.next();
    await waitForMessage(fixture, 'Arranging the items…');

    // One poll later the backend reports a new phase. Nothing else in the app
    // has changed, so only a tracked signal read can repaint this row.
    metaProvider = () => of(meta({ status: 'building', message: 'Building the pyramid…' }));
    await waitForMessage(fixture, 'Building the pyramid…');

    service.cancel();
  });
});
