import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { Observable, of, throwError } from 'rxjs';
import { HttpErrorResponse } from '@angular/common/http';

import { BrowseSubsetPrepService } from './browse-subset-prep.service';
import { BrowseSubsetService } from './browse-subset.service';
import { ProjectionApiService } from './projection-api.service';
import { ToastService } from './toast.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import type { ProjectionBuildResponse, ProjectionMeta } from '../models/projection.models';

/**
 * The Find view's Browse button must build its map **in Find**, behind a
 * progress bar, and only then hand off to the browse view — never drop the user
 * into an empty browse window to watch a multi-minute UMAP fit. These tests pin
 * that contract: no navigation until the layout is ready, a live whole-job bar
 * while it builds, and failures that leave the user in Find.
 */
describe('BrowseSubsetPrepService', () => {
  let service: BrowseSubsetPrepService;
  let router: Router;
  let subsetHandoff: BrowseSubsetService;
  let toastErrors: string[];

  // Reassigned per test to script the build/meta responses.
  let metaProvider: () => Observable<ProjectionMeta>;
  let buildProvider: () => Observable<ProjectionBuildResponse>;
  let buildIds: number[][];

  const DS = 'ds1';
  const IDS = [3, 1, 4];

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

  beforeEach(() => {
    metaProvider = () => of(meta({ status: 'building' }));
    buildProvider = () => of({ status: 'building' });
    buildIds = [];
    toastErrors = [];

    const projectionApiStub = {
      getMeta: (): Observable<ProjectionMeta> => metaProvider(),
      buildSubset: (ids: number[]): Observable<ProjectionBuildResponse> => {
        buildIds.push(ids);
        return buildProvider();
      },
    };
    const toastStub = {
      error: (opts: { message: string }) => {
        toastErrors.push(opts.message);
        return 0;
      },
    };

    configureZoneless({
      providers: [
        provideRouter([]),
        { provide: ProjectionApiService, useValue: projectionApiStub },
        { provide: ToastService, useValue: toastStub },
      ],
    });

    service = TestBed.inject(BrowseSubsetPrepService);
    subsetHandoff = TestBed.inject(BrowseSubsetService);
    router = TestBed.inject(Router);
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
  });

  it('navigates straight through when the layout is already fit', () => {
    buildProvider = () => of({ status: 'ready' });

    service.start(DS, IDS);

    expect(buildIds).toEqual([IDS]);
    expect(router.navigate).toHaveBeenCalledWith(['/browse', DS], {
      queryParams: { subset: 1 },
    });
    expect(subsetHandoff.take()).toEqual({ datasetId: DS, ids: IDS });
    expect(service.preparing()).toBe(false);
  });

  it('holds the user in Find with a whole-job bar until the build lands', async () => {
    vi.useFakeTimers();
    try {
      service.start(DS, IDS);

      // Still in Find: the wait is up and nothing has navigated.
      expect(service.preparing()).toBe(true);
      expect(router.navigate).not.toHaveBeenCalled();

      // Mid-build: the bar reads the whole-job fraction, and the phase line
      // names which of the build's steps is running.
      metaProvider = () =>
        of(
          meta({
            status: 'building',
            current: 1,
            total: 2,
            message: 'building pyramid',
            step: 2,
            total_steps: 3,
            overall: 0.5,
          }),
        );
      await vi.advanceTimersByTimeAsync(1000);

      expect(service.bar()).toMatchObject({ value: 0.5, max: 1, indeterminate: false });
      expect(service.count()).toBe('1 / 2');
      expect(service.detail()).toBe('Step 2 of 3 · building pyramid');
      expect(router.navigate).not.toHaveBeenCalled();

      // The layout lands → hand off to the browse view.
      metaProvider = () => of(meta({ status: 'ready', point_count: 3 }));
      await vi.advanceTimersByTimeAsync(1000);

      expect(router.navigate).toHaveBeenCalledWith(['/browse', DS], {
        queryParams: { subset: 1 },
      });
      expect(service.preparing()).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('pulses in place while a phase reports no counts of its own', async () => {
    vi.useFakeTimers();
    try {
      service.start(DS, IDS);
      metaProvider = () =>
        of(
          meta({
            status: 'building',
            message: 'UMAP fit (900 points) — 12s elapsed',
            step: 1,
            total_steps: 3,
            overall: 0,
            overall_step_end: 1 / 3,
          }),
        );
      await vi.advanceTimersByTimeAsync(1000);

      // The UMAP fit has no fraction to report, so the bar keeps its earned
      // fill and sweeps the phase's slice rather than pretending to advance.
      // The slice bound has to survive the meta → ProgressEvent copy: without
      // it this bar has a 0%-wide fill and nothing to animate, leaving the
      // overlay visibly dead for the longest phase of the build.
      expect(service.bar().pulsing).toBe(true);
      expect(service.bar().pulseTo).toBeCloseTo(1 / 3);
      expect(service.count()).toBe('');
      expect(service.detail()).toBe('Step 1 of 3 · UMAP fit (900 points) — 12s elapsed');
    } finally {
      vi.useRealTimers();
    }
  });

  it('leaves the user in Find when the build fails', async () => {
    vi.useFakeTimers();
    try {
      service.start(DS, IDS);
      metaProvider = () => of(meta({ status: 'error', error: 'no embeddings' }));
      await vi.advanceTimersByTimeAsync(1000);

      expect(toastErrors).toEqual(['no embeddings']);
      expect(router.navigate).not.toHaveBeenCalled();
      expect(service.preparing()).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('drops the wait when the build request itself is rejected', () => {
    buildProvider = () =>
      throwError(() => new HttpErrorResponse({ status: 409, error: { message: 'nope' } }));

    service.start(DS, IDS);

    // The global error interceptor already toasted the rejection; all this
    // service owes the user is releasing the Find view.
    expect(service.preparing()).toBe(false);
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('cancelling stops the poll and leaves no stale handoff behind', async () => {
    vi.useFakeTimers();
    try {
      service.start(DS, IDS);
      service.cancel();
      expect(service.preparing()).toBe(false);

      // A late "ready" must not yank a user who walked away into the browser.
      metaProvider = () => of(meta({ status: 'ready', point_count: 3 }));
      await vi.advanceTimersByTimeAsync(5000);

      expect(router.navigate).not.toHaveBeenCalled();
      expect(subsetHandoff.take()).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores an empty selection and a second start while preparing', () => {
    service.start(DS, []);
    expect(buildIds).toEqual([]);
    expect(service.preparing()).toBe(false);

    service.start(DS, IDS);
    service.start(DS, [9, 9]);
    expect(buildIds).toEqual([IDS]);
  });
});
