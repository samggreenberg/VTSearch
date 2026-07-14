import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { FindViewComponent } from './find-view.component';
import { SortStateService } from '../../services/sort-state.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the Find view (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 2.5). Phase 2.5 signalized find-view's own subscribe/effect
 * written template-bound state (`datasetName`, `gridGoalWidthLeft`, …, and the
 * `unverifiedSortOrder` computed) AND signalized the shared SortStateService /
 * VoteStateService it binds, so its `sortState.sortBusy`-style getter bindings
 * repaint under zoneless with no per-consumer bridge.
 *
 * Both tests run under a zoneless `TestBed`, drive state through the *production
 * channel* with NO manual `detectChanges()`, then assert on the rendered DOM:
 *  - the dataset name written from the un-bound `/api/dataset/status` subscribe
 *    (a local signal), and
 *  - the scoring overlay gated on `@if (sortState.sortBusy)`, driven by a
 *    `SortStateService` setter — proving the signal-backed service repaints a
 *    getter-bound view (the Phase 2.5 win).
 */
describe('FindViewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<FindViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindViewComponent],
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(FindViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    fixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
    fixture.destroy();
  });

  // Drain find-view's init loads, holding back only the dataset-status response
  // that the first test asserts. The medias/settings reads ride promise-based
  // `rxResource` loaders that issue their GET on a microtask, so drain with
  // `settleResource()` (NOT `whenStable()`, which deadlocks on a loading
  // resource) across a few cycles.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) =>
        req.flush([{ id: 1, media_type: 'audio' }]),
      );
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 80 }));
      httpMock.match('/api/inclusion').forEach((req) => req.flush({ inclusion: 0 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
    }
  }

  it('renders the dataset name pushed from the /api/dataset/status subscribe, no manual detectChanges', async () => {
    await flushInit();
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.dataset-name')).toBeNull();

    // Production channel: the un-bound dataset-status subscribe writes the
    // `datasetName` signal.
    httpMock.expectOne('/api/dataset/status').flush({ display_name: 'Find Canary' });
    await settleZoneless(fixture);

    const nameEl = fixture.nativeElement.querySelector('.dataset-name');
    expect(nameEl).not.toBeNull();
    expect(nameEl!.textContent).toContain('Find Canary');
  });

  it('toggles the scoring overlay from a SortStateService setter, no manual detectChanges', async () => {
    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    const sortState = TestBed.inject(SortStateService);
    // Not busy after init → the `@if (sortState.sortBusy)` overlay is absent.
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).toBeNull();

    // A signal-backed service setter, called outside any bound handler, must
    // schedule CD for the getter-bound `@if` — that is the Phase 2.5 guarantee.
    sortState.setSortBusy(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).not.toBeNull();

    sortState.setSortBusy(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.find-wait-overlay')).toBeNull();
  });

  // Find/Train share the singleton SortStateService, so a fresh entry still
  // holds the previous session's ranking. Against a smaller dataset those stale
  // ids fire a storm of image 404s; ngOnInit resets the ranking before loading.
  it('clears the previous session ranking on a fresh entry', async () => {
    const sortState = TestBed.inject(SortStateService);
    // Seed a ranking from a "previous" (larger) dataset before ngOnInit runs.
    sortState.setSortResults([{ id: 999, score: 0.9 }], 0.5);
    expect(sortState.sortOrder?.length).toBe(1);

    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    // Reset happened before loadMedias(), so no stale id survives to fire a 404.
    expect(sortState.sortOrder ?? []).toEqual([]);
  });

  // Returning from the Browser is the exception: the preserved ranking and the
  // just-recorded verifications are exactly what we keep, so the reset is
  // skipped.
  it('preserves the ranking when returning from the Browser', async () => {
    const sortState = TestBed.inject(SortStateService);
    const browseSubset = TestBed.inject(BrowseSubsetService);
    sortState.setSortResults([{ id: 999, score: 0.9 }], 0.5);
    browseSubset.markReturningToFind();

    await flushInit();
    httpMock.match('/api/dataset/status').forEach((req) => req.flush({ display_name: 'x' }));
    await settleZoneless(fixture);

    expect(sortState.sortOrder?.map((s) => s.id)).toEqual([999]);
  });
});
