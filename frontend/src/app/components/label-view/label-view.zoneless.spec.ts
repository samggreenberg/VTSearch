import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { LabelViewComponent } from './label-view.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the label view (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 2.4). Phase 2.4 signalized label-view's subscribe/timer/effect
 * written template-bound state (`datasetName`, `labelingStatus`,
 * `trainableModelName`, the panel widths, the autopilot flags, `showResortPrompt`,
 * `cropPending`) and bridged the still-Observable `SortStateService` /
 * `VoteStateService` channels it binds into signals via `toSignal`.
 *
 * This spec runs under a zoneless `TestBed` and drives the component through the
 * *production channel* — the `/api/dataset/status` HTTP response, handled in an
 * un-bound `.subscribe()` callback — then asserts on the rendered DOM after
 * `settleZoneless()` with NO manual `detectChanges()`. The subscribe write to the
 * `datasetName` signal is the only thing that can schedule CD for that un-bound
 * chain; were it still a plain field, the left panel's `.dataset-name` header
 * would never appear and this assertion would fail.
 */
describe('LabelViewComponent (zoneless dataset-name canary)', () => {
  let fixture: ComponentFixture<LabelViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [LabelViewComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    fixture.componentInstance.ngOnDestroy();
    // Drain the timer-driven pollers (labeling-status / votes) and any
    // child-panel reads still in flight; cancelled requests can't be flushed.
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
    fixture.destroy();
  });

  it('renders the dataset name pushed from the /api/dataset/status subscribe, with no manual detectChanges', async () => {
    // `TestBed.tick()` runs the initial CD (ngOnInit) and the rxResource loader
    // effect so the init GETs are issued. We must NOT `whenStable()` while the
    // medias/settings resources are loading (a loading rxResource holds the app
    // unstable and would deadlock), so flush them first, holding back only the
    // dataset-status response that is under assertion.
    TestBed.tick();
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

    await settleZoneless(fixture);

    // datasetName starts '' → the left panel's `.dataset-name` node is absent.
    expect(fixture.nativeElement.querySelector('.dataset-name')).toBeNull();

    // Production channel: the un-bound dataset-status subscribe writes the
    // `datasetName` signal.
    httpMock.expectOne('/api/dataset/status').flush({ display_name: 'Canary Dataset' });
    await settleZoneless(fixture);

    // The signal write scheduled CD with no manual pump; the header now reflects
    // it. A plain field would have left this null/empty.
    const nameEl = fixture.nativeElement.querySelector('.dataset-name');
    expect(nameEl).not.toBeNull();
    expect(nameEl!.textContent).toContain('Canary Dataset');
  });
});
