import { vi } from 'vitest';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController, TestRequest } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { LabelViewComponent } from './label-view.component';
import { LabelSessionService } from '../../services/label-session.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the label view.
 * Phase 2.4 signalized label-view's subscribe/timer/effect
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
      providers: [...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelViewComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    // Destroy first: `fixture.destroy()` runs the component's own `ngOnDestroy`
    // *and* destroys the component-provided `PairScopeService`, which is what
    // cancels the in-flight pair-scoped requests (a bare `ngOnDestroy()` call no
    // longer does). Then drain the timer-driven pollers (labeling-status /
    // votes) and any child-panel reads still in flight; cancelled requests
    // can't be flushed.
    fixture.destroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  // Drain label-view's init loads, holding back only the dataset-status
  // response that is under assertion. The medias/settings reads ride
  // promise-based `rxResource` loaders that issue their GET on a microtask, so
  // we drain with `settleResource()` (macrotask + tick — NOT `whenStable()`,
  // which would deadlock on a loading resource) across a few cycles before any
  // `whenStable()` so every resource is resolved (not loading) by then.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/medias/ids').forEach((req) =>
        req.flush([{ id: 1, media_type: 'audio' }]),
      );
      // The Train window ends any live Find session before it reads the votes
      // (#3212), so the GET below is only issued once this POST answers.
      httpMock.match('/api/find/end-session').forEach((req) =>
        req.flush({ ok: true, ended: false }),
      );
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 80 }));
      httpMock.match('/api/inclusion').forEach((req) => req.flush({ inclusion: 0 }));
      httpMock.match('/api/media-types').forEach((req) => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach((req) => req.flush([]));
      // Include smart/stable/span: the autopilot panel's ngOnChanges feeds this
      // into AutopilotStateService.updateFromLabelingStatus, which reads them.
      httpMock.match('/api/labeling-status').forEach((req) =>
        req.flush({
          good_count: 0,
          bad_count: 0,
          total_count: 0,
          smart: { status: 'green' },
          stable: { status: 'green' },
          span: { status: 'green' },
        }),
      );
    }
  }

  it('renders the dataset name pushed from the /api/dataset/status subscribe, with no manual detectChanges', async () => {
    await flushInit();
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

describe('LabelViewComponent', () => {
  let component: LabelViewComponent;
  let fixture: ComponentFixture<LabelViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [LabelViewComponent],
      providers: [
        ...provideHttpTesting(),
        provideRouter([]),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelViewComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  // Flush the HTTP calls that fire synchronously during the first
  // `fixture.detectChanges()`. label-view's ngOnInit loads medias, votes,
  // settings, dataset status, inclusion; the left panel loads media-types
  // and embedders. The `/api/labeling-status` and polling `/api/votes`
  // requests are driven by `timer(0, …)`, which only fires on a real macrotask
  // after the synchronous test body returns, so they are NOT flushed here —
  // they are drained by `afterEach`'s catch-all instead.
  function flushInitialRequests(): void {
    TestBed.tick();
    // The medias and settings reads ride `rxResource`, whose loader runs in a
    // root effect rather than synchronously during `detectChanges()`; tick so
    // the GETs are actually issued before we match them.
    TestBed.tick();
    // /api/medias/ids
    httpMock.match('/api/medias/ids').forEach(req =>
      req.flush([
        { id: 1, media_type: 'audio' },
        { id: 2, media_type: 'audio' },
      ]),
    );
    // /api/find/end-session (the Train window's hand-off out of a Find
    // session, #3212); loadVotes is chained onto its response, so the GET
    // below only exists after this POST is answered and the tick lands it.
    httpMock.match('/api/find/end-session').forEach(req =>
      req.flush({ ok: true, ended: false }),
    );
    TestBed.tick();
    // /api/votes (label-view loadVotes)
    httpMock.match('/api/votes').forEach(req =>
      req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
    );
    // /api/settings
    httpMock.match('/api/settings').forEach(req =>
      req.flush({ volume: 80 }),
    );
    // /api/dataset/status
    httpMock.match('/api/dataset/status').forEach(req =>
      req.flush({ display_name: 'Test dataset' }),
    );
    // /api/inclusion
    httpMock.match('/api/inclusion').forEach(req =>
      req.flush({ inclusion: 0 }),
    );
    // /api/media-types (left panel)
    httpMock.match('/api/media-types').forEach(req =>
      req.flush({ media_types: [] }),
    );
    // /api/embedders (left panel)
    httpMock.match('/api/embedders').forEach(req =>
      req.flush([]),
    );
  }

  afterEach(async () => {
    // Destroy first. This runs `ngOnDestroy`, tears down the component view +
    // its child views (no template can be re-checked), disposes the component
    // injector — with the medias/settings rxResource loaders bound to it — and
    // destroys the component-provided `PairScopeService`, whose teardown is
    // what cancels the in-flight pair-scoped requests.
    fixture.destroy();
    // Kill the shared VoteStateService poll explicitly. Under zoneless (no
    // zone.js patching test timers) a leftover `timer(0, N)` poll keeps firing
    // real macrotasks across spec boundaries; once the TestBed injector is reset
    // its next tick issues an HTTP request through a destroyed injector (NG0205).
    TestBed.inject(VoteStateService).stopPolling();
    // Drain any outstanding polling requests from right-panel or label-view
    // (the timer-driven /api/votes and /api/labeling-status pollers, the
    // metadata-batch fetch from selectMedia, plus anything a test left in
    // flight). The destroy above unsubscribed the component's own streams,
    // which cancels their in-flight requests; cancelled requests can't be
    // flushed, so skip them. Flush an empty array so the metadata-batch
    // handler (which iterates the body) doesn't choke on a non-iterable.
    httpMock.match(() => true).forEach(req => {
      if (!req.cancelled) req.flush([]);
    });
    // Drain one macrotask while the TestBed injector is still alive. Without
    // zone.js the framework no longer tracks the app's timers/microtasks, so the
    // SSE pollers' `timer(0, N)` first emissions and root-singleton rxResource
    // reloads fire on a macrotask AFTER this teardown; letting them land now
    // (injector alive, the reset happens in the next spec's beforeEach) turns
    // what would be a post-reset NG0205 into a harmless unflushed request.
    await new Promise<void>((resolve) => setTimeout(resolve));
    httpMock.match(() => true).forEach(req => {
      if (!req.cancelled) req.flush([]);
    });
  });

  it('should create', () => {
    flushInitialRequests();
    expect(component).toBeTruthy();
  });

  it('should load medias on init', async () => {
    flushInitialRequests();
    await settleResource();
    expect(component.mediaState.mediasSignal().length).toBe(2);
  });

  it('should load votes on init', () => {
    flushInitialRequests();
    expect(component.voteState.goodVotes.size).toBe(0);
    expect(component.voteState.badVotes.size).toBe(0);
  });

  it('ends a live Find session before it reads the votes (#3212)', () => {
    // Find's bulk presumptions live in the same per-detector vote dicts the
    // Train window trains from, so reading /api/votes first would show the whole
    // collection as voted (and land Autopilot in a terminal phase on arrival).
    const loadVotes = vi.spyOn(TestBed.inject(VoteStateService), 'loadVotes');
    TestBed.tick();
    TestBed.tick();
    expect(loadVotes).not.toHaveBeenCalled();

    const handoff = httpMock.expectOne('/api/find/end-session');
    expect(handoff.request.method).toBe('POST');
    handoff.flush({ ok: true, ended: true });
    TestBed.tick();

    // Only now, against the votes the hand-off restored from the labelset.
    expect(loadVotes).toHaveBeenCalledTimes(1);
  });

  it('snaps both panels tight once medias first render', async () => {
    // Grid layout is unavailable in jsdom, so stub rAF to a no-op: we assert the
    // on-load snap is *kicked off* once content arrives (the divider used to be
    // the only trigger), not that the bounded readiness poll converges — it
    // can't without a real layout engine.
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0);
    const snapSpy = vi.spyOn(
      component as unknown as { snapPanelsOnLoad: () => void },
      'snapPanelsOnLoad',
    );
    flushInitialRequests();
    await settleResource();
    TestBed.tick();
    expect(snapSpy).toHaveBeenCalledTimes(1);
  });

  it('should render 3-panel layout', () => {
    flushInitialRequests();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.panel-left')).toBeTruthy();
    expect(el.querySelector('.panel-center')).toBeTruthy();
    expect(el.querySelector('.panel-right')).toBeTruthy();
  });

  it('should handle text sort', () => {
    flushInitialRequests();
    component.onTextSort('cat');
    expect(component.sortState.sortBusy).toBe(true);

    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat' });
    req.flush({
      results: [{ id: 2, similarity: 0.9 }, { id: 1, similarity: 0.3 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortBusy).toBe(false);
    expect(component.sortState.sortOrder).toEqual([{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }]);
    expect(component.sortState.threshold).toBe(0.5);
  });

  it('should handle learned sort', () => {
    flushInitialRequests();
    // Set votes via vote state service
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    component.onLearnedSort();
    expect(component.sortState.sortBusy).toBe(true);

    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      status: 'done',
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortBusy).toBe(false);
    expect(component.sortState.sortOrder!.length).toBe(2);
  });

  it('should not trigger learned sort without votes', () => {
    flushInitialRequests();
    component.onLearnedSort();
    // No HTTP request should be made
    expect(component.sortState.sortBusy).toBe(false);
  });

  it('should handle media selection', () => {
    flushInitialRequests();
    component.onMediaSelect(2);
    expect(component.mediaState.selectedId()).toBe(2);
  });

  it('should handle sort mode change', () => {
    flushInitialRequests();
    component.onSortModeChange('learned');
    expect(component.sortState.sortMode).toBe('learned');
  });

  it('should handle select mode change to new', () => {
    flushInitialRequests();
    // autoSelectNext (and thus the coverage-atlas fetch for 'new' mode) only
    // runs when there is a sort order to act on, so seed one first.
    component.sortState.setSortResults(
      [{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }],
      0.5,
    );
    component.onSelectModeChange('new');
    expect(component.sortState.selectMode).toBe('new');

    // With a sort order present the diversity fetch POSTs the scores.
    const req = httpMock.expectOne('/api/coverage-atlas/next');
    req.flush({ id: 1, coverage_level: 2.0, exhausted: false });
    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should reselect media when switching select mode to top', () => {
    flushInitialRequests();

    // Set up sort order so autoSelectNext has something to work with
    component.sortState.setSortResults(
      [{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }],
      0.5,
    );

    // Mark id 2 as voted so the top unlabeled is id 1... wait, need to set votes
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [2], bad: [], click_times: {}, learned_scores: {} });

    // Switch to top mode
    component.onSelectModeChange('top');
    expect(component.sortState.selectMode).toBe('top');
    // Should auto-select id 1 (first unlabeled in sort order)
    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should reselect media when switching select mode to hard', () => {
    flushInitialRequests();

    // Set up sort order with threshold
    component.sortState.setSortResults(
      [{ id: 2, score: 0.9 }, { id: 1, score: 0.4 }],
      0.5,
    );

    // Switch to hard mode
    component.onSelectModeChange('hard');
    expect(component.sortState.selectMode).toBe('hard');
    // id 1 (score 0.4) is closer to threshold 0.5 than id 2 (score 0.9)
    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should handle inclusion change', () => {
    flushInitialRequests();
    component.onInclusionChange(5);
    expect(component.sortState.inclusion).toBe(5);

    const req = httpMock.expectOne('/api/inclusion');
    expect(req.request.body).toEqual({ inclusion: 5 });
    req.flush({ inclusion: 5 });
  });

  it('should render center panel component', () => {
    flushInitialRequests();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-center-panel')).toBeTruthy();
  });

  it('should render right panel component', () => {
    flushInitialRequests();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-right-panel')).toBeTruthy();
  });

  it('should resolve selectedMedia from selectedId', async () => {
    flushInitialRequests();
    await settleResource();
    component.onMediaSelect(2);
    expect(component.mediaState.selectedMedia).toBeTruthy();
    expect(component.mediaState.selectedMedia!.id).toBe(2);
  });

  it('should return null selectedMedia when no selection', () => {
    flushInitialRequests();
    expect(component.mediaState.selectedMedia).toBeNull();
  });

  it('should auto-select next unlabeled media after text sort (top mode)', () => {
    flushInitialRequests();
    component.sortState.setSelectMode('top');
    // Mark id 2 as good via vote state
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [2], bad: [], click_times: {}, learned_scores: {} });

    component.onTextSort('test');
    const req = httpMock.expectOne('/api/sort');
    req.flush({
      results: [{ id: 2, similarity: 0.9 }, { id: 1, similarity: 0.3 }],
      threshold: 0.5,
    });

    // Should auto-select id 1 (first unlabeled)
    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should trigger text sort on autopilot start when session has text query', async () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'dog barking';
    flushInitialRequests();
    // The medias list rides an rxResource: its value commits on a microtask and
    // the media-type/autopilot effect runs on the next tick, so drain before the
    // deferred initial sort fires.
    await settleResource();

    // The left panel's ngOnInit already emits autopilotStart when autopilot is
    // enabled; with a text query and medias now loaded, that fires one initial
    // text sort. Drain it so the explicit onAutopilotStart() below is the
    // request under assertion.
    httpMock.match('/api/sort').forEach(req =>
      req.flush({
        results: [{ id: 1, similarity: 0.9 }, { id: 2, similarity: 0.3 }],
        threshold: 0.5,
      }),
    );

    // Simulate a second autopilot start (e.g. user re-entered autopilot).
    component.onAutopilotStart();

    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'dog barking' });
    req.flush({
      results: [{ id: 1, similarity: 0.9 }, { id: 2, similarity: 0.3 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortOrder).toBeTruthy();
    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should defer autopilot text sort until medias are loaded', async () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'cat meowing';

    // Call onAutopilotStart before medias are loaded
    component.onAutopilotStart();
    // No sort request yet (no medias)
    httpMock.expectNone('/api/sort');

    // Now trigger init which loads medias. The medias GET rides an rxResource
    // loader effect, so tick to issue it before flushing.
    TestBed.tick();
    TestBed.tick();
    httpMock.match('/api/medias/ids').forEach(req =>
      req.flush([{ id: 1, media_type: 'audio' }]),
    );
    httpMock.match('/api/votes').forEach(req =>
      req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
    );
    httpMock.match('/api/settings').forEach(req =>
      req.flush({ volume: 80 }),
    );
    httpMock.match('/api/dataset/status').forEach(req =>
      req.flush({ display_name: 'Test dataset' }),
    );
    httpMock.match('/api/inclusion').forEach(req =>
      req.flush({ inclusion: 0 }),
    );
    httpMock.match('/api/media-types').forEach(req =>
      req.flush({ media_types: [] }),
    );
    httpMock.match('/api/embedders').forEach(req =>
      req.flush([]),
    );

    // Drain the rxResource value commit + media-type/autopilot effect so the
    // deferred sort fires.
    await settleResource();

    // Now the deferred sort should fire
    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat meowing' });
    req.flush({
      results: [{ id: 1, similarity: 0.8 }],
      threshold: 0.5,
    });

    expect(component.mediaState.selectedId()).toBe(1);
  });

  it('should not trigger text sort on autopilot start when no text query', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = '';
    flushInitialRequests();

    component.onAutopilotStart();
    httpMock.expectNone('/api/sort');
  });

  describe('no-text dataset gating', () => {
    // Load a dataset whose media were embedded by a vision-only encoder
    // (DINOv3, supports_text=false) so the text-support checks resolve false.
    function loadNoTextDataset(): void {
      TestBed.tick();
      TestBed.tick();
      httpMock.match('/api/medias/ids').forEach(req =>
        req.flush([{ id: 1, media_type: 'image', embedder: 'dinov3' }]),
      );
      httpMock.match('/api/votes').forEach(req =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
      httpMock.match('/api/settings').forEach(req => req.flush({ volume: 80 }));
      httpMock.match('/api/dataset/status').forEach(req =>
        req.flush({ display_name: 'DINOv3 dataset' }),
      );
      httpMock.match('/api/inclusion').forEach(req => req.flush({ inclusion: 0 }));
      httpMock.match('/api/media-types').forEach(req => req.flush({ media_types: [] }));
      httpMock.match('/api/embedders').forEach(req =>
        req.flush({ embedders: [{ name: 'dinov3', supports_text: false }] }),
      );
    }

    it('reports text unsupported and disables autopilot for a text-hint-only detector', async () => {
      const session = TestBed.inject(LabelSessionService);
      session.textQuery = 'a red car';
      session.mediaExample = '';
      loadNoTextDataset();
      await settleResource();

      expect(component.textSupported).toBe(false);
      expect(component.autopilotDisabled).toBe(true);
    });

    it('skips the doomed autopilot text sort on a no-text dataset', async () => {
      const session = TestBed.inject(LabelSessionService);
      session.textQuery = 'a red car';
      session.mediaExample = '';
      loadNoTextDataset();
      await settleResource();

      // The deferred autopilot text sort must be dropped, never sent.
      httpMock.expectNone('/api/sort');
    });

    it('keeps autopilot available when the detector carries a media-example seed', async () => {
      const session = TestBed.inject(LabelSessionService);
      session.textQuery = '';
      session.mediaExample = 'example.jpg';
      // Keep the panel out of autopilot so we assert the gating getters
      // without firing an example sort that auto-start would kick off.
      component.autopilotEnabled.set(false);
      loadNoTextDataset();
      await settleResource();

      expect(component.textSupported).toBe(false);
      expect(component.autopilotDisabled).toBe(false);
    });
  });

  it('should trigger learned sort when autopilot transitions from bad to hard', () => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    const sortState = TestBed.inject(SortStateService);

    // Set up votes so learned sort will fire
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    // Activate autopilot → good phase
    autopilot.activate();
    TestBed.tick();

    // Transition good → bad
    autopilot.checkPhaseTransition(3, 0);
    TestBed.tick();
    expect(autopilot.state.phase).toBe('bad');

    // Transition bad → hard: should trigger learned sort
    autopilot.checkPhaseTransition(3, 4);
    TestBed.tick();
    expect(autopilot.state.phase).toBe('hard');
    expect(sortState.selectMode).toBe('hard');
    expect(sortState.sortMode).toBe('learned');
    expect(sortState.sortBusy).toBe(true);

    // Flush the learned sort request
    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      status: 'done',
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    expect(sortState.sortBusy).toBe(false);
    expect(sortState.threshold).toBe(0.5);
  });

  it('should switch to hard select mode when bouncing from new back to hard', () => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    const sortState = TestBed.inject(SortStateService);

    // Set up votes so learned sort will fire
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    // Activate and advance to new phase
    autopilot.activate();
    TestBed.tick();
    autopilot.checkPhaseTransition(3, 0); // good → bad
    TestBed.tick();
    autopilot.checkPhaseTransition(3, 4); // bad → hard
    TestBed.tick();
    // Flush learned sort from hard transition
    httpMock.expectOne('/api/learned-sort').flush({
      status: 'done',
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    autopilot.updateFromLabelingStatus({
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    });
    autopilot.checkPhaseTransition(10, 10); // hard → new
    TestBed.tick();
    expect(autopilot.state.phase).toBe('new');
    expect(sortState.selectMode).toBe('new');

    // Surprise vote causes smart to drop → bounce back to hard
    autopilot.updateFromLabelingStatus({
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'yellow' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    });
    autopilot.checkPhaseTransition(12, 12); // new → hard
    TestBed.tick();
    expect(autopilot.state.phase).toBe('hard');
    expect(sortState.selectMode).toBe('hard');
    expect(sortState.sortMode).toBe('learned');
    expect(sortState.sortBusy).toBe(true);

    // Flush the learned sort request from bounce-back
    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      status: 'done',
      results: [{ id: 1, score: 0.7 }, { id: 2, score: 0.3 }],
      threshold: 0.5,
    });
    expect(sortState.sortBusy).toBe(false);
  });

  it('should select hard items by index distance, not score distance', () => {
    flushInitialRequests();

    // Scores cluster above the threshold; by score distance, id 4 (0.55) is
    // closest to 0.5, but by index the threshold sits between id 4 and id 5.
    // Both sides of the boundary should get equal consideration.
    component.sortState.setSortResults(
      [
        { id: 1, score: 0.95 },
        { id: 2, score: 0.90 },
        { id: 3, score: 0.80 },
        { id: 4, score: 0.55 },  // index 3, just above threshold
        { id: 5, score: 0.10 },  // index 4, just below threshold
        { id: 6, score: 0.05 },  // index 5
      ],
      0.5,
    );

    // Vote on id 4 (the one right at the boundary). With score-based selection
    // the next pick would be id 3 (score 0.80, dist=0.30) over id 5 (score 0.10,
    // dist=0.40), biasing toward goods. Index-based should pick id 5 (index 4,
    // one step from threshold index 4) over id 3 (index 2, two steps away).
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [4], bad: [], click_times: {}, learned_scores: {} });

    component.onSelectModeChange('hard');
    expect(component.mediaState.selectedId()).toBe(5);
  });

  it('should advance past just-voted item even when vote state is stale', () => {
    flushInitialRequests();

    // Set up sort order in hard mode
    component.sortState.setSortResults(
      [{ id: 2, score: 0.9 }, { id: 1, score: 0.5 }],
      0.5,
    );
    component.sortState.setSelectMode('hard');

    // Simulate voting on id 1 (closest to threshold) but votes not yet loaded
    // (vote state still shows empty; the async loadVotes hasn't returned)
    component.onMediaVoted({ id: 1, vote: 'bad' });

    // Flush the loadVotes triggered by onMediaVoted
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });

    // Even with stale votes (id 1 not yet in badVotes), the selection should
    // have advanced to id 2 because onMediaVoted excludes the just-voted id.
    expect(component.mediaState.selectedId()).toBe(2);
  });

  it('should deactivate autopilot state when switching to manual mode', () => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    autopilot.activate();
    expect(autopilot.running).toBe(true);
    expect(autopilot.state.phase).toBe('good');

    component.onAutopilotStop();
    expect(autopilot.running).toBe(false);
    expect(autopilot.state.phase).toBe('idle');
  });

  it('should not show resort prompt after switching to manual mode', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'test query';
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    autopilot.activate();
    component.onAutopilotStart();

    // Switch to manual mode
    component.onAutopilotStop();

    // Vote many times; resort prompt should never fire because autopilot is off
    for (let i = 0; i < 15; i++) {
      component.onMediaVoted({ id: 1, vote: 'good' });
      httpMock.match('/api/votes').forEach(req =>
        req.flush({ good: [1], bad: [], click_times: {}, learned_scores: {} }),
      );
    }

    expect(component.showResortPrompt()).toBe(false);
  });

  it('should not show resort prompt after phase transitions past good', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'test query';
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    autopilot.activate();
    component.onAutopilotStart();

    // Simulate optimistic votes that push good count past threshold
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1, 2, 3], bad: [], click_times: {}, learned_scores: {} });

    // Phase should transition to 'bad' when checkResortPrompt eagerly checks
    // Vote 11 times (exceeding default resort interval of 10)
    for (let i = 0; i < 11; i++) {
      component.onMediaVoted({ id: 1, vote: 'bad' });
      httpMock.match('/api/votes').forEach(req =>
        req.flush({ good: [1, 2, 3], bad: [4], click_times: {}, learned_scores: {} }),
      );
    }

    // Phase should have transitioned to 'bad', so no resort prompt
    expect(autopilot.state.phase).toBe('bad');
    expect(component.showResortPrompt()).toBe(false);
  });

  it('should re-select autopilot suggestion on refocus', () => {
    flushInitialRequests();

    // Set up sort order and select mode
    component.sortState.setSortResults(
      [{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }],
      0.5,
    );
    component.sortState.setSelectMode('top');

    // Manually select a different media (simulating user clicking right panel)
    component.onMediaSelect(1);
    expect(component.mediaState.selectedId()).toBe(1);

    // Refocus should re-select the autopilot suggestion (top unlabeled = id 2)
    component.onAutopilotRefocus();
    expect(component.mediaState.selectedId()).toBe(2);
  });

  it('should clear stale autopilot state from previous session on init', () => {
    const autopilot = TestBed.inject(AutopilotStateService);
    // Simulate leftover state from a previous detector session
    autopilot.activate();
    autopilot.checkPhaseTransition(3, 0); // advance to 'bad'
    expect(autopilot.state.phase).toBe('bad');

    // Creating a new label-view should clear stale state
    const freshFixture = TestBed.createComponent(LabelViewComponent);
    TestBed.tick();

    // After init, autopilot should be in 'good' phase (cleared then re-activated
    // by the child autopilot panel), NOT stuck in 'bad' from the old session
    expect(autopilot.state.phase).toBe('good');

    freshFixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true);
    freshFixture.destroy();
  });

  it('should clear stale vote state so autopilot starts in good phase', () => {
    const autopilot = TestBed.inject(AutopilotStateService);
    const voteState = TestBed.inject(VoteStateService);

    // Simulate leftover votes from a previous session (e.g. old detector)
    voteState.applyOptimisticState(1, 'good');
    voteState.applyOptimisticState(2, 'good');
    voteState.applyOptimisticState(3, 'good');
    voteState.applyOptimisticState(4, 'bad');
    voteState.applyOptimisticState(5, 'bad');
    voteState.applyOptimisticState(6, 'bad');
    voteState.applyOptimisticState(7, 'bad');
    expect(voteState.goodVotes.size).toBe(3);
    expect(voteState.badVotes.size).toBe(4);

    // Creating a new label-view should clear stale votes before autopilot
    // activates, so it starts in 'good' phase (NOT 'hard'/Refine Boundary)
    const freshFixture = TestBed.createComponent(LabelViewComponent);
    TestBed.tick();

    expect(voteState.goodVotes.size).toBe(0);
    expect(voteState.badVotes.size).toBe(0);
    expect(autopilot.state.phase).toBe('good');

    freshFixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true);
    freshFixture.destroy();
  });

  // Issue #2921: sort work started for one (dataset, detector) pair must not
  // apply its results into whatever pair is active when it finally settles.
  // Detector scoring and learned-sort training both run for minutes, so a
  // context switch mid-run used to leave the old subscription live, calling
  // `applySortWindow` — and then auto-selecting an id that need not exist in
  // the new dataset — against the new pair.
  describe('pair-switch supersession', () => {
    /** Seed an active pair *before* ngOnInit so the `pair$` replay is the
     *  subscription's skipped first emission rather than a spurious reload. */
    function seedPair(): ActiveContextService {
      const activeContext = TestBed.inject(ActiveContextService);
      activeContext.setActivePair('ds1', 'det1');
      return activeContext;
    }

    /** The detector-registry read the switch fires off `modelId$` needs its
     *  real shape; the catch-all `[]` drain elsewhere would break its handler. */
    function flushDetectorRegistry(): void {
      httpMock
        .match((req) => req.url.startsWith('/api/detectors'))
        .forEach((req) => req.flush({ detectors: [] }));
    }

    it('cancels an in-flight detector scoring run when the active pair changes', () => {
      const activeContext = seedPair();
      flushInitialRequests();
      flushDetectorRegistry();

      component.onModelSelected('det1');
      const staleScore = httpMock.expectOne('/api/find-label');
      expect(component.sortState.sortBusy).toBe(true);

      activeContext.setActivePair('ds2', 'det2');
      flushDetectorRegistry();

      // Aborted client-side, so the old pair's ranking can never be installed.
      expect(staleScore.cancelled).toBe(true);
      // That subscription carries no `finalize`, so `reloadForNewPair` resets
      // the busy flag itself — otherwise the overlay would hang forever.
      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortOrder ?? []).toEqual([]);
    });

    it('stops polling a learned-sort job when the active pair changes', async () => {
      const activeContext = seedPair();
      flushInitialRequests();
      flushDetectorRegistry();

      component.voteState.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

      component.onLearnedSort();
      httpMock.expectOne('/api/learned-sort').flush({ status: 'running', job_id: 'job-1' });

      // Positive control: the adaptivePoll job poll is genuinely running.
      await new Promise<void>((resolve) => setTimeout(resolve, 300));
      const firstPoll = httpMock.match((req) => req.url.startsWith('/api/learned-sort/result'));
      expect(firstPoll.length).toBe(1);
      firstPoll[0].flush({ status: 'running' });

      activeContext.setActivePair('ds2', 'det2');
      // Drain the reload's own requests so they can't be mistaken for a poll.
      // The two whose handlers read into the body need their real shape; a
      // bare `[]` from the catch-all would throw inside them.
      flushDetectorRegistry();
      httpMock.match('/api/labeling-status').forEach((req) =>
        req.flush({
          good_count: 0,
          bad_count: 0,
          total_count: 0,
          smart: { status: 'green' },
          stable: { status: 'green' },
          span: { status: 'green' },
        }),
      );
      httpMock.match(() => true).forEach((req) => {
        if (!req.cancelled) req.flush([]);
      });

      // The poll is torn down with the pair it belonged to: no further result
      // reads, so the finished job can never rank the new pair.
      await new Promise<void>((resolve) => setTimeout(resolve, 800));
      httpMock.expectNone((req) => req.url.startsWith('/api/learned-sort/result'));
      expect(component.sortState.sortBusy).toBe(false);
    });
  });

  // Issue #2948: the learned-sort result poll used `timer(200, 500)` +
  // `switchMap`, so every 500ms tick aborted the in-flight GET. A backend
  // slower than the interval — exactly the case while an MLP training job is
  // hogging the process — had every read cancelled, never saw a non-running
  // status, and left the panel on 'Training…' with `sortBusy` stuck true. One
  // transient HTTP error was also fatal, reporting 'Training failed' for a job
  // still running server-side.
  describe('learned-sort job polling', () => {
    /** Kick a learned-sort run off and leave it `running`, with the result poll
     *  live and its first read in flight. */
    function startRunningJob(): void {
      flushInitialRequests();
      component.voteState.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

      component.onLearnedSort();
      httpMock.expectOne('/api/learned-sort').flush({ status: 'running', job_id: 'job-1' });
    }

    /** Result-poll reads issued since the last call (matching consumes them,
     *  so an empty result means "no *new* read was issued"). */
    function resultPolls(): TestRequest[] {
      return httpMock.match((req) => req.url.startsWith('/api/learned-sort/result'));
    }

    it('never cancels a result read that outlives the poll interval', async () => {
      startRunningJob();
      const first = resultPolls();
      expect(first.length).toBe(1);

      // Well past the 500ms fast cadence. The read is still alive (nothing
      // aborted it) and no second read piled up behind it: adaptivePoll waits
      // for each poll to finish before scheduling the next.
      await new Promise<void>((resolve) => setTimeout(resolve, 1500));
      expect(first[0].cancelled).toBe(false);
      expect(resultPolls().length).toBe(0);

      // The slow answer lands and ranks normally — under the old poll it never
      // could, because it was cancelled a second into its life.
      first[0].flush({
        status: 'done',
        results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
        threshold: 0.5,
      });
      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortStatus).toBe('');
      expect(component.sortState.sortOrder!.length).toBe(2);
    });

    it('rides out a transient poll failure instead of failing the run', async () => {
      startRunningJob();
      const first = resultPolls();
      expect(first.length).toBe(1);

      // A network-level failure: the job is untouched server-side, so the run
      // must stay busy rather than report failure.
      first[0].error(new ProgressEvent('error'));
      expect(component.sortState.sortBusy).toBe(true);
      expect(component.sortState.sortStatus).toBe('Training…');

      // The poll keeps ticking and the run completes on the next read.
      await new Promise<void>((resolve) => setTimeout(resolve, 700));
      const second = resultPolls();
      expect(second.length).toBe(1);
      second[0].flush({ status: 'done', results: [{ id: 1, score: 0.8 }], threshold: 0.5 });
      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortStatus).toBe('');
    });

    it('ends the run when the result endpoint reports the job errored', () => {
      startRunningJob();
      // 500 is how the endpoint reports a failed job (there is no `error`
      // status in a 200 body), so it is terminal, not a blip to ride out.
      resultPolls()[0].flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });

      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortStatus).toBe('Training failed');
    });

    it('ends the run when the job is no longer known to the backend', () => {
      startRunningJob();
      // 404 means the job was evicted or never existed; polling it forever
      // would just spin the panel.
      resultPolls()[0].flush({ error: 'Not Found' }, { status: 404, statusText: 'Not Found' });

      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortStatus).toBe('Training job expired');
    });

    it('reports a cancelled job', () => {
      startRunningJob();
      resultPolls()[0].flush({ status: 'cancelled' });

      expect(component.sortState.sortBusy).toBe(false);
      expect(component.sortState.sortStatus).toBe('Cancelled');
    });
  });
});
