import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { LabelViewComponent } from './label-view.component';
import { LabelSessionService } from '../../services/label-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';

describe('LabelViewComponent', () => {
  let component: LabelViewComponent;
  let fixture: ComponentFixture<LabelViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelViewComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelViewComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  function flushInitialRequests(): void {
    fixture.detectChanges();
    // Flush /api/medias
    httpMock.expectOne('/api/medias/ids').flush([
      { id: 1, media_type: 'audio' },
      { id: 2, media_type: 'audio' },
    ]);
    // Flush /api/votes (label-view + right-panel both poll)
    httpMock.match('/api/votes').forEach(req =>
      req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
    );
    // Flush /api/settings (label-view + right-panel both request)
    httpMock.match('/api/settings').forEach(req =>
      req.flush({ volume: 80 }),
    );
    // Flush initial labeling-status poll
    httpMock.expectOne('/api/labeling-status').flush({});
  }

  afterEach(() => {
    component.ngOnDestroy();
    // Flush any outstanding polling requests from right-panel or label-view
    httpMock.match(() => true);
  });

  it('should create', () => {
    flushInitialRequests();
    expect(component).toBeTruthy();
  });

  it('should load medias on init', () => {
    flushInitialRequests();
    expect(component.mediaState.medias.length).toBe(2);
  });

  it('should load votes on init', () => {
    flushInitialRequests();
    expect(component.voteState.goodVotes.size).toBe(0);
    expect(component.voteState.badVotes.size).toBe(0);
  });

  it('should render 3-panel layout', () => {
    flushInitialRequests();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.panel-left')).toBeTruthy();
    expect(el.querySelector('.panel-center')).toBeTruthy();
    expect(el.querySelector('.panel-right')).toBeTruthy();
  });

  it('should handle text sort', () => {
    flushInitialRequests();
    component.onTextSort('cat');
    expect(component.sortState.sortBusy).toBeTrue();

    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat' });
    req.flush({
      results: [{ id: 2, similarity: 0.9 }, { id: 1, similarity: 0.3 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortBusy).toBeFalse();
    expect(component.sortState.sortOrder).toEqual([{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }]);
    expect(component.sortState.threshold).toBe(0.5);
  });

  it('should handle learned sort', fakeAsync(() => {
    flushInitialRequests();
    // Set votes via vote state service
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    component.onLearnedSort();
    expect(component.sortState.sortBusy).toBeTrue();

    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortBusy).toBeFalse();
    expect(component.sortState.sortOrder!.length).toBe(2);
  }));

  it('should not trigger learned sort without votes', () => {
    flushInitialRequests();
    component.onLearnedSort();
    // No HTTP request should be made
    expect(component.sortState.sortBusy).toBeFalse();
  });

  it('should handle media selection', () => {
    flushInitialRequests();
    component.onMediaSelect(2);
    expect(component.mediaState.selectedId).toBe(2);
  });

  it('should handle sort mode change', () => {
    flushInitialRequests();
    component.onSortModeChange('learned');
    expect(component.sortState.sortMode).toBe('learned');
  });

  it('should handle select mode change to new', () => {
    flushInitialRequests();
    component.onSelectModeChange('new');
    expect(component.sortState.selectMode).toBe('new');

    const req = httpMock.expectOne('/api/diversity-tree/next');
    req.flush({ id: 1, diversity_level: 2.0, exhausted: false });
    expect(component.mediaState.selectedId).toBe(1);
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
    expect(component.mediaState.selectedId).toBe(1);
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
    expect(component.mediaState.selectedId).toBe(1);
  });

  it('should handle inclusion change', fakeAsync(() => {
    flushInitialRequests();
    component.onInclusionChange(5);
    expect(component.sortState.inclusion).toBe(5);

    const req = httpMock.expectOne('/api/inclusion');
    expect(req.request.body).toEqual({ inclusion: 5 });
    req.flush({ inclusion: 5 });
  }));

  it('should render center panel component', () => {
    flushInitialRequests();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-center-panel')).toBeTruthy();
  });

  it('should render right panel component', () => {
    flushInitialRequests();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-right-panel')).toBeTruthy();
  });

  it('should resolve selectedMedia from selectedId', () => {
    flushInitialRequests();
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
    expect(component.mediaState.selectedId).toBe(1);
  });

  it('should trigger text sort on autopilot start when session has text query', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'dog barking';
    flushInitialRequests();

    // onAutopilotStart is called by left-panel ngOnInit; simulate it
    component.onAutopilotStart();

    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'dog barking' });
    req.flush({
      results: [{ id: 1, similarity: 0.9 }, { id: 2, similarity: 0.3 }],
      threshold: 0.5,
    });

    expect(component.sortState.sortOrder).toBeTruthy();
    expect(component.mediaState.selectedId).toBe(1);
  });

  it('should defer autopilot text sort until medias are loaded', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = 'cat meowing';

    // Call onAutopilotStart before medias are loaded
    component.onAutopilotStart();
    // No sort request yet (no medias)
    httpMock.expectNone('/api/sort');

    // Now trigger init which loads medias
    fixture.detectChanges();
    httpMock.expectOne('/api/medias/ids').flush([
      { id: 1, media_type: 'audio' },
    ]);
    httpMock.match('/api/votes').forEach(req =>
      req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
    );
    httpMock.match('/api/settings').forEach(req =>
      req.flush({ volume: 80 }),
    );
    httpMock.expectOne('/api/labeling-status').flush({});

    // Now the deferred sort should fire
    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat meowing' });
    req.flush({
      results: [{ id: 1, similarity: 0.8 }],
      threshold: 0.5,
    });

    expect(component.mediaState.selectedId).toBe(1);
  });

  it('should not trigger text sort on autopilot start when no text query', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = '';
    flushInitialRequests();

    component.onAutopilotStart();
    httpMock.expectNone('/api/sort');
  });

  it('should trigger learned sort when autopilot transitions from bad to hard', fakeAsync(() => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    const sortState = TestBed.inject(SortStateService);

    // Set up votes so learned sort will fire
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    // Activate autopilot → good phase
    autopilot.activate();
    fixture.detectChanges();

    // Transition good → bad
    autopilot.checkPhaseTransition(3, 0);
    fixture.detectChanges();
    expect(autopilot.state.phase).toBe('bad');

    // Transition bad → hard: should trigger learned sort
    autopilot.checkPhaseTransition(3, 4);
    fixture.detectChanges();
    expect(autopilot.state.phase).toBe('hard');
    expect(sortState.selectMode).toBe('hard');
    expect(sortState.sortMode).toBe('learned');
    expect(sortState.sortBusy).toBeTrue();

    // Flush the learned sort request
    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    expect(sortState.sortBusy).toBeFalse();
    expect(sortState.threshold).toBe(0.5);
  }));

  it('should switch to hard select mode when bouncing from new back to hard', fakeAsync(() => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    const sortState = TestBed.inject(SortStateService);

    // Set up votes so learned sort will fire
    component.voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });

    // Activate and advance to new phase
    autopilot.activate();
    fixture.detectChanges();
    autopilot.checkPhaseTransition(3, 0); // good → bad
    fixture.detectChanges();
    autopilot.checkPhaseTransition(3, 4); // bad → hard
    fixture.detectChanges();
    // Flush learned sort from hard transition
    httpMock.expectOne('/api/learned-sort').flush({
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
    fixture.detectChanges();
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
    fixture.detectChanges();
    expect(autopilot.state.phase).toBe('hard');
    expect(sortState.selectMode).toBe('hard');
    expect(sortState.sortMode).toBe('learned');
    expect(sortState.sortBusy).toBeTrue();

    // Flush the learned sort request from bounce-back
    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      results: [{ id: 1, score: 0.7 }, { id: 2, score: 0.3 }],
      threshold: 0.5,
    });
    expect(sortState.sortBusy).toBeFalse();
  }));

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
    expect(component.mediaState.selectedId).toBe(5);
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
    expect(component.mediaState.selectedId).toBe(2);
  });

  it('should deactivate autopilot state when switching to manual mode', () => {
    flushInitialRequests();

    const autopilot = TestBed.inject(AutopilotStateService);
    autopilot.activate();
    expect(autopilot.running).toBeTrue();
    expect(autopilot.state.phase).toBe('good');

    component.onAutopilotStop();
    expect(autopilot.running).toBeFalse();
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

    expect(component.showResortPrompt).toBeFalse();
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
    expect(component.showResortPrompt).toBeFalse();
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
    expect(component.mediaState.selectedId).toBe(1);

    // Refocus should re-select the autopilot suggestion (top unlabeled = id 2)
    component.onAutopilotRefocus();
    expect(component.mediaState.selectedId).toBe(2);
  });

  it('should clear stale autopilot state from previous session on init', () => {
    const autopilot = TestBed.inject(AutopilotStateService);
    // Simulate leftover state from a previous detector session
    autopilot.activate();
    autopilot.checkPhaseTransition(3, 0); // advance to 'bad'
    expect(autopilot.state.phase).toBe('bad');

    // Creating a new label-view should clear stale state
    const freshFixture = TestBed.createComponent(LabelViewComponent);
    freshFixture.detectChanges();

    // After init, autopilot should be in 'good' phase (cleared then re-activated
    // by the child autopilot panel), NOT stuck in 'bad' from the old session
    expect(autopilot.state.phase).toBe('good');

    freshFixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true);
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
    freshFixture.detectChanges();

    expect(voteState.goodVotes.size).toBe(0);
    expect(voteState.badVotes.size).toBe(0);
    expect(autopilot.state.phase).toBe('good');

    freshFixture.componentInstance.ngOnDestroy();
    httpMock.match(() => true);
  });
});
