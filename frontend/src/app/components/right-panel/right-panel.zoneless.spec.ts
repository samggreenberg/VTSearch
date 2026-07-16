import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';

import { RightPanelComponent } from './right-panel.component';
import { VoteStateService } from '../../services/vote-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the right panel (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 2.5/2.8). The panel used to mirror six `VoteStateService`
 * observables into plain fields via `subscribe`; Phase 2.5 signalized the service
 * and replaced those mirrors with `computed`s over its getters. This drives the
 * vote state through the production channel (`applyOptimisticState`, the same
 * signal write a vote click makes) and asserts the rendered DOM repaints with NO
 * manual `detectChanges()` — the Find-mode Browse action enables only when
 * `goodIds().length > 0`, so a stale mirror would leave it disabled.
 */
describe('RightPanelComponent (zoneless votes canary)', () => {
  let fixture: ComponentFixture<RightPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [RightPanelComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(RightPanelComponent);
    httpMock = TestBed.inject(HttpTestingController);
    // Find mode renders the good bucket with the Browse/To-Dataset/Export
    // actions whose disabled state reads the `goodIds()` computed.
    fixture.componentRef.setInput('mode', 'find');
    fixture.componentRef.setInput('medias', [
      { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'x', custom_metadata: {} },
    ]);
  });

  afterEach(() => {
    fixture.componentInstance.ngOnDestroy();
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    });
    fixture.destroy();
  });

  function browseButton(): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('button[aria-label="Browse"]');
  }

  // Drain init: the settings rxResource loader and the first votes poll. Hold
  // nothing back; the state change under test is driven without HTTP below.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 1 }));
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
    }
  }

  it('enables the Find Browse action when a good vote lands, no manual detectChanges', async () => {
    await flushInit();
    await settleZoneless(fixture);

    // No goods yet → the Browse action is disabled.
    const btn = browseButton();
    expect(btn).not.toBeNull();
    expect(btn!.disabled).toBe(true);

    // Production channel: an optimistic good vote writes the goodVotes signal
    // (the same path a vote click takes), with no HTTP.
    TestBed.inject(VoteStateService).applyOptimisticState(1, 'good');
    await settleZoneless(fixture);

    // The `goodIds()` computed updated and the getter-bound `[disabled]`
    // repainted with no manual pump. A stale mirror would keep it disabled.
    expect(browseButton()!.disabled).toBe(false);
  });
});

describe('RightPanelComponent', () => {
  let component: RightPanelComponent;
  let fixture: ComponentFixture<RightPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [RightPanelComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(RightPanelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  function cleanup(): void {
    // Destroy component to cancel all subscriptions, then flush any outstanding
    component.ngOnDestroy();
    const voteState = TestBed.inject(VoteStateService);
    voteState.stopPolling();
    httpMock.match(() => true); // discard any pending requests
  }

  async function flushInit(): Promise<void> {
    TestBed.tick();
    // Allow the votes poll's `timer(0, …)` first emission to fire on a real
    // macrotask (and drain microtasks) so its GET is issued.
    await new Promise<void>((resolve) => setTimeout(resolve));
    // Settings request
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({
      volume: 1,
    });
    // First votes poll
    httpMock.expectOne('/api/votes').flush({
      good: [],
      bad: [],
      click_times: {},
      learned_scores: {},
    });
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
    cleanup();
  });

  it('should poll for votes on init', async () => {
    TestBed.tick();
    await new Promise<void>((resolve) => setTimeout(resolve));
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({ volume: 1 });
    httpMock.expectOne('/api/votes').flush({
      good: [1, 2],
      bad: [3],
      click_times: { '1': 1, '2': 2, '3': 3 },
      learned_scores: { '1': 0.9, '2': 0.8, '3': 0.1 },
    });

    expect(component.goodIds()).toEqual([1, 2]);
    expect(component.badIds()).toEqual([3]);
    expect(component.clickTimes()).toEqual({ '1': 1, '2': 2, '3': 3 });
    expect(component.learnedScores()).toEqual({ '1': 0.9, '2': 0.8, '3': 0.1 });
    cleanup();
  });

  it('should change sort mode', async () => {
    await flushInit();
    component.onSortModeChange('name-asc');
    expect(component.sortMode).toBe('name-asc');
    cleanup();
  });

  it('should emit mediaSelected', async () => {
    await flushInit();
    vi.spyOn(component.mediaSelected, 'emit');
    component.onMediaSelected(42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
    cleanup();
  });

  it('should rename detector via detectors-registry-api', async () => {
    fixture.componentRef.setInput('trainMode', { model: { name: 'Old', registry_id: 'r1' } });
    await flushInit();

    component.onDetectorRenamed('New Name');
    const req = httpMock.expectOne('/api/detectors/registry/r1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New Name' });
    req.flush({});

    expect(component.trainMode()!.model.name).toBe('New Name');
    cleanup();
  });

  it('should not rename if no registry_id', async () => {
    fixture.componentRef.setInput('trainMode', { model: { name: 'Old' } });
    await flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  });

  it('should not rename if no trainMode', async () => {
    fixture.componentRef.setInput('trainMode', null);
    await flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  });

  it('should render detector context bar when trainMode set', async () => {
    fixture.componentRef.setInput('trainMode', { model: { name: 'Test Detector', registry_id: 'r1' } });
    await flushInit();
    TestBed.tick();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
    cleanup();
  });

  it('should not render detector context bar when trainMode is null', async () => {
    fixture.componentRef.setInput('trainMode', null);
    await flushInit();
    TestBed.tick();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeFalsy();
    cleanup();
  });

  it('should render both good and bad label lists', async () => {
    await flushInit();
    TestBed.tick();

    const el = fixture.nativeElement as HTMLElement;
    const labelLists = el.querySelectorAll('vt-label-list');
    expect(labelLists.length).toBe(2);
    cleanup();
  });

  it('should render label sort dropdown', async () => {
    await flushInit();
    TestBed.tick();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-label-sort')).toBeTruthy();
    cleanup();
  });
});
