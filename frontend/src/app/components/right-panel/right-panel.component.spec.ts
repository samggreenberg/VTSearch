import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RightPanelComponent } from './right-panel.component';
import { VoteStateService } from '../../services/vote-state.service';
import { provideZoneless } from '../../testing/zoneless-testbed';

describe('RightPanelComponent', () => {
  let component: RightPanelComponent;
  let fixture: ComponentFixture<RightPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RightPanelComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
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
    component.trainMode = { model: { name: 'Old', registry_id: 'r1' } };
    await flushInit();

    component.onDetectorRenamed('New Name');
    const req = httpMock.expectOne('/api/detectors/registry/r1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New Name' });
    req.flush({});

    expect(component.trainMode.model.name).toBe('New Name');
    cleanup();
  });

  it('should not rename if no registry_id', async () => {
    component.trainMode = { model: { name: 'Old' } };
    await flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  });

  it('should not rename if no trainMode', async () => {
    component.trainMode = null;
    await flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  });

  it('should render detector context bar when trainMode set', async () => {
    component.trainMode = { model: { name: 'Test Detector', registry_id: 'r1' } };
    await flushInit();
    TestBed.tick();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
    cleanup();
  });

  it('should not render detector context bar when trainMode is null', async () => {
    component.trainMode = null;
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
