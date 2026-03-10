import { ComponentFixture, TestBed, fakeAsync, tick, discardPeriodicTasks } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RightPanelComponent } from './right-panel.component';
import { VoteStateService } from '../../services/vote-state.service';

describe('RightPanelComponent', () => {
  let component: RightPanelComponent;
  let fixture: ComponentFixture<RightPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RightPanelComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
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

  function flushInit(): void {
    fixture.detectChanges();
    tick(); // Allow timer(0, ...) to fire
    // Settings request
    httpMock.expectOne('/api/settings').flush({
      volume: 1,
      view_mode_right: 'grid',
    });
    // First votes poll
    httpMock.expectOne('/api/votes').flush({
      good: [],
      bad: [],
      click_times: {},
      learned_scores: {},
    });
  }

  it('should create', fakeAsync(() => {
    flushInit();
    expect(component).toBeTruthy();
    cleanup();
  }));

  it('should load view mode from settings on init', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    httpMock.expectOne('/api/settings').flush({ volume: 1, view_mode_right: 'list' });
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    expect(component.viewMode).toBe('list');
    cleanup();
  }));

  it('should default viewMode to grid when not in settings', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    httpMock.expectOne('/api/settings').flush({ volume: 1 });
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    expect(component.viewMode).toBe('grid');
    cleanup();
  }));

  it('should poll for votes on init', fakeAsync(() => {
    fixture.detectChanges();
    tick();
    httpMock.expectOne('/api/settings').flush({ volume: 1 });
    httpMock.expectOne('/api/votes').flush({
      good: [1, 2],
      bad: [3],
      click_times: { '1': 1, '2': 2, '3': 3 },
      learned_scores: { '1': 0.9, '2': 0.8, '3': 0.1 },
    });

    expect(component.goodIds).toEqual([1, 2]);
    expect(component.badIds).toEqual([3]);
    expect(component.clickTimes).toEqual({ '1': 1, '2': 2, '3': 3 });
    expect(component.learnedScores).toEqual({ '1': 0.9, '2': 0.8, '3': 0.1 });
    cleanup();
  }));

  it('should change sort mode', fakeAsync(() => {
    flushInit();
    component.onSortModeChange('name-asc');
    expect(component.sortMode).toBe('name-asc');
    cleanup();
  }));

  it('should emit mediaSelected', fakeAsync(() => {
    flushInit();
    spyOn(component.mediaSelected, 'emit');
    component.onMediaSelected(42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
    cleanup();
  }));

  it('should rename detector via modelsApi', fakeAsync(() => {
    component.trainMode = { model: { name: 'Old', registry_id: 'r1' } };
    flushInit();

    component.onDetectorRenamed('New Name');
    const req = httpMock.expectOne('/api/models/registry/r1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New Name' });
    req.flush({});

    expect(component.trainMode.model.name).toBe('New Name');
    cleanup();
  }));

  it('should not rename if no registry_id', fakeAsync(() => {
    component.trainMode = { model: { name: 'Old' } };
    flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  }));

  it('should not rename if no trainMode', fakeAsync(() => {
    component.trainMode = null;
    flushInit();

    component.onDetectorRenamed('New');
    cleanup();
  }));

  it('should render detector context bar when trainMode set', fakeAsync(() => {
    component.trainMode = { model: { name: 'Test Detector', registry_id: 'r1' } };
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
    cleanup();
  }));

  it('should not render detector context bar when trainMode is null', fakeAsync(() => {
    component.trainMode = null;
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeFalsy();
    cleanup();
  }));

  it('should render both good and bad label lists', fakeAsync(() => {
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const labelLists = el.querySelectorAll('vt-label-list');
    expect(labelLists.length).toBe(2);
    cleanup();
  }));

  it('should render label sort dropdown', fakeAsync(() => {
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-label-sort')).toBeTruthy();
    cleanup();
  }));
});
