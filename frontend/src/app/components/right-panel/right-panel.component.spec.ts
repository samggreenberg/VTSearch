import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RightPanelComponent } from './right-panel.component';

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

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    // Settings request
    httpMock.expectOne('/api/settings').flush({
      volume: 1,
      show_thumbnails_right: true,
    });
    // First votes poll
    httpMock.expectOne('/api/votes').flush({
      good: [],
      bad: [],
      click_times: {},
      learned_scores: {},
    });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load settings on init', () => {
    fixture.detectChanges();
    const settingsReq = httpMock.expectOne('/api/settings');
    settingsReq.flush({ volume: 1, show_thumbnails_right: false });
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    expect(component.showThumbnails).toBeFalse();
  });

  it('should default showThumbnails to true when not in settings', () => {
    fixture.detectChanges();
    httpMock.expectOne('/api/settings').flush({ volume: 1 });
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    expect(component.showThumbnails).toBeTrue();
  });

  it('should poll for votes on init', () => {
    fixture.detectChanges();
    httpMock.expectOne('/api/settings').flush({ volume: 1 });
    const votesReq = httpMock.expectOne('/api/votes');
    votesReq.flush({
      good: [1, 2],
      bad: [3],
      click_times: { '1': 1, '2': 2, '3': 3 },
      learned_scores: { '1': 0.9, '2': 0.8, '3': 0.1 },
    });

    expect(component.goodIds).toEqual([1, 2]);
    expect(component.badIds).toEqual([3]);
    expect(component.clickTimes).toEqual({ '1': 1, '2': 2, '3': 3 });
    expect(component.learnedScores).toEqual({ '1': 0.9, '2': 0.8, '3': 0.1 });
  });

  it('should change sort mode', () => {
    flushInit();
    component.onSortModeChange('name-asc');
    expect(component.sortMode).toBe('name-asc');
  });

  it('should emit mediaSelected', () => {
    flushInit();
    spyOn(component.mediaSelected, 'emit');
    component.onMediaSelected(42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should rename detector via modelsApi', () => {
    component.trainMode = { model: { name: 'Old', registry_id: 'r1' } };
    flushInit();

    component.onDetectorRenamed('New Name');
    const req = httpMock.expectOne('/api/models/registry/r1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New Name' });
    req.flush({});

    expect(component.trainMode.model.name).toBe('New Name');
  });

  it('should not rename if no registry_id', () => {
    component.trainMode = { model: { name: 'Old' } };
    flushInit();

    component.onDetectorRenamed('New');
    httpMock.expectNone('/api/models/registry');
  });

  it('should not rename if no trainMode', () => {
    component.trainMode = null;
    flushInit();

    component.onDetectorRenamed('New');
    httpMock.expectNone('/api/models/registry');
  });

  it('should render detector context bar when trainMode set', () => {
    component.trainMode = { model: { name: 'Test Detector', registry_id: 'r1' } };
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
  });

  it('should not render detector context bar when trainMode is null', () => {
    component.trainMode = null;
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeFalsy();
  });

  it('should render both good and bad label lists', () => {
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const labelLists = el.querySelectorAll('vt-label-list');
    expect(labelLists.length).toBe(2);
  });

  it('should render label sort dropdown', () => {
    flushInit();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('vt-label-sort')).toBeTruthy();
  });
});
