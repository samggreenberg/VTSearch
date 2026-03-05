import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { LabelViewComponent } from './label-view.component';
import { LabelSessionService } from '../../services/label-session.service';

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
    httpMock.expectOne('/api/medias').flush([
      { id: 1, type: 'audio', duration: 5, file_size: 1024, filename: 'a.wav', category: '', md5: 'a1' },
      { id: 2, type: 'audio', duration: 3, file_size: 512, filename: 'b.wav', category: '', md5: 'b2' },
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
    expect(component.medias.length).toBe(2);
  });

  it('should load votes on init', () => {
    flushInitialRequests();
    expect(component.goodVotes.size).toBe(0);
    expect(component.badVotes.size).toBe(0);
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
    expect(component.sortBusy).toBeTrue();

    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat' });
    req.flush({
      results: [{ id: 2, similarity: 0.9 }, { id: 1, similarity: 0.3 }],
      threshold: 0.5,
    });

    expect(component.sortBusy).toBeFalse();
    expect(component.sortOrder).toEqual([{ id: 2, score: 0.9 }, { id: 1, score: 0.3 }]);
    expect(component.threshold).toBe(0.5);
  });

  it('should handle learned sort', fakeAsync(() => {
    flushInitialRequests();
    component.goodVotes = new Set([1]);
    component.badVotes = new Set([2]);

    component.onLearnedSort();
    expect(component.sortBusy).toBeTrue();

    const req = httpMock.expectOne('/api/learned-sort');
    req.flush({
      results: [{ id: 1, score: 0.8 }, { id: 2, score: 0.2 }],
      threshold: 0.5,
    });

    expect(component.sortBusy).toBeFalse();
    expect(component.sortOrder!.length).toBe(2);
  }));

  it('should not trigger learned sort without votes', () => {
    flushInitialRequests();
    component.goodVotes = new Set();
    component.badVotes = new Set();
    component.onLearnedSort();
    // No HTTP request should be made
    expect(component.sortBusy).toBeFalse();
  });

  it('should handle media selection', () => {
    flushInitialRequests();
    component.onMediaSelect(2);
    expect(component.selectedId).toBe(2);
  });

  it('should handle sort mode change', () => {
    flushInitialRequests();
    component.onSortModeChange('learned');
    expect(component.sortMode).toBe('learned');
  });

  it('should handle select mode change to new', () => {
    flushInitialRequests();
    component.onSelectModeChange('new');
    expect(component.selectMode).toBe('new');

    const req = httpMock.expectOne('/api/diversity-tree/next');
    req.flush({ id: 1, diversity_level: 2.0, exhausted: false });
    expect(component.selectedId).toBe(1);
  });

  it('should handle inclusion change', fakeAsync(() => {
    flushInitialRequests();
    component.onInclusionChange(5);
    expect(component.inclusion).toBe(5);

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
    expect(component.selectedMedia).toBeTruthy();
    expect(component.selectedMedia!.id).toBe(2);
  });

  it('should return null selectedMedia when no selection', () => {
    flushInitialRequests();
    expect(component.selectedMedia).toBeNull();
  });

  it('should auto-select next unlabeled media after text sort (top mode)', () => {
    flushInitialRequests();
    component.selectMode = 'top';
    component.goodVotes = new Set([2]);

    component.onTextSort('test');
    const req = httpMock.expectOne('/api/sort');
    req.flush({
      results: [{ id: 2, similarity: 0.9 }, { id: 1, similarity: 0.3 }],
      threshold: 0.5,
    });

    // Should auto-select id 1 (first unlabeled)
    expect(component.selectedId).toBe(1);
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

    expect(component.sortOrder).toBeTruthy();
    expect(component.selectedId).toBe(1);
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
    httpMock.expectOne('/api/medias').flush([
      { id: 1, type: 'audio', duration: 5, file_size: 1024, filename: 'a.wav', category: '', md5: 'a1' },
    ]);
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    httpMock.expectOne('/api/settings').flush({ volume: 80 });
    httpMock.expectOne('/api/labeling-status').flush({});

    // Now the deferred sort should fire
    const req = httpMock.expectOne('/api/sort');
    expect(req.request.body).toEqual({ text: 'cat meowing' });
    req.flush({
      results: [{ id: 1, similarity: 0.8 }],
      threshold: 0.5,
    });

    expect(component.selectedId).toBe(1);
  });

  it('should not trigger text sort on autopilot start when no text query', () => {
    const session = TestBed.inject(LabelSessionService);
    session.textQuery = '';
    flushInitialRequests();

    component.onAutopilotStart();
    httpMock.expectNone('/api/sort');
  });
});
