import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { SortingApiService } from './sorting-api.service';

describe('SortingApiService', () => {
  let service: SortingApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SortingApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('sort should POST text query', () => {
    service.sort({ text: 'test' }).subscribe(data => {
      expect(data.results).toBeDefined();
      expect(data.threshold).toBeDefined();
    });
    const req = httpMock.expectOne('/api/sort');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ text: 'test' });
    req.flush({ results: [{ id: 1, similarity: 0.9 }], threshold: 0.5 });
  });

  it('learnedSort should POST and return a job envelope', () => {
    service.learnedSort().subscribe(data => {
      expect(data.job_id).toBeDefined();
      expect(data.status).toBe('done');
      expect(data.results).toBeDefined();
    });
    const req = httpMock.expectOne('/api/learned-sort');
    expect(req.request.method).toBe('POST');
    req.flush({ job_id: 'xyz', status: 'done', results: [], threshold: 0.5 });
  });

  it('getLearnedSortResult should GET the result endpoint with the job_id', () => {
    service.getLearnedSortResult('xyz').subscribe();
    const req = httpMock.expectOne((r) => r.url === '/api/learned-sort/result');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.get('job_id')).toBe('xyz');
    req.flush({ job_id: 'xyz', status: 'done', results: [], threshold: 0.5 });
  });

  it('getVotes should GET', () => {
    service.getVotes().subscribe(data => {
      expect(data.good).toBeDefined();
      expect(data.bad).toBeDefined();
      expect(data.click_times).toBeDefined();
      expect(data.learned_scores).toBeDefined();
    });
    const req = httpMock.expectOne('/api/votes');
    expect(req.request.method).toBe('GET');
    req.flush({ good: [1], bad: [2], click_times: {}, learned_scores: {} });
  });

  it('clearVotes should POST', () => {
    service.clearVotes().subscribe();
    const req = httpMock.expectOne('/api/votes/clear');
    expect(req.request.method).toBe('POST');
    req.flush({ ok: true });
  });

  it('getInclusion should GET', () => {
    service.getInclusion().subscribe(data => expect(data.inclusion).toBe(3));
    const req = httpMock.expectOne('/api/inclusion');
    expect(req.request.method).toBe('GET');
    req.flush({ inclusion: 3 });
  });

  it('setInclusion should POST', () => {
    service.setInclusion(5).subscribe();
    const req = httpMock.expectOne('/api/inclusion');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ inclusion: 5 });
    req.flush({ inclusion: 5 });
  });

  it('getSafeThresholds should GET', () => {
    service.getSafeThresholds().subscribe(data => expect(data.safe_thresholds).toBeTrue());
    const req = httpMock.expectOne('/api/safe-thresholds');
    expect(req.request.method).toBe('GET');
    req.flush({ safe_thresholds: true });
  });

  it('setSafeThresholds should POST', () => {
    service.setSafeThresholds(false).subscribe();
    const req = httpMock.expectOne('/api/safe-thresholds');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ safe_thresholds: false });
    req.flush({ safe_thresholds: false });
  });

  it('exportLabels should GET', () => {
    service.exportLabels().subscribe(data => expect(data.labels).toBeDefined());
    const req = httpMock.expectOne('/api/labels/export');
    expect(req.request.method).toBe('GET');
    req.flush({ labels: [] });
  });

  it('exportLabels with goodsOnly should include param', () => {
    service.exportLabels(true).subscribe();
    const req = httpMock.expectOne(r => r.url === '/api/labels/export');
    expect(req.request.params.get('goods_only')).toBe('true');
    req.flush({ labels: [] });
  });

  it('importLabels should POST', () => {
    service.importLabels({ labels: [{ md5: 'abc', label: 'good' }] }).subscribe(data => {
      expect(data.applied).toBe(1);
    });
    const req = httpMock.expectOne('/api/labels/import');
    expect(req.request.method).toBe('POST');
    req.flush({ applied: 1, skipped: 0 });
  });

  it('fillFromSort should POST', () => {
    service.fillFromSort({ sort_results: [], threshold: 0.5, sides: 'both', confirm: false }).subscribe();
    const req = httpMock.expectOne('/api/labels/fill-from-sort');
    expect(req.request.method).toBe('POST');
    req.flush({ good_count: 0, bad_count: 0 });
  });

  it('getTextsortSuggestions should GET', () => {
    service.getTextsortSuggestions().subscribe(data => expect(data.suggestions).toEqual(['test']));
    const req = httpMock.expectOne('/api/textsort-suggestions');
    expect(req.request.method).toBe('GET');
    req.flush({ suggestions: ['test'] });
  });

  it('addTextsortSuggestion should POST', () => {
    service.addTextsortSuggestion('hello').subscribe();
    const req = httpMock.expectOne('/api/textsort-suggestions');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ text: 'hello' });
    req.flush({ ok: true });
  });

  it('getLabelingStatus should GET', () => {
    service.getLabelingStatus().subscribe();
    const req = httpMock.expectOne('/api/labeling-status');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('getDiversityTreeNext without scores should GET', () => {
    service.getDiversityTreeNext().subscribe(data => expect(data.id).toBeNull());
    const req = httpMock.expectOne('/api/diversity-tree/next');
    expect(req.request.method).toBe('GET');
    req.flush({ id: null, diversity_level: 0, exhausted: false });
  });

  it('getDiversityTreeNext with scores should POST', () => {
    service.getDiversityTreeNext({ '1': 0.9 }, 0.5).subscribe();
    const req = httpMock.expectOne('/api/diversity-tree/next');
    expect(req.request.method).toBe('POST');
    req.flush({ id: 1, diversity_level: 1, exhausted: false });
  });

  it('getServerMediaFiles should GET', () => {
    service.getServerMediaFiles().subscribe();
    const req = httpMock.expectOne('/api/server-media-files');
    expect(req.request.method).toBe('GET');
    req.flush({ files: [] });
  });

  it('exampleSortServer should POST', () => {
    service.exampleSortServer({ filename: 'test.wav' }).subscribe();
    const req = httpMock.expectOne('/api/example-sort-server');
    expect(req.request.method).toBe('POST');
    req.flush({ results: [], threshold: 0 });
  });
});
