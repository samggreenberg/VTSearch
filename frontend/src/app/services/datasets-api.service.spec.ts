import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DatasetsApiService } from './datasets-api.service';

describe('DatasetsApiService', () => {
  let service: DatasetsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DatasetsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('getStatus should GET', () => {
    service.getStatus().subscribe(data => {
      expect(data.loaded).toBeTrue();
      expect(data.num_medias).toBe(10);
    });
    const req = httpMock.expectOne('/api/dataset/status');
    expect(req.request.method).toBe('GET');
    req.flush({ loaded: true, num_medias: 10, has_votes: false });
  });

  it('getImporters should GET', () => {
    service.getImporters().subscribe(data => expect(data.importers).toBeDefined());
    const req = httpMock.expectOne('/api/dataset/importers');
    expect(req.request.method).toBe('GET');
    req.flush({ importers: [] });
  });

  it('getAllImporters should GET', () => {
    service.getAllImporters().subscribe();
    const req = httpMock.expectOne('/api/dataset/all-importers');
    expect(req.request.method).toBe('GET');
    req.flush({ importers: [] });
  });

  it('getMediaTypes should GET', () => {
    service.getMediaTypes().subscribe(data => expect(data.media_types.length).toBeGreaterThan(0));
    const req = httpMock.expectOne('/api/media-types');
    expect(req.request.method).toBe('GET');
    req.flush({ media_types: [{ type_id: 'audio', name: 'Audio' }] });
  });

  it('clearDataset should POST', () => {
    service.clearDataset().subscribe();
    const req = httpMock.expectOne('/api/dataset/clear');
    expect(req.request.method).toBe('POST');
    req.flush({ ok: true });
  });

  it('getRegistry should GET', () => {
    service.getRegistry().subscribe(data => expect(data.datasets).toBeDefined());
    const req = httpMock.expectOne('/api/datasets/registry');
    expect(req.request.method).toBe('GET');
    req.flush({ datasets: [] });
  });

  it('loadRegistered should POST', () => {
    service.loadRegistered('ds1').subscribe();
    const req = httpMock.expectOne('/api/datasets/registry/ds1/load');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('deleteRegistered should DELETE', () => {
    service.deleteRegistered('ds1').subscribe();
    const req = httpMock.expectOne('/api/datasets/registry/ds1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});
  });

  it('renameRegistered should PUT', () => {
    service.renameRegistered('ds1', 'new-name').subscribe();
    const req = httpMock.expectOne('/api/datasets/registry/ds1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'new-name' });
    req.flush({});
  });

  it('loadDemo should POST', () => {
    service.loadDemo('demo1').subscribe();
    const req = httpMock.expectOne('/api/dataset/load-demo');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('runImporter should POST', () => {
    service.runImporter('folder', { path: '/data' }).subscribe();
    const req = httpMock.expectOne('/api/dataset/import/folder');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
