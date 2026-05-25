import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DatasetsCrudApiService } from './datasets-crud-api.service';

describe('DatasetsCrudApiService', () => {
  let service: DatasetsCrudApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DatasetsCrudApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('getImporters should GET', () => {
    service.getImporters().subscribe((data) => expect(data.importers).toBeDefined());
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

  it('clearDataset should POST', () => {
    service.clearDataset().subscribe();
    const req = httpMock.expectOne('/api/dataset/clear');
    expect(req.request.method).toBe('POST');
    req.flush({ ok: true });
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
