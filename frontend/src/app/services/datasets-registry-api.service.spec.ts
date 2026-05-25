import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';

describe('DatasetsRegistryApiService', () => {
  let service: DatasetsRegistryApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DatasetsRegistryApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('getStatus should GET', () => {
    service.getStatus().subscribe((data) => {
      expect(data.loaded).toBeTrue();
      expect(data.num_medias).toBe(10);
    });
    const req = httpMock.expectOne('/api/dataset/status');
    expect(req.request.method).toBe('GET');
    req.flush({ loaded: true, num_medias: 10, has_votes: false });
  });

  it('getRegistry should GET', () => {
    service.getRegistry().subscribe((data) => expect(data.datasets).toBeDefined());
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
});
