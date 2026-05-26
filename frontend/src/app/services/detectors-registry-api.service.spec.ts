import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';

describe('DetectorsRegistryApiService', () => {
  let service: DetectorsRegistryApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DetectorsRegistryApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('setAutorun should PUT to the model registry', () => {
    service.setAutorun('m1', true).subscribe();
    const req = httpMock.expectOne('/api/detectors/registry/m1/autorun');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ autorun: true });
    req.flush({ ok: true });
  });
});
