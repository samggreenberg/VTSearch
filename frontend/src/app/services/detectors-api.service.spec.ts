import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorsApiService } from './detectors-api.service';

describe('DetectorsApiService', () => {
  let service: DetectorsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DetectorsApiService);
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

  it('autoDetect should POST', () => {
    service.autoDetect({}).subscribe();
    const req = httpMock.expectOne('/api/auto-detect');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('findLabel should POST', () => {
    service.findLabel({ detector_id: 'm1' }).subscribe();
    const req = httpMock.expectOne('/api/find-label');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('getAutorunExtractors should GET', () => {
    service.getAutorunExtractors().subscribe();
    const req = httpMock.expectOne('/api/autorun-extractors');
    expect(req.request.method).toBe('GET');
    req.flush({ extractors: [] });
  });

  it('getAutorunLocalizers should GET', () => {
    service.getAutorunLocalizers().subscribe();
    const req = httpMock.expectOne('/api/autorun-localizers');
    expect(req.request.method).toBe('GET');
    req.flush({ localizers: [] });
  });

  it('find should POST', () => {
    service.find({ datasets: [], models: [] }).subscribe();
    const req = httpMock.expectOne('/api/find');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
