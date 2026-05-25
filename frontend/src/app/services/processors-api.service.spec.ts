import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { ProcessorsApiService } from './processors-api.service';

describe('ProcessorsApiService', () => {
  let service: ProcessorsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ProcessorsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

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
});
