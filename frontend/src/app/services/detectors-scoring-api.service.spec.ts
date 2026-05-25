import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorsScoringApiService } from './detectors-scoring-api.service';

describe('DetectorsScoringApiService', () => {
  let service: DetectorsScoringApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DetectorsScoringApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('autoDetect should POST', () => {
    service.autoDetect({}).subscribe();
    const req = httpMock.expectOne('/api/auto-detect');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
