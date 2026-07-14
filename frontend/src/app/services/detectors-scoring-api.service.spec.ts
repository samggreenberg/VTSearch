import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { DetectorsScoringApiService } from './detectors-scoring-api.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('DetectorsScoringApiService', () => {
  let service: DetectorsScoringApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
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
