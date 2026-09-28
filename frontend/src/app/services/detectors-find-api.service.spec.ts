import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { DetectorsFindApiService } from './detectors-find-api.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('DetectorsFindApiService', () => {
  let service: DetectorsFindApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(DetectorsFindApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('getAutorunRun should GET the run by id', () => {
    service.getAutorunRun('_autorun_1').subscribe();
    const req = httpMock.expectOne('/api/autorun/runs/_autorun_1');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('findLabel should POST', () => {
    service.findLabel({ detector_id: 'm1' }).subscribe();
    const req = httpMock.expectOne('/api/find-label');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
