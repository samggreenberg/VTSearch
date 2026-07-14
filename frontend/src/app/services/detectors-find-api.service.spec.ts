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

  it('findLabel should POST', () => {
    service.findLabel({ detector_id: 'm1' }).subscribe();
    const req = httpMock.expectOne('/api/find-label');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('find should POST', () => {
    service.find({ dataset_ids: [], detector_ids: [] }).subscribe();
    const req = httpMock.expectOne('/api/find');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
