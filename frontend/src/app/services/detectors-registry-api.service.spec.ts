import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { DATASET_OVERRIDE } from '../interceptors/active-context.interceptor';
import { provideHttpTesting } from '../testing/test-providers';

describe('DetectorsRegistryApiService', () => {
  let service: DetectorsRegistryApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(DetectorsRegistryApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('loadDetector names the dataset it loads for, with no detector (#5358)', () => {
    service.loadDetector('m1', 'd1').subscribe();
    const req = httpMock.expectOne('/api/detectors/registry/load');
    expect(req.request.body).toEqual({ detector_id: 'm1' });
    expect(req.request.context.get(DATASET_OVERRIDE)).toBe('d1');
    req.flush({ ok: true, message: '' });
  });

  it('loadDetector without a dataset leaves the active pair to the interceptor', () => {
    service.loadDetector('m1').subscribe();
    const req = httpMock.expectOne('/api/detectors/registry/load');
    expect(req.request.context.get(DATASET_OVERRIDE)).toBe('');
    req.flush({ ok: true, message: '' });
  });

  it('setAutofind should PUT to the model registry', () => {
    service.setAutofind('m1', true).subscribe();
    const req = httpMock.expectOne('/api/detectors/registry/m1/autofind');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ autofind: true });
    req.flush({ ok: true });
  });
});
