import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { DetectorsRegistryApiService } from './detectors-registry-api.service';
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

  it('setAutofind should PUT to the model registry', () => {
    service.setAutofind('m1', true).subscribe();
    const req = httpMock.expectOne('/api/detectors/registry/m1/autofind');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ autofind: true });
    req.flush({ ok: true });
  });
});
