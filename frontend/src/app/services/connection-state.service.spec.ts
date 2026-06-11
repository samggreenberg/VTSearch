import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { ConnectionStateService } from './connection-state.service';

describe('ConnectionStateService', () => {
  let service: ConnectionStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ConnectionStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('starts online', () => {
    expect(service.isOffline).toBe(false);
  });

  it('stays online below the failure threshold', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(false);
  });

  it('trips offline once consecutive failures reach the threshold', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(true);
  });

  it('a success resets the failure tally', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    service.recordSuccess();
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(false);
  });

  it('recordSuccess clears the offline state', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(true);
    service.recordSuccess();
    expect(service.isOffline).toBe(false);
  });

  it('retry() probes /healthz and comes back online on any response', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(true);

    let retrying = false;
    service.retrying$.subscribe((v) => (retrying = v));

    service.retry();
    expect(retrying).toBe(true);

    const req = httpMock.expectOne('/healthz');
    expect(req.request.method).toBe('GET');
    req.flush('ok');

    expect(service.isOffline).toBe(false);
    expect(retrying).toBe(false);
  });

  it('retry() is a no-op while a probe is already in flight', () => {
    service.retry();
    // A second call while the first probe is still pending must not enqueue
    // another request.
    service.retry();
    const reqs = httpMock.match('/healthz');
    expect(reqs.length).toBe(1);
    reqs[0].flush('ok');
  });
});
