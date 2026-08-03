import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { ConnectionStateService } from './connection-state.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('ConnectionStateService', () => {
  let service: ConnectionStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
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

    service.retry();
    expect(service.retrying()).toBe(true);

    const req = httpMock.expectOne('/healthz');
    expect(req.request.method).toBe('GET');
    req.flush('ok');

    expect(service.isOffline).toBe(false);
    expect(service.retrying()).toBe(false);
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

  // --- recordStreamFailure(): classifying ambiguous SSE errors (#2816) ---

  it('recordStreamFailure() probes /healthz instead of counting a failure', () => {
    // Three SSE errors in a row used to trip the breaker even though the
    // backend was answering (503 slot-cap rejections). Now each error only
    // fires a probe; a healthy answer keeps the app online.
    service.recordStreamFailure();
    const req = httpMock.expectOne('/healthz');
    req.flush('ok');
    service.recordStreamFailure();
    httpMock.expectOne('/healthz').flush('ok');
    service.recordStreamFailure();
    httpMock.expectOne('/healthz').flush('ok');
    expect(service.isOffline).toBe(false);
  });

  it('recordStreamFailure() sends one probe at a time', () => {
    service.recordStreamFailure();
    service.recordStreamFailure();
    const reqs = httpMock.match('/healthz');
    expect(reqs.length).toBe(1);
    reqs[0].flush('ok');
    // Once the probe settles, a new stream error may probe again.
    service.recordStreamFailure();
    httpMock.expectOne('/healthz').flush('ok');
  });

  it('recordStreamFailure() does not probe while offline', () => {
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    service.recordNetworkFailure();
    expect(service.isOffline).toBe(true);
    // Recovery stays manual: no automatic probing once the breaker tripped.
    service.recordStreamFailure();
    httpMock.expectNone('/healthz');
  });

  it('recordStreamFailure() probes again after a failed probe settles', () => {
    // A network-level probe failure (backend really gone) must not wedge the
    // in-flight flag; the next stream error probes again.
    service.recordStreamFailure();
    httpMock.expectOne('/healthz').error(new ProgressEvent('error'));
    service.recordStreamFailure();
    httpMock.expectOne('/healthz').flush('ok');
  });
});
