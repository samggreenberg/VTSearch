import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { HttpClient, HttpContext, HttpErrorResponse } from '@angular/common/http';

import { errorInterceptor, SKIP_ERROR_TOAST } from './error.interceptor';
import { ActiveContextService } from '../services/active-context.service';
import { CONNECTION_PROBE, ConnectionStateService } from '../services/connection-state.service';
import { ToastService } from '../services/toast.service';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The error interceptor is the global error-to-toast funnel and the
 * connection circuit-breaker chokepoint. It carries the most logic of the
 * four interceptors, so these tests exercise: error-body parsing, context
 * enrichment, the SKIP_ERROR_TOAST opt-out, the silent status-0 path, and
 * the offline breaker (suppression + probe passthrough).
 */
describe('errorInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let toast: { error: ReturnType<typeof vi.fn> };
  let ctx: ActiveContextService;
  let connection: ConnectionStateService;

  beforeEach(() => {
    toast = { error: vi.fn() };
    TestBed.configureTestingModule({
      providers: [
        ...provideHttpTesting(errorInterceptor),
        { provide: ToastService, useValue: toast },
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
    ctx = TestBed.inject(ActiveContextService);
    connection = TestBed.inject(ConnectionStateService);
  });

  afterEach(() => httpMock.verify());

  function tripOffline(): void {
    connection.recordNetworkFailure();
    connection.recordNetworkFailure();
    connection.recordNetworkFailure();
  }

  it('pushes a structured error toast for a non-zero HTTP status', () => {
    let caught: HttpErrorResponse | undefined;
    http.get('/api/foo').subscribe({ error: (e: HttpErrorResponse) => (caught = e) });
    httpMock
      .expectOne('/api/foo')
      .flush(
        { code: 500, status: 'Internal Server Error', message: 'Boom', detail: 'more detail', request_id: 'req-1' },
        { status: 500, statusText: 'Server Error' },
      );

    expect(toast.error).toHaveBeenCalledTimes(1);
    const arg = toast.error.mock.calls[0][0];
    expect(arg.message).toBe('Boom');
    expect(arg.detail).toBe('more detail');
    expect(arg.dedupKey).toBe('http:500:Boom');
    expect(arg.errorContext.status).toBe(500);
    expect(arg.errorContext.statusText).toBe('Server Error');
    expect(arg.errorContext.method).toBe('GET');
    expect(arg.errorContext.url).toBe('/api/foo');
    expect(arg.errorContext.requestId).toBe('req-1');
    // The failure still propagates to the caller unchanged.
    expect(caught).toBeInstanceOf(HttpErrorResponse);
    expect(caught?.status).toBe(500);
  });

  it('enriches the error context with the active dataset/detector ids', () => {
    ctx.setActive('ds-9', 'mdl-3');
    http.get('/api/foo').subscribe({ error: () => {} });
    httpMock.expectOne('/api/foo').flush({ message: 'x' }, { status: 400, statusText: 'Bad Request' });

    const arg = toast.error.mock.calls[0][0];
    expect(arg.errorContext.datasetId).toBe('ds-9');
    expect(arg.errorContext.detectorId).toBe('mdl-3');
  });

  it('falls back to the X-Request-Id header when the body has no request_id', () => {
    http.get('/api/foo').subscribe({ error: () => {} });
    httpMock
      .expectOne('/api/foo')
      .flush(
        { message: 'x' },
        { status: 500, statusText: 'err', headers: { 'X-Request-Id': 'hdr-1' } },
      );

    expect(toast.error.mock.calls[0][0].errorContext.requestId).toBe('hdr-1');
  });

  it('surfaces unknown top-level body fields under extra', () => {
    http.get('/api/foo').subscribe({ error: () => {} });
    httpMock
      .expectOne('/api/foo')
      .flush(
        { code: 422, status: 'Unprocessable Content', message: 'x', missing_fields: ['a', 'b'] },
        { status: 422, statusText: 'err' },
      );

    // ``code`` and ``status`` are envelope furniture the toast already
    // renders from the HttpErrorResponse, so they stay out of ``extra``.
    expect(toast.error.mock.calls[0][0].errorContext.extra).toEqual({
      missing_fields: ['a', 'b'],
    });
  });

  it('captures a non-JSON string body as rawBody and uses a default message', () => {
    http.get('/api/foo', { responseType: 'text' }).subscribe({ error: () => {} });
    httpMock
      .expectOne('/api/foo')
      .flush('<html>503</html>', { status: 503, statusText: 'Service Unavailable' });

    const arg = toast.error.mock.calls[0][0];
    expect(arg.errorContext.rawBody).toBe('<html>503</html>');
    expect(arg.message).toBe('Request failed (503 Service Unavailable).');
  });

  it('suppresses the toast when SKIP_ERROR_TOAST is set but still propagates the error', () => {
    let caught: HttpErrorResponse | undefined;
    http
      .get('/api/foo', { context: new HttpContext().set(SKIP_ERROR_TOAST, true) })
      .subscribe({ error: (e: HttpErrorResponse) => (caught = e) });
    httpMock.expectOne('/api/foo').flush({ message: 'x' }, { status: 500, statusText: 'err' });

    expect(toast.error).not.toHaveBeenCalled();
    expect(caught?.status).toBe(500);
  });

  it('stays silent on a network error (status 0) and never toasts', () => {
    let caught: HttpErrorResponse | undefined;
    http.get('/api/foo').subscribe({ error: (e: HttpErrorResponse) => (caught = e) });
    httpMock.expectOne('/api/foo').error(new ProgressEvent('error'));

    expect(toast.error).not.toHaveBeenCalled();
    expect(caught).toBeInstanceOf(HttpErrorResponse);
    expect(caught?.status).toBe(0);
  });

  it('trips the connection breaker after repeated network failures', () => {
    for (let i = 0; i < 3; i++) {
      http.get('/api/foo').subscribe({ error: () => {} });
      httpMock.expectOne('/api/foo').error(new ProgressEvent('error'));
    }
    expect(connection.isOffline).toBe(true);
  });

  it('records reachability on a successful response, resetting the failure tally', () => {
    connection.recordNetworkFailure();
    connection.recordNetworkFailure();

    http.get('/api/foo').subscribe();
    httpMock.expectOne('/api/foo').flush({ ok: true });

    // The success reset the tally, so a single further failure must not trip.
    http.get('/api/bar').subscribe({ error: () => {} });
    httpMock.expectOne('/api/bar').error(new ProgressEvent('error'));
    expect(connection.isOffline).toBe(false);
  });

  it('short-circuits non-probe requests while offline without hitting the wire', () => {
    tripOffline();
    expect(connection.isOffline).toBe(true);

    let caught: HttpErrorResponse | undefined;
    http.get('/api/foo').subscribe({ error: (e: HttpErrorResponse) => (caught = e) });

    // The request must never reach the backend.
    httpMock.expectNone('/api/foo');
    expect(caught).toBeInstanceOf(HttpErrorResponse);
    expect(caught?.status).toBe(0);
    expect(toast.error).not.toHaveBeenCalled();
  });

  it('lets the connection probe through the offline breaker', () => {
    tripOffline();
    http
      .get('/healthz', { context: new HttpContext().set(CONNECTION_PROBE, true), responseType: 'text' })
      .subscribe();
    httpMock.expectOne('/healthz').flush('ok');
    expect(connection.isOffline).toBe(false);
  });
});
