import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { METRICS_FLUSH_AT, METRICS_FLUSH_DELAY_MS, ServerMetricsProvider } from './server-metrics-provider';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { provideHttpTesting } from '../testing/test-providers';
import type { MetricsBatch } from '../generated/api-client/models/metrics-batch';

describe('ServerMetricsProvider', () => {
  let provider: ServerMetricsProvider;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    vi.useFakeTimers();
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    provider = TestBed.inject(ServerMetricsProvider);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('batches events and posts them once the delay passes', () => {
    provider.send({ group: 'view', attributes: { view: 'browse' } });
    provider.send({ group: 'click', attributes: { target: 'imageZoomIn' } });
    httpMock.expectNone('/api/metrics');

    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    const req = httpMock.expectOne('/api/metrics');
    const body = req.request.body as MetricsBatch;
    expect(req.request.method).toBe('POST');
    expect(body.events.map((e) => e.group)).toEqual(['view', 'click']);
    expect(body.events[0].attributes).toEqual({ view: 'browse' });
    expect(typeof body.events[0].client_ts).toBe('number');
    expect(typeof body.session).toBe('string');
    req.flush({ recorded: 2, enabled: true });
  });

  it('opts out of the global error toast and swallows a failure', () => {
    provider.send({ group: 'view' });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    const req = httpMock.expectOne('/api/metrics');
    expect(req.request.context.get(SKIP_ERROR_TOAST)).toBe(true);
    expect(() => req.flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' })).not.toThrow();
  });

  it(`sends at once when the batch reaches ${METRICS_FLUSH_AT} events`, () => {
    for (let i = 0; i < METRICS_FLUSH_AT; i++) provider.send({ group: 'click' });
    const req = httpMock.expectOne('/api/metrics');
    expect((req.request.body as MetricsBatch).events.length).toBe(METRICS_FLUSH_AT);
    req.flush({ recorded: METRICS_FLUSH_AT, enabled: true });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    httpMock.expectNone('/api/metrics');
  });

  it('keeps one session id across batches', () => {
    provider.send({ group: 'a' });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    const first = httpMock.expectOne('/api/metrics');
    first.flush({ recorded: 1, enabled: true });
    provider.send({ group: 'b' });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    const second = httpMock.expectOne('/api/metrics');
    second.flush({ recorded: 1, enabled: true });
    expect((second.request.body as MetricsBatch).session).toBe((first.request.body as MetricsBatch).session);
  });

  it('stops sending once the server reports metrics disabled', () => {
    provider.send({ group: 'view' });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    httpMock.expectOne('/api/metrics').flush({ recorded: 0, enabled: false });

    provider.send({ group: 'view' });
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    httpMock.expectNone('/api/metrics');
  });

  it('sends nothing when the buffer is empty', () => {
    provider.flush();
    httpMock.expectNone('/api/metrics');
  });

  it('hands the buffer to a keepalive fetch when the page goes away', () => {
    const fetchSpy = vi.fn(() => Promise.resolve(new Response('{}')));
    vi.stubGlobal('fetch', fetchSpy);
    provider.send({ group: 'resultsExported', attributes: { exporter: 'csv' } });

    window.dispatchEvent(new Event('pagehide'));

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/metrics');
    expect(init.keepalive).toBe(true);
    expect(JSON.parse(init.body as string).events[0].attributes).toEqual({ exporter: 'csv' });
    // The buffer went with the fetch, so the timer has nothing left to send.
    vi.advanceTimersByTime(METRICS_FLUSH_DELAY_MS);
    httpMock.expectNone('/api/metrics');
  });
});
