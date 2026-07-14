import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { HttpClient } from '@angular/common/http';

import { timezoneInterceptor } from './timezone.interceptor';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The timezone interceptor carries the browser's local UTC offset to the
 * backend so wall-clock achievement buckets reflect the user's clock. Pin
 * that it stamps the offset on every request and leaves existing headers
 * intact.
 */
describe('timezoneInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        ...provideHttpTesting(timezoneInterceptor),
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    vi.restoreAllMocks();
  });

  it('stamps X-Timezone-Offset with the current browser offset', () => {
    const expected = String(new Date().getTimezoneOffset());
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Timezone-Offset')).toBe(expected);
    req.flush({});
  });

  it('reflects the actual getTimezoneOffset value', () => {
    vi.spyOn(Date.prototype, 'getTimezoneOffset').mockReturnValue(-330);
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Timezone-Offset')).toBe('-330');
    req.flush({});
  });

  it('preserves headers already on the request', () => {
    http.get('/api/foo', { headers: { 'X-Custom': 'keep-me' } }).subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Custom')).toBe('keep-me');
    expect(req.request.headers.has('X-Timezone-Offset')).toBe(true);
    req.flush({});
  });
});
