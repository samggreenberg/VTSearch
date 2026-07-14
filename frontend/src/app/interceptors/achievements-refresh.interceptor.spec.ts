import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { HttpClient } from '@angular/common/http';

import { achievementsRefreshInterceptor } from './achievements-refresh.interceptor';
import { AchievementsService } from '../services/achievements.service';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The achievements-refresh interceptor nudges AchievementsService after any
 * action endpoint that could unlock a tier. Pin the gate: POST + a watched
 * path + a 2xx response, and nothing else.
 */
describe('achievementsRefreshInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let achievements: { refresh: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    achievements = { refresh: vi.fn() };
    TestBed.configureTestingModule({
      providers: [
        ...provideHttpTesting(achievementsRefreshInterceptor),
        { provide: AchievementsService, useValue: achievements },
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  const watched = [
    '/api/medias/xyz/vote',
    '/api/detectors/mydet/labels/l1/vote',
    '/api/find-label',
    '/api/auto-detect',
    '/api/label-importers/import/server_json_file',
    '/api/datasets/registry/reg-1/load',
  ];

  for (const url of watched) {
    it(`refreshes after a successful POST to ${url}`, () => {
      http.post(url, {}).subscribe();
      httpMock.expectOne(url).flush({ ok: true });
      expect(achievements.refresh).toHaveBeenCalledTimes(1);
    });
  }

  it('does not refresh when a watched POST fails with a non-2xx status', () => {
    http.post('/api/find-label', {}).subscribe({ error: () => {} });
    httpMock
      .expectOne('/api/find-label')
      .flush({ error: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(achievements.refresh).not.toHaveBeenCalled();
  });

  it('ignores non-POST methods on a watched path', () => {
    http.get('/api/medias/xyz/vote').subscribe();
    httpMock.expectOne('/api/medias/xyz/vote').flush({});
    expect(achievements.refresh).not.toHaveBeenCalled();
  });

  it('ignores POSTs to unwatched endpoints', () => {
    http.post('/api/datasets', {}).subscribe();
    httpMock.expectOne('/api/datasets').flush({});
    expect(achievements.refresh).not.toHaveBeenCalled();
  });

  it('matches a watched path even with a query string', () => {
    http.post('/api/auto-detect?dry=1', {}).subscribe();
    httpMock.expectOne('/api/auto-detect?dry=1').flush({});
    expect(achievements.refresh).toHaveBeenCalledTimes(1);
  });

  it('does not match a superstring of a watched path', () => {
    // The anchored regex must not fire for a longer, unrelated path.
    http.post('/api/find-label/extra', {}).subscribe();
    httpMock.expectOne('/api/find-label/extra').flush({});
    expect(achievements.refresh).not.toHaveBeenCalled();
  });
});
