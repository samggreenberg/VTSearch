import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { Subscription } from 'rxjs';
import { RunningJobsService, pairKey } from './running-jobs.service';

describe('pairKey', () => {
  it('joins dataset and detector ids with a stable separator', () => {
    expect(pairKey('ds', 'det')).toBe('ds::det');
  });
});

describe('RunningJobsService', () => {
  let service: RunningJobsService;
  let httpMock: HttpTestingController;
  const subs: Subscription[] = [];

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(RunningJobsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    subs.forEach((s) => s.unsubscribe());
    subs.length = 0;
    httpMock.verify();
  });

  it('isBusy short-circuits to false for a half pair without polling', () => {
    let value: boolean | undefined;
    subs.push(service.isBusy('', 'det').subscribe((v) => (value = v)));
    expect(value).toBe(false);
    httpMock.expectNone('/api/jobs/active');
  });

  it('polls /api/jobs/active on first subscription and exposes busy pairs', async () => {
    vi.useFakeTimers();
    try {
      let latest = new Map<string, string[]>();
      subs.push(service.busyPairs$.subscribe((m) => (latest = m)));

      await vi.advanceTimersByTimeAsync(0);
      const req = httpMock.expectOne('/api/jobs/active');
      expect(req.request.method).toBe('GET');
      req.flush({ busy_pairs: [{ dataset_id: 'd', detector_id: 'x', job_types: ['eval'] }] });

      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('isBusy reflects whether the pair has a running job', async () => {
    vi.useFakeTimers();
    try {
      const seen: boolean[] = [];
      subs.push(service.isBusy('d', 'x').subscribe((v) => seen.push(v)));

      await vi.advanceTimersByTimeAsync(0);
      httpMock
        .expectOne('/api/jobs/active')
        .flush({ busy_pairs: [{ dataset_id: 'd', detector_id: 'x', job_types: ['eval'] }] });

      expect(seen[seen.length - 1]).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('re-polls on each interval tick', async () => {
    vi.useFakeTimers();
    try {
      subs.push(service.busyPairs$.subscribe());

      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/jobs/active').flush({ busy_pairs: [] });

      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectOne('/api/jobs/active').flush({ busy_pairs: [] });
    } finally {
      vi.useRealTimers();
    }
  });

  it('stops polling and clears state once the last observer unsubscribes', async () => {
    vi.useFakeTimers();
    try {
      let latest = new Map<string, string[]>();
      const sub = service.busyPairs$.subscribe((m) => (latest = m));

      await vi.advanceTimersByTimeAsync(0);
      httpMock
        .expectOne('/api/jobs/active')
        .flush({ busy_pairs: [{ dataset_id: 'd', detector_id: 'x', job_types: ['eval'] }] });
      expect(latest.size).toBe(1);

      sub.unsubscribe();
      // A later resubscriber must not see stale data before its first poll.
      let resubscribed = new Map<string, string[]>();
      subs.push(service.busyPairs$.subscribe((m) => (resubscribed = m)));
      expect(resubscribed.size).toBe(0);

      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/jobs/active').flush({ busy_pairs: [] });
    } finally {
      vi.useRealTimers();
    }
  });

  it('survives a transient poll failure and keeps polling', async () => {
    vi.useFakeTimers();
    try {
      let latest = new Map<string, string[]>();
      subs.push(service.busyPairs$.subscribe((m) => (latest = m)));

      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/jobs/active').flush('boom', { status: 500, statusText: 'Server Error' });
      // The error path emits an empty payload (clears stale spinners).
      expect(latest.size).toBe(0);

      // The pipeline is intact: the next tick still polls.
      await vi.advanceTimersByTimeAsync(5000);
      httpMock
        .expectOne('/api/jobs/active')
        .flush({ busy_pairs: [{ dataset_id: 'd', detector_id: 'x', job_types: [] }] });
      expect(latest.size).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
