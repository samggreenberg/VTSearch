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

  it('ignores interval ticks while a poll is still in flight (exhaustMap)', async () => {
    vi.useFakeTimers();
    try {
      subs.push(service.busyPairs$.subscribe());

      await vi.advanceTimersByTimeAsync(0);
      const inFlight = httpMock.expectOne('/api/jobs/active');

      // A tick fires while the first request is still pending: exhaustMap must
      // NOT issue a second request (at most one poll in flight at a time).
      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectNone('/api/jobs/active');

      inFlight.flush({ busy_pairs: [] });
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

  describe('poll failures', () => {
    const BUSY = { busy_pairs: [{ dataset_id: 'd', detector_id: 'x', job_types: ['eval'] }] };
    const failPoll = () =>
      httpMock.expectOne('/api/jobs/active').flush('boom', { status: 500, statusText: 'Server Error' });

    let latest: Map<string, string[]>;

    // Subscribe and land one successful poll reporting a busy pair, so each
    // test starts from a spinner that a failure could wrongly clear.
    async function startBusy(): Promise<void> {
      latest = new Map();
      subs.push(service.busyPairs$.subscribe((m) => (latest = m)));
      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/jobs/active').flush(BUSY);
      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);
    }

    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('holds the last known map through a single failed poll and keeps polling', async () => {
      await startBusy();

      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      // One failed poll is no evidence the job finished: the spinner stays.
      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);

      // The pipeline is intact: the next tick still polls, and its answer wins.
      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectOne('/api/jobs/active').flush({ busy_pairs: [] });
      expect(latest.size).toBe(0);
    });

    it('holds the last known map through a single timed-out poll and keeps polling', async () => {
      await startBusy();

      await vi.advanceTimersByTimeAsync(5000);
      const hung = httpMock.expectOne('/api/jobs/active');
      // Run past the 10 s request timeout.
      await vi.advanceTimersByTimeAsync(10000);
      expect(hung.cancelled).toBe(true);
      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);

      // Polling resumes. Whether the tick that coincides with the timeout
      // fires before or after it, exactly one fresh request is open a tick later.
      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectOne('/api/jobs/active').flush({ busy_pairs: [] });
      expect(latest.size).toBe(0);
    });

    it('clears the map on the third consecutive failure and keeps polling', async () => {
      await startBusy();

      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      expect(latest.size).toBe(1);

      // Three in a row: the backend is unreachable, so stale spinners would mislead.
      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      expect(latest.size).toBe(0);

      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectOne('/api/jobs/active').flush(BUSY);
      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);
    });

    it('restarts the failure count after a successful poll', async () => {
      await startBusy();

      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      await vi.advanceTimersByTimeAsync(5000);
      httpMock.expectOne('/api/jobs/active').flush(BUSY);

      // Two more failures are only two in a row, not four: the map holds.
      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      expect(latest.get(pairKey('d', 'x'))).toEqual(['eval']);

      await vi.advanceTimersByTimeAsync(5000);
      failPoll();
      expect(latest.size).toBe(0);
    });
  });
});
