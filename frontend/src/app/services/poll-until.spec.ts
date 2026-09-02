import { Subject, of, throwError } from 'rxjs';
import { pollUntil } from './poll-until';

/**
 * Unit coverage for the {@link pollUntil} primitive that replaced the three
 * hand-rolled projection-build poll loops (issue #3446). Each test pins one of
 * the behaviours the copies had disagreed on — when the loop stops, how a
 * failed request is retried, and how many consecutive failures give up — plus
 * the non-overlap property it inherits from the pattern in `adaptivePoll`.
 */
describe('pollUntil', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('waits one interval before the first poll, then repeats while apply says continue', async () => {
    let calls = 0;
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        return of(calls);
      },
      apply: () => 'continue' as const,
      onLostContact: () => undefined,
    });

    // Unlike adaptivePoll, the caller has already fetched once itself; the loop
    // exists to watch what happens *next*.
    expect(calls).toBe(0);
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(2);

    handle.stop();
  });

  it("stops, and reports inactive, as soon as apply returns 'stop'", async () => {
    let calls = 0;
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        return of(calls);
      },
      apply: (n) => (n >= 2 ? ('stop' as const) : ('continue' as const)),
      onLostContact: () => undefined,
    });

    expect(handle.active()).toBe(true);
    await vi.advanceTimersByTimeAsync(2000);
    expect(calls).toBe(2);
    expect(handle.active()).toBe(false);

    // No further polls after it settled.
    await vi.advanceTimersByTimeAsync(10000);
    expect(calls).toBe(2);
  });

  it('never overlaps: no new poll starts while one is still in flight', async () => {
    const pending: Subject<number>[] = [];
    const handle = pollUntil({
      fetch: () => {
        const s = new Subject<number>();
        pending.push(s);
        return s;
      },
      apply: () => 'continue' as const,
      onLostContact: () => undefined,
    });

    await vi.advanceTimersByTimeAsync(1000);
    expect(pending.length).toBe(1);
    // Far past several intervals: a backend slower than the poll interval must
    // degrade to its own response time, not have every request superseded.
    await vi.advanceTimersByTimeAsync(5000);
    expect(pending.length).toBe(1);

    pending[0].next(1);
    pending[0].complete();
    await vi.advanceTimersByTimeAsync(1000);
    expect(pending.length).toBe(2);

    handle.stop();
  });

  it('absorbs a single failure and retries with backoff rather than at the poll interval', async () => {
    let calls = 0;
    let failNext = true;
    const applied: number[] = [];
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        if (failNext) return throwError(() => new Error('boom'));
        return of(calls);
      },
      apply: (n) => {
        applied.push(n);
        return 'continue' as const;
      },
      onLostContact: () => undefined,
    });

    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(1);
    expect(applied).toEqual([]);

    // The retry is the 2s backoff step, not another 1s poll interval — a
    // struggling backend gets room instead of the original cadence.
    failNext = false;
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(2);
    expect(applied).toEqual([2]);
    expect(handle.active()).toBe(true);

    handle.stop();
  });

  it('gives up after five consecutive failures, and only then', async () => {
    let calls = 0;
    let lostContact = 0;
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        return throwError(() => new Error('boom'));
      },
      apply: () => 'continue' as const,
      onLostContact: () => {
        lostContact += 1;
      },
    });

    // 1s to the first poll, then backoff 2s, 4s, 8s between the retries.
    await vi.advanceTimersByTimeAsync(1000 + 2000 + 4000 + 8000);
    expect(calls).toBe(4);
    expect(lostContact).toBe(0);
    expect(handle.active()).toBe(true);

    await vi.advanceTimersByTimeAsync(16000);
    expect(calls).toBe(5);
    expect(lostContact).toBe(1);
    expect(handle.active()).toBe(false);

    // Terminal: no sixth attempt, and onLostContact fires exactly once.
    await vi.advanceTimersByTimeAsync(60000);
    expect(calls).toBe(5);
    expect(lostContact).toBe(1);
  });

  it('resets the failure count on any success, so intermittent errors never give up', async () => {
    let calls = 0;
    let lostContact = 0;
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        // Four failures, then a success, on repeat — never five in a row.
        return calls % 5 === 0 ? of(calls) : throwError(() => new Error('boom'));
      },
      apply: () => 'continue' as const,
      onLostContact: () => {
        lostContact += 1;
      },
    });

    await vi.advanceTimersByTimeAsync(120000);
    expect(calls).toBeGreaterThan(5);
    expect(lostContact).toBe(0);

    handle.stop();
  });

  it('stop() cancels a pending timer and is safe to call repeatedly', async () => {
    let calls = 0;
    const handle = pollUntil({
      fetch: () => {
        calls += 1;
        return of(calls);
      },
      apply: () => 'continue' as const,
      onLostContact: () => undefined,
    });

    handle.stop();
    handle.stop();
    await vi.advanceTimersByTimeAsync(10000);
    expect(calls).toBe(0);
    expect(handle.active()).toBe(false);
  });

  it('leaves no zombie timer when apply stops the loop and still returns continue', async () => {
    // The prep services tear themselves down from inside `apply` (finishing
    // navigates away). Guard against the loop out-living that teardown because
    // the verdict it went on to return said otherwise.
    let calls = 0;
    let handle: { stop: () => void } | null = null;
    handle = pollUntil({
      fetch: () => {
        calls += 1;
        return of(calls);
      },
      apply: () => {
        handle?.stop();
        return 'continue' as const;
      },
      onLostContact: () => undefined,
    });

    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(1);
    await vi.advanceTimersByTimeAsync(10000);
    expect(calls).toBe(1);
  });
});
