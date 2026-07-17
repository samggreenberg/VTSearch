import { Observable, Subject, of, throwError } from 'rxjs';
import { adaptivePoll } from './adaptive-poll';

/**
 * Unit coverage for the {@link adaptivePoll} primitive that replaced the
 * `timer(0, n)` + `switchMap` pollers (issue #2572). The three behaviours that
 * distinguish it from the old pattern each get a test: no request overlap, an
 * adaptive fast→slow cadence, and pause-while-hidden.
 */
describe('adaptivePoll', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('polls immediately, then repeats at fastMs while the response keeps changing', async () => {
    let calls = 0;
    const sub = adaptivePoll(
      () => {
        calls += 1;
        return of(calls); // always changing → never eases off fast
      },
      { fastMs: 1000, slowMs: 10000 },
    ).subscribe();

    expect(calls).toBe(1); // fires synchronously on subscribe, no initial timer
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(2);
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(3);

    sub.unsubscribe();
  });

  it('never overlaps: no new poll starts while one is still in flight', async () => {
    const pending: Subject<number>[] = [];
    const sub = adaptivePoll(
      () => {
        const s = new Subject<number>();
        pending.push(s);
        return s; // stays open until we complete it
      },
      { fastMs: 1000, slowMs: 10000 },
    ).subscribe();

    expect(pending.length).toBe(1);
    // Far past several fast intervals — the old switchMap poll would have
    // cancelled and re-issued repeatedly here.
    await vi.advanceTimersByTimeAsync(5000);
    expect(pending.length).toBe(1);

    // Complete the in-flight request; the next poll is scheduled only now.
    pending[0].next(1);
    pending[0].complete();
    await vi.advanceTimersByTimeAsync(1000);
    expect(pending.length).toBe(2);

    sub.unsubscribe();
  });

  it('eases from fastMs to slowMs after rampAfter unchanged polls, and snaps back on a change', async () => {
    let value = 'a';
    let calls = 0;
    const sub = adaptivePoll(
      () => {
        calls += 1;
        return of(value);
      },
      { fastMs: 1000, slowMs: 10000, rampAfter: 3 },
    ).subscribe();

    expect(calls).toBe(1); // t=0,     idle 0
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(2); // t=1000,  idle 1 (unchanged)
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(3); // t=2000,  idle 2
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(4); // t=3000,  idle 3 → next delay is slowMs

    // Fast interval no longer fires a poll — it is on the slow heartbeat now.
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(4);

    // The next poll lands on the slow heartbeat; flip the value so it counts as
    // "fresh" and the cadence snaps back to fast.
    value = 'b';
    await vi.advanceTimersByTimeAsync(9000); // t=13000
    expect(calls).toBe(5);

    await vi.advanceTimersByTimeAsync(1000); // back on fast cadence
    expect(calls).toBe(6);

    sub.unsubscribe();
  });

  it('absorbs a poll error and keeps polling (no emission for the failed tick)', async () => {
    let mode: 'err' | 'ok' = 'err';
    const received: string[] = [];
    const sub = adaptivePoll<string>(
      () => (mode === 'err' ? throwError(() => new Error('boom')) : of('ok')),
      { fastMs: 1000, slowMs: 10000 },
    ).subscribe((v) => received.push(v));

    expect(received).toEqual([]); // first tick errored → nothing emitted
    mode = 'ok';
    await vi.advanceTimersByTimeAsync(1000);
    expect(received).toEqual(['ok']); // loop survived the error

    sub.unsubscribe();
  });

  it('pauses while the tab is hidden and resumes with an immediate poll', async () => {
    let hidden = false;
    // Shadow the prototype's `hidden` getter with an own property on the
    // instance; deleting it in the finally restores the prototype accessor.
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    try {
      let calls = 0;
      const sub = adaptivePoll(
        () => {
          calls += 1;
          return of(calls);
        },
        { fastMs: 1000, slowMs: 10000 },
      ).subscribe();

      expect(calls).toBe(1);

      // Hide the tab: polling suspends.
      hidden = true;
      document.dispatchEvent(new Event('visibilitychange'));
      await vi.advanceTimersByTimeAsync(5000);
      expect(calls).toBe(1);

      // Reveal the tab: an immediate poll fires.
      hidden = false;
      document.dispatchEvent(new Event('visibilitychange'));
      expect(calls).toBe(2);

      sub.unsubscribe();
    } finally {
      delete (document as { hidden?: boolean }).hidden;
    }
  });

  it('honours a custom signature so an always-changing field does not defeat the back-off', async () => {
    let seq = 0;
    let calls = 0;
    const sub = adaptivePoll<{ stable: string; seq: number }>(
      () => {
        calls += 1;
        return of({ stable: 'x', seq: (seq += 1) }); // seq changes every poll
      },
      { fastMs: 1000, slowMs: 10000, rampAfter: 2, signature: (v) => v.stable },
    ).subscribe();

    // Signature is only `stable`, so despite seq changing the cadence still
    // eases: idle reaches 2 after two unchanged polls, then goes slow.
    expect(calls).toBe(1); // idle 0
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(2); // idle 1
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(3); // idle 2 → next delay slow
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toBe(3); // fast interval no longer fires

    sub.unsubscribe();
    void seq;
  });
});
