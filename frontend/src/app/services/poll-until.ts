import { Signal, signal } from '@angular/core';
import { Observable, Subscription } from 'rxjs';
import { take } from 'rxjs/operators';

/** What a caller's {@link PollUntilConfig.apply} decided about the response it
 *  was just handed: keep polling, or settle (the caller has already applied
 *  whatever terminal state — ready, errored — it wants). */
export type PollStep = 'continue' | 'stop';

/** Delay between successive polls, and before the first one. */
const INTERVAL_MS = 1000;
/** Consecutive request failures tolerated before giving up on the job. */
const MAX_ERRORS = 5;

export interface PollUntilConfig<T> {
  /** Issue one poll. Subscribed with `take(1)`; a new one is only issued once
   *  this one has settled, so polls never overlap. */
  fetch: () => Observable<T>;
  /**
   * Apply a successful response. Return `'continue'` to schedule another poll,
   * or `'stop'` once the caller has reached a terminal state of its own (the
   * job finished, or the response itself reported an error). The loop is
   * already torn down by the time `'stop'` takes effect — there is nothing
   * further to cancel.
   */
  apply: (value: T) => PollStep;
  /** Called once, after five *consecutive* request failures, with the loop
   *  already stopped. A single failure is absorbed and retried. */
  onLostContact: () => void;
}

export interface PollHandle {
  /** True while the loop is scheduled or a request is in flight; flips to
   *  false the moment it settles (done, failed, or {@link stop}ped). Backed by
   *  a signal, so a caller can drive a template off it under zoneless. */
  readonly active: Signal<boolean>;
  /** Abandon the loop: cancel any pending timer and in-flight request. Safe to
   *  call repeatedly, and after the loop has already settled. */
  stop(): void;
}

/**
 * A one-shot "poll until this job finishes" loop: the terminal-state sibling of
 * {@link adaptivePoll}, which is for open-ended background polling.
 *
 * The two are deliberately separate helpers rather than one configurable one,
 * because their defining behaviours are opposites. `adaptivePoll` never
 * terminates, eases its cadence toward a heartbeat as the response goes stale,
 * and **suspends while the tab is hidden**. All three are wrong for a job the
 * user is waiting on: it *does* terminate, its progress is never stale while it
 * runs, and pausing it in a background tab would strand the user behind a bar
 * that stopped moving — or, in the prep services, silently defer the navigation
 * they are waiting for.
 *
 * What it shares with `adaptivePoll` is the property that motivated it
 * (issue #2572): each request runs to completion before the next is scheduled,
 * so a backend slower than the interval degrades to its own response time
 * instead of having every request cancelled by the following tick.
 *
 * **Error policy.** A failed request does not stop the loop; it increments a
 * consecutive-failure count (reset by any success) and retries with exponential
 * backoff — 2s, 4s, 8s, … capped at 30s — so a struggling backend is given room
 * rather than hammered at the poll interval. Only five failures in a row
 * give up, via {@link PollUntilConfig.onLostContact}. The cadence and the
 * failure budget are fixed rather than configurable: all three call sites had
 * independently settled on the same 1s / 5-failure numbers, and a knob nobody
 * turns is a place for them to drift apart again.
 */
export function pollUntil<T>(config: PollUntilConfig<T>): PollHandle {
  const active = signal(true);
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let inFlight: Subscription | null = null;
  let errors = 0;

  const stop = (): void => {
    stopped = true;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (inFlight !== null) {
      inFlight.unsubscribe();
      inFlight = null;
    }
    active.set(false);
  };

  // Every scheduling point funnels through here, so a caller that tore the loop
  // down from inside `apply` (the prep services stop themselves as part of
  // navigating away) can never leave a zombie timer behind, whatever verdict
  // that `apply` went on to return.
  const schedule = (delay: number): void => {
    if (stopped) return;
    timer = setTimeout(tick, delay);
  };

  function tick(): void {
    timer = null;
    // A synchronous observable (tests, a cached response) settles *inside*
    // `subscribe`, before it returns — so record whether that happened and skip
    // storing a subscription that is already spent, which would otherwise
    // survive as a stale handle and let `stop()` unsubscribe the wrong tick.
    let settled = false;
    const sub = config
      .fetch()
      .pipe(take(1))
      .subscribe({
        next: (value) => {
          settled = true;
          inFlight = null;
          errors = 0;
          if (config.apply(value) === 'continue') schedule(INTERVAL_MS);
          else stop();
        },
        error: () => {
          settled = true;
          inFlight = null;
          errors += 1;
          if (errors >= MAX_ERRORS) {
            stop();
            config.onLostContact();
            return;
          }
          schedule(retryDelayMs(errors));
        },
      });
    if (!settled) inFlight = sub;
  }

  schedule(INTERVAL_MS);
  return { active: active.asReadonly(), stop };
}

/** Exponential backoff after `n` consecutive failures: 2s, 4s, 8s, … capped at
 *  30s. Deliberately slower than the poll interval — a run of failures means
 *  the backend is in trouble, which is the worst moment to keep the original
 *  cadence. */
function retryDelayMs(n: number): number {
  return Math.min(2000 * 2 ** (n - 1), 30000);
}
