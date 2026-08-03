import {
  HttpClient,
  HttpContext,
  HttpContextToken,
} from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';

/**
 * Marks a request as the connection-recovery probe so the
 * {@link errorInterceptor} lets it through the offline circuit breaker.
 *
 * Every other request is short-circuited while offline (returned an
 * immediate synthetic network error without touching the wire), which is
 * what stops the browser console from filling with `net::ERR_CONNECTION_REFUSED`
 * lines — those are emitted by the browser for every real failed fetch and
 * cannot be silenced from JS; the only cure is to not make the request. The
 * probe is the one exception: it is the single request allowed to test the
 * wire so a manual Retry can discover the backend is back.
 */
export const CONNECTION_PROBE = new HttpContextToken<boolean>(() => false);

export type ConnectionStatus = 'online' | 'offline';

/**
 * Front-end circuit breaker for backend connectivity.
 *
 * The app fires a handful of background pollers (votes, labelset, active
 * jobs, disk/RAM) plus a persistent SSE stream. When the backend goes away
 * (e.g. `Ctrl-C` on `app.py`) every one of those keeps retrying on its own
 * timer, flooding the console with connection-refused errors and hammering
 * the port on restart. This service watches the central
 * {@link errorInterceptor} for network failures (HTTP status 0), and once a
 * small number pile up it flips to `offline`. While offline:
 *
 *  - the interceptor suppresses every request (no wire traffic, no console
 *    flood), and
 *  - {@link ProgressEventsService} closes the SSE EventSource so its native
 *    auto-reconnect stops trying.
 *
 * Recovery is **manual**: nothing probes the backend until the user clicks
 * Retry (or the browser fires an `online` event after an `offline` one is
 * intentionally NOT treated as recovery — see the constructor). {@link retry}
 * sends a single `/healthz` probe; any HTTP response at all (even an error
 * status) proves the server is reachable and flips back to `online`, which
 * resumes the pollers and reconnects the stream.
 *
 * Per the repo's "no persisted vectors/state" spirit this is purely
 * in-memory, process-scoped UI state: it exists so the frontend degrades
 * gracefully while the backend is down, which is exactly the case where
 * nothing can be persisted anyway.
 */
@Injectable({ providedIn: 'root' })
export class ConnectionStateService {
  private http = inject(HttpClient);

  /**
   * Consecutive network failures needed to declare the backend offline. A
   * threshold above 1 keeps a single transient blip (one dropped request
   * while the backend is otherwise healthy) from locking the whole app into
   * the offline state; with pollers ticking every 1.5–3s a genuinely-down
   * backend still trips the breaker within a few seconds.
   */
  private static readonly OFFLINE_THRESHOLD = 3;

  private readonly _status = signal<ConnectionStatus>('online');
  private readonly _retrying = signal(false);

  /**
   * Current connectivity, `online` until proven otherwise. A signal so a write
   * from a raw `offline`-event listener (or the interceptor) schedules change
   * detection on bound templates under zoneless without an `NgZone.run`.
   * Read-only to callers; only this service flips it.
   */
  readonly status = this._status.asReadonly();
  /** True while a Retry probe is in flight (drives the button's busy state). */
  readonly retrying = this._retrying.asReadonly();

  private consecutiveFailures = 0;
  /** True while a stream-failure classification probe is in flight. */
  private streamProbeInFlight = false;

  constructor() {
    // A browser-level `offline` event (the OS lost its network interface) is
    // an unambiguous, immediate signal — trip the breaker at once rather
    // than waiting for the failure threshold. The matching `online` event is
    // deliberately ignored: recovery is manual (the user's Retry click), and
    // the network returning does not guarantee the backend is back.
    if (typeof window !== 'undefined') {
      window.addEventListener('offline', () => this.goOffline());
    }
  }

  get isOffline(): boolean {
    return this._status() === 'offline';
  }

  /**
   * Record that a request reached the server (any HTTP response, including
   * an error status — a 404/500 still proves connectivity). Resets the
   * failure tally and clears the offline state.
   */
  recordSuccess(): void {
    this.consecutiveFailures = 0;
    if (this._status() !== 'online') {
      this._status.set('online');
    }
  }

  /**
   * Record a network-level failure (HTTP status 0: the request never got a
   * response). Trips the breaker once enough pile up consecutively.
   */
  recordNetworkFailure(): void {
    this.consecutiveFailures += 1;
    if (this.consecutiveFailures >= ConnectionStateService.OFFLINE_THRESHOLD) {
      this.goOffline();
    }
  }

  /**
   * Record an error reported by the SSE stream ({@link ProgressEventsService}).
   *
   * `EventSource` hides HTTP status codes, so a stream error is ambiguous:
   * the backend may be unreachable — or it answered the connect with an error
   * such as the `/api/events` 503 slot-cap rejection, which *proves* it is
   * alive. Counting the ambiguous error as a network failure used to lock the
   * whole app offline when three cap rejections landed inside the slot-release
   * window (#2816). Instead, send a single `/healthz` probe and let the
   * interceptor classify the outcome: any HTTP response resets the failure
   * tally (so the stream keeps retrying on its 2s timer until a slot frees),
   * while a status-0 failure counts toward tripping the breaker exactly like
   * any other request's.
   */
  recordStreamFailure(): void {
    if (this.isOffline || this.streamProbeInFlight) return;
    this.streamProbeInFlight = true;
    this.http
      .get('/healthz', {
        context: new HttpContext().set(CONNECTION_PROBE, true),
        responseType: 'text',
      })
      .subscribe({
        // The interceptor already recorded the outcome (success or network
        // failure); this subscription only tracks that the probe finished.
        next: () => (this.streamProbeInFlight = false),
        error: () => (this.streamProbeInFlight = false),
      });
  }

  private goOffline(): void {
    if (this._status() !== 'offline') {
      this._status.set('offline');
    }
  }

  /**
   * Manually probe the backend. Sends a single `/healthz` request that
   * bypasses the offline circuit breaker; the interceptor records the
   * outcome (success → back online, network error → still offline), so this
   * only needs to manage the in-flight flag. No-ops if a probe is already
   * running.
   */
  retry(): void {
    if (this._retrying()) return;
    this._retrying.set(true);
    this.http
      .get('/healthz', {
        context: new HttpContext().set(CONNECTION_PROBE, true),
        responseType: 'text',
      })
      .subscribe({
        next: () => {
          this.recordSuccess();
          this._retrying.set(false);
        },
        // A network error leaves us offline (the interceptor already counted
        // it); any HTTP error status means the server answered, so the
        // interceptor will have flipped us back online. Either way the probe
        // is done.
        error: () => this._retrying.set(false),
      });
  }
}
