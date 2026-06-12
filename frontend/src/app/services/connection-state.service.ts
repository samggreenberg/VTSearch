import {
  HttpClient,
  HttpContext,
  HttpContextToken,
} from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

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
  /**
   * Consecutive network failures needed to declare the backend offline. A
   * threshold above 1 keeps a single transient blip (one dropped request
   * while the backend is otherwise healthy) from locking the whole app into
   * the offline state; with pollers ticking every 1.5–3s a genuinely-down
   * backend still trips the breaker within a few seconds.
   */
  private static readonly OFFLINE_THRESHOLD = 3;

  private readonly statusSubject = new BehaviorSubject<ConnectionStatus>('online');
  private readonly retryingSubject = new BehaviorSubject<boolean>(false);

  /** Current connectivity, `online` until proven otherwise. */
  readonly status$ = this.statusSubject.asObservable();
  /** True while a Retry probe is in flight (drives the button's busy state). */
  readonly retrying$ = this.retryingSubject.asObservable();

  private consecutiveFailures = 0;

  constructor(private http: HttpClient) {
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
    return this.statusSubject.value === 'offline';
  }

  get status(): ConnectionStatus {
    return this.statusSubject.value;
  }

  /**
   * Record that a request reached the server (any HTTP response, including
   * an error status — a 404/500 still proves connectivity). Resets the
   * failure tally and clears the offline state.
   */
  recordSuccess(): void {
    this.consecutiveFailures = 0;
    if (this.statusSubject.value !== 'online') {
      this.statusSubject.next('online');
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

  private goOffline(): void {
    if (this.statusSubject.value !== 'offline') {
      this.statusSubject.next('offline');
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
    if (this.retryingSubject.value) return;
    this.retryingSubject.next(true);
    this.http
      .get('/healthz', {
        context: new HttpContext().set(CONNECTION_PROBE, true),
        responseType: 'text',
      })
      .subscribe({
        next: () => {
          this.recordSuccess();
          this.retryingSubject.next(false);
        },
        // A network error leaves us offline (the interceptor already counted
        // it); any HTTP error status means the server answered, so the
        // interceptor will have flipped us back online. Either way the probe
        // is done.
        error: () => this.retryingSubject.next(false),
      });
  }
}
