import { DestroyRef, Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import { recordMetrics } from '../generated/api-client/fn/metrics/record-metrics';
import type { MetricsBatch } from '../generated/api-client/models/metrics-batch';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import type { MetricsEvent, MetricsProvider } from './metrics.service';

/** How long the first buffered event waits for company before the batch goes. */
export const METRICS_FLUSH_DELAY_MS = 10_000;

/** A batch this large goes at once rather than waiting out the delay. */
export const METRICS_FLUSH_AT = 50;

/**
 * An id for this page load, so the server can group a visit's events. It only
 * has to tell visits apart (`crypto.randomUUID` would need a secure context,
 * which an intranet deployment on plain http is not).
 */
function newSessionId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

/**
 * The default {@link MetricsProvider} (#5475): batches events to
 * `POST /api/metrics`, which appends them to the server's metrics file.
 *
 * Best-effort throughout. A batch goes {@link METRICS_FLUSH_DELAY_MS} after its
 * first event, or at once when it reaches {@link METRICS_FLUSH_AT}; one that
 * fails is dropped, never toasted or retried. When the page goes away the
 * buffer leaves on a `keepalive` fetch, which outlives the page where an
 * `HttpClient` request would be cancelled with it. Once the server answers
 * `enabled: false` (its `enable_metrics` setting is off) the provider stops
 * sending for the rest of the visit.
 */
@Injectable({ providedIn: 'root' })
export class ServerMetricsProvider implements MetricsProvider {
  private readonly http = inject(HttpClient);
  private readonly config = inject(ApiConfiguration);
  private readonly session = newSessionId();
  private buffer: MetricsBatch['events'] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private disabled = false;

  constructor() {
    const destroyRef = inject(DestroyRef);
    destroyRef.onDestroy(() => this.clearTimer());
    if (typeof window === 'undefined') return;
    const onPageHide = () => this.flushOnPageHide();
    window.addEventListener('pagehide', onPageHide);
    destroyRef.onDestroy(() => window.removeEventListener('pagehide', onPageHide));
  }

  send(event: MetricsEvent): void {
    if (this.disabled) return;
    this.buffer.push({ group: event.group, attributes: event.attributes ?? {}, client_ts: Date.now() });
    if (this.buffer.length >= METRICS_FLUSH_AT) {
      this.flush();
    } else {
      this.timer ??= setTimeout(() => this.flush(), METRICS_FLUSH_DELAY_MS);
    }
  }

  /** Send everything buffered now. */
  flush(): void {
    const body = this.takeBatch();
    if (!body) return;
    recordMetrics(this.http, this.config.rootUrl, { body }, new HttpContext().set(SKIP_ERROR_TOAST, true)).subscribe({
      next: (r) => {
        if (!r.body.enabled) this.disable();
      },
      error: () => {},
    });
  }

  private flushOnPageHide(): void {
    const body = this.takeBatch();
    if (!body || typeof fetch !== 'function') return;
    fetch(`${this.config.rootUrl}${recordMetrics.PATH}`, {
      method: 'POST',
      keepalive: true,
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).catch(() => {});
  }

  /** Empty the buffer into a request body, or `null` when there is nothing to send. */
  private takeBatch(): MetricsBatch | null {
    this.clearTimer();
    if (this.buffer.length === 0) return null;
    const events = this.buffer;
    this.buffer = [];
    return { session: this.session, events };
  }

  private disable(): void {
    this.disabled = true;
    this.buffer = [];
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }
}
