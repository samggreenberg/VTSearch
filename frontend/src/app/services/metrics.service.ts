import { DestroyRef, Injectable, InjectionToken, inject } from '@angular/core';

import { ServerMetricsProvider } from './server-metrics-provider';

/** One attribute value. Metrics lines are flat, so scalars only. */
export type MetricsAttributeValue = string | number | boolean | null;

/** One usage event (#5475). */
export interface MetricsEvent {
  /** What happened: `view`, `datasetImported`, `resultsExported`, `settingChanged`, `click`, … */
  group: string;
  /** Flat details, e.g. `{ exporter: 'csv' }`. */
  attributes?: Record<string, MetricsAttributeValue>;
}

/**
 * Where metrics events go. The default, {@link ServerMetricsProvider}, batches
 * them to `POST /api/metrics`, which appends them to the server's metrics
 * file. A deployment that wants them elsewhere supplies its own provider in
 * `app.config.ts`:
 *
 * ```ts
 * { provide: METRICS_PROVIDER, useClass: MyCustomMetricsProvider },
 * ```
 *
 * A custom provider replaces the server sink; to keep both, inject
 * `ServerMetricsProvider` in it and forward each event there too.
 */
export interface MetricsProvider {
  send(event: MetricsEvent): void;
}

export const METRICS_PROVIDER = new InjectionToken<MetricsProvider>('METRICS_PROVIDER', {
  providedIn: 'root',
  factory: () => inject(ServerMetricsProvider),
});

/** The attribute a template puts on a control to count its clicks. */
const CLICK_ATTRIBUTE = 'data-metric';

/**
 * MetricsService: the one door usage events go through (#5475). Call
 * {@link logEvent} at the point a feature is used; the service hands the event
 * to the configured {@link METRICS_PROVIDER}.
 *
 * Clicks need no code at all: a control carrying `data-metric="<name>"` (on
 * itself or an ancestor) logs a `click` event with `target: <name>`. The
 * listener runs in the capture phase, so a handler that stops propagation
 * does not hide the click.
 *
 * Constructed in `AppComponent`, so the click listener is live from startup.
 */
@Injectable({ providedIn: 'root' })
export class MetricsService {
  private readonly provider = inject(METRICS_PROVIDER);

  constructor() {
    if (typeof document === 'undefined') return;
    const onClick = (event: Event) => this.logClick(event);
    document.addEventListener('click', onClick, { capture: true });
    inject(DestroyRef).onDestroy(() => document.removeEventListener('click', onClick, { capture: true }));
  }

  logEvent(event: MetricsEvent): void {
    this.provider.send({ group: event.group, attributes: { ...event.attributes } });
  }

  private logClick(event: Event): void {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const name = target.closest(`[${CLICK_ATTRIBUTE}]`)?.getAttribute(CLICK_ATTRIBUTE);
    if (name) this.logEvent({ group: 'click', attributes: { target: name } });
  }
}
