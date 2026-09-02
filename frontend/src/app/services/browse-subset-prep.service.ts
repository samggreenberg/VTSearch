import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { take } from 'rxjs/operators';
import { BrowseSubsetService } from './browse-subset.service';
import { ProjectionApiService } from './projection-api.service';
import { pollUntil, type PollHandle, type PollStep } from './poll-until';
import { ToastService } from './toast.service';
import { formatEta, progressBarState } from '../utils/format-progress';
import type { ProgressEvent } from '../models/api.models';
import type { ProjectionMeta } from '../models/projection.models';

/**
 * Orchestrates the Find view's **Browse** buttons: build the ephemeral subset
 * projection (a UMAP fit over just this run's positives) *while the user is
 * still in Find*, showing progress there, and navigate to the browse view only
 * once the map is ready.
 *
 * This exists because a Find-positives browse can take minutes to fit, and the
 * browse view has nothing to render until it lands — navigating first stranded
 * the user on an empty canvas watching a status line, which is not how any
 * other wait in the app behaves. Every long operation in Find (detector
 * scoring, dataset loads) keeps the user where they are behind a progress bar,
 * and this makes Browse do the same. It is the Find-side sibling of
 * {@link BrowsePrepService}, which does the equivalent job for the dashboard's
 * full-dataset Browse button.
 *
 * Progress rides ``GET /api/projection/meta?subset=1``, which reports the
 * build's coarse phase (arranging → tiling → naming regions) plus a whole-job
 * ``overall`` fraction, so the bar fills once across the build rather than
 * restarting per phase.
 */
@Injectable({ providedIn: 'root' })
export class BrowseSubsetPrepService {
  private router = inject(Router);
  private projectionApi = inject(ProjectionApiService);
  private browseSubset = inject(BrowseSubsetService);
  private toast = inject(ToastService);

  /** True while a build is in flight, i.e. while Find shows the wait overlay. */
  readonly preparing = signal(false);
  /**
   * The latest build progress, shaped as a {@link ProgressEvent} so it feeds
   * the same `progressBarState` / `formatEta` helpers every other progress
   * surface in the app uses.
   */
  readonly progress = signal<ProgressEvent | null>(null);

  /** Resolved `<vt-progress-bar>` inputs for the current build. */
  readonly bar = computed(() => progressBarState(this.progress()));

  /** `"12 / 345"` for the phase's own counts, or `''` when it has none. */
  readonly count = computed(() => {
    const p = this.progress();
    const total = p?.total ?? 0;
    if (total <= 0) return '';
    return `${(p?.current ?? 0).toLocaleString()} / ${total.toLocaleString()}`;
  });

  /**
   * The phase line under the bar: `"Step 2 of 3 · building pyramid"`. The step
   * counter is what makes the wait legible — the UMAP fit is one opaque call
   * that reports no fraction of its own (it ticks elapsed seconds into the
   * message instead), so knowing which of three phases is running is the only
   * honest sense of how much work is left.
   */
  readonly detail = computed(() => {
    const p = this.progress();
    const step = p?.step;
    const totalSteps = p?.total_steps;
    const phase = step != null && totalSteps != null && totalSteps > 1 ? `Step ${step} of ${totalSteps}` : '';
    return [phase, p?.message].filter(Boolean).join(' · ');
  });

  readonly eta = computed(() => formatEta(this.progress()?.eta_seconds));

  private poll: PollHandle | null = null;
  private datasetId = '';
  private ids: number[] = [];

  /**
   * Build the subset projection over *ids* and, when it's ready, hand off to
   * `/browse/:datasetId?subset=1`. No-ops on an empty selection or while a
   * previous preparation is still running.
   */
  start(datasetId: string, ids: number[]): void {
    if (this.preparing() || !datasetId || ids.length === 0) return;
    this.datasetId = datasetId;
    this.ids = ids;
    this.preparing.set(true);
    this.progress.set({ message: 'Arranging the items…' });

    this.projectionApi
      .buildSubset(ids)
      .pipe(take(1))
      .subscribe({
        next: (resp) => {
          if (!this.preparing()) return;
          if (resp.status === 'ready') {
            // Already fit (an unchanged subset re-binned from the cached
            // layout) — go straight through with no visible wait.
            this.finish();
            return;
          }
          this.startPoll();
        },
        // The build POST raises the global error toast on its own, so there's
        // nothing to say here beyond dropping the overlay and staying in Find.
        error: () => this.clear(),
      });
  }

  /**
   * Abandon the wait and stay in Find. The background fit keeps running
   * server-side (it has no cancellation hook, and its result is cached by
   * signature), so pressing Browse again picks the same job back up rather
   * than starting a second fit.
   */
  cancel(): void {
    this.clear();
  }

  // --- polling ------------------------------------------------------------

  private startPoll(): void {
    this.stopPoll();
    this.poll = pollUntil<ProjectionMeta>({
      fetch: () => this.projectionApi.getMeta(true),
      apply: (meta) => this.handleMeta(meta),
      onLostContact: () => this.fail('Lost contact with the server while building the map.'),
    });
  }

  /** Apply one projection meta, and say whether the build is still running. */
  private handleMeta(meta: ProjectionMeta): PollStep {
    if (!this.preparing()) return 'stop';
    if (meta.point_count > 0) {
      this.finish();
      return 'stop';
    }
    if (meta.status === 'error') {
      this.fail(meta.error || 'Failed to build the map');
      return 'stop';
    }
    this.progress.set({
      message: meta.message ?? '',
      current: meta.current ?? 0,
      total: meta.total ?? 0,
      step: meta.step ?? null,
      total_steps: meta.total_steps ?? null,
      overall: meta.overall ?? null,
      overall_step_end: meta.overall_step_end ?? null,
    });
    return 'continue';
  }

  /** The map is ready: hand the ids to the browse view and navigate. */
  private finish(): void {
    const datasetId = this.datasetId;
    const ids = this.ids;
    this.clear();
    // Set the handoff only now, so a cancelled preparation never leaves a
    // stale pending subset behind for the next browse to pick up.
    this.browseSubset.set({ datasetId, ids });
    this.router.navigate(['/browse', datasetId], { queryParams: { subset: 1 } });
  }

  private fail(message: string): void {
    this.clear();
    this.toast.error({ message, dedupKey: 'find-browse-build-error' });
  }

  private clear(): void {
    this.stopPoll();
    this.preparing.set(false);
    this.progress.set(null);
  }

  private stopPoll(): void {
    this.poll?.stop();
    this.poll = null;
  }
}
