import { Injectable, inject, signal } from '@angular/core';
import { take } from 'rxjs/operators';
import { Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { ProgressEventsService } from './progress-events.service';
import { ProjectionApiService } from './projection-api.service';
import { pollUntil, type PollHandle, type PollStep } from './poll-until';
import { LoadingTask } from '../models/api.models';
import type { ProgressKind } from '../utils/format-progress';
import type { ProjectionMeta } from '../models/projection.models';

/** Which step of the browse preparation we're on. */
type Phase = 'loading' | 'projecting' | 'error';

interface BrowsePrepState {
  datasetId: string;
  phase: Phase;
  current: number;
  total: number;
  message: string;
  /** Whole-job build position reported by the projection meta: which coarse
   *  phase (arranging → tiling → naming regions) is running, and the stitched
   *  0..1 fraction across all of them. Carried onto the synthetic row so the
   *  dashboard bar fills once across the build instead of restarting per phase. */
  step: number | null;
  totalSteps: number | null;
  overall: number | null;
  /** Whole-job fraction at which the running step's slice ends, so a
   *  count-less step renders as a bounded sweep instead of a shimmer over the
   *  parked fill. See ProgressEvent.overall_step_end. */
  stepEnd: number | null;
  error: string;
}

/** task_id prefix for the synthetic projection-phase row, so the dashboard
 *  can route its Cancel/Dismiss clicks back here instead of to a real
 *  dataset-load task. */
const SENTINEL_PREFIX = '__browseprep__';

/**
 * Orchestrates the Dashboard's "Browse" button: load the dataset (if needed)
 * **and** build its projection, surfacing progress on the dataset's grid row,
 * and only navigate to the browse view once both are ready. Mirrors how the
 * Train button loads a dataset before entering the label view — the user's
 * complaint about a missing load or missing projection is raised on the
 * dashboard, never after we've left it for the browse window.
 *
 * Progress for the **load** phase rides the existing SSE loading-task channel
 * (the same row the Load button shows); this service only synthesizes a row
 * for the **projection** phase, which the projection job exposes via meta
 * polling rather than that channel.
 */
@Injectable({ providedIn: 'root' })
export class BrowsePrepService {
  private router = inject(Router);
  private activeContext = inject(ActiveContextService);
  private contextSwitch = inject(ContextSwitchService);
  private progressEvents = inject(ProgressEventsService);
  private projectionApi = inject(ProjectionApiService);

  // A signal, not a `BehaviorSubject`: the dashboard reads this state straight
  // off `displayTask()` / `taskKind()` / `preparing` in its template, with no
  // `AsyncPipe` or `toSignal` bridge anywhere. Under zoneless a subject read
  // through those accessors notifies nobody (docs/FRONTEND.md §5), so the
  // projection row used to repaint only when some *other* signal happened to
  // dirty the view — in practice the 5s SSE heartbeat, against a 1s poll. A
  // signal read inside a getter or method during template evaluation is tracked
  // as a dependency of that view, so every poll now repaints the row.
  private readonly state = signal<BrowsePrepState | null>(null);

  private poll: PollHandle | null = null;

  /** True while a browse preparation is in flight (not counting the
   *  terminal error state, which waits for the user to dismiss). */
  get preparing(): boolean {
    const s = this.state();
    return !!s && s.phase !== 'error';
  }

  /**
   * Begin preparing *datasetId* for browsing: ensure it's loaded and active,
   * then ensure its projection is built, then navigate to the browse view.
   * No-ops if a preparation is already running.
   */
  prepareAndBrowse(datasetId: string): void {
    if (this.preparing) return;
    this.stopPoll();
    this.state.set({
      datasetId,
      phase: 'loading',
      current: 0,
      total: 0,
      message: 'Loading dataset…',
      step: null,
      totalSteps: null,
      overall: null,
      stepEnd: null,
      error: '',
    });

    // Preserve the active detector (browsing doesn't need one, but clearing it
    // would clobber the user's selection); this mirrors browseContextGuard.
    this.contextSwitch
      .applyActivePair(datasetId, this.activeContext.modelId || '')
      .pipe(take(1))
      .subscribe({
        next: () => {
          if (!this.isCurrent(datasetId)) return;
          // A failed load leaves an errored task on the SSE row, which the
          // dashboard already shows; bail out silently rather than stacking a
          // second (projection) error on top of it.
          const loadFailed = this.progressEvents.loadingTasks().some(
            (t) => t.dataset_id === datasetId && !!t.error,
          );
          if (loadFailed) {
            this.clear();
            return;
          }
          this.startProjection(datasetId);
        },
        complete: () => {
          // The switch was superseded/cancelled before emitting (e.g. a
          // competing context switch). Don't leave the prep latched in the
          // loading phase, which would keep the dashboard disabled forever.
          const s = this.state();
          if (s && s.datasetId === datasetId && s.phase === 'loading') {
            this.clear();
          }
        },
      });
  }

  /** Cancel an in-flight preparation (Cancel button on the projection row). */
  cancel(): void {
    this.clear();
  }

  /** Dismiss a finished-with-error preparation (Dismiss button on the row). */
  dismiss(): void {
    this.clear();
  }

  /** True when *taskId* is this service's synthetic projection row. */
  ownsTask(taskId: string): boolean {
    return taskId.startsWith(SENTINEL_PREFIX);
  }

  /** Which progress vocabulary the dashboard should render the row with. */
  taskKind(datasetId: string): ProgressKind {
    const s = this.state();
    return s && s.datasetId === datasetId && s.phase !== 'loading' ? 'projection' : 'dataset';
  }

  /**
   * The synthetic loading-task row to show for *datasetId*, or ``null`` to
   * defer to the real SSE load task. We only own the row during the
   * projection phase (and a brief pre-load "Preparing…" flash before the SSE
   * task appears); the load phase is the real task's to render.
   */
  displayTask(datasetId: string): LoadingTask | null {
    const s = this.state();
    if (!s || s.datasetId !== datasetId) return null;

    if (s.phase === 'loading') {
      // Defer to the real SSE task once it shows; until then, a placeholder
      // so the row gives immediate feedback instead of flashing blank.
      const hasReal = this.progressEvents.loadingTasks().some(
        (t) => t.dataset_id === datasetId && t.status !== 'idle',
      );
      if (hasReal) return null;
      return this.synthTask(datasetId, 'loading', 'Loading dataset…', 0, 0, '');
    }
    if (s.phase === 'error') {
      return this.synthTask(datasetId, 'idle', '', 0, 0, s.error);
    }
    return {
      ...this.synthTask(datasetId, 'building', s.message, s.current, s.total, ''),
      step: s.step,
      total_steps: s.totalSteps,
      overall: s.overall,
      overall_step_end: s.stepEnd,
    };
  }

  // --- projection phase ---

  private startProjection(datasetId: string): void {
    this.patch({
      phase: 'projecting',
      current: 0,
      total: 0,
      message: 'Building the map…',
      step: null,
      totalSteps: null,
      overall: null,
      stepEnd: null,
    });
    this.projectionApi
      .getMeta()
      .pipe(take(1))
      .subscribe({
        next: (meta) => {
          // The same verdict the poll acts on: a build already in flight means
          // start watching it, anything else has settled on its own.
          if (this.handleMeta(datasetId, meta) === 'continue') this.startPoll(datasetId);
        },
        error: (err) => this.fail(datasetId, this.errMessage(err, 'Failed to build the map')),
      });
  }

  /**
   * Apply one projection meta, and say whether the build is still running.
   * Shared by the opening fetch and every poll tick, so both read a build
   * status exactly the same way.
   */
  private handleMeta(datasetId: string, meta: ProjectionMeta): PollStep {
    if (!this.isCurrent(datasetId)) return 'stop';

    if (meta.point_count > 0) {
      this.finish(datasetId);
      return 'stop';
    }
    if (meta.status === 'error') {
      this.fail(datasetId, meta.error || 'Failed to build the map');
      return 'stop';
    }
    if (meta.status === 'building') {
      this.patch({
        current: meta.current ?? 0,
        total: meta.total ?? 0,
        message: meta.message || 'Building the map…',
        step: meta.step ?? null,
        totalSteps: meta.total_steps ?? null,
        overall: meta.overall ?? null,
        stepEnd: meta.overall_step_end ?? null,
      });
      return 'continue';
    }
    // status === "idle": nothing built yet. Kick off the build, and stand the
    // poll down until the POST answers so a still-idle server is never sent a
    // second build request per tick.
    this.projectionApi
      .build()
      .pipe(take(1))
      .subscribe({
        next: (resp) => {
          if (!this.isCurrent(datasetId)) return;
          if (resp.status === 'ready') {
            this.finish(datasetId);
            return;
          }
          this.startPoll(datasetId);
        },
        error: (err) => this.fail(datasetId, this.errMessage(err, 'Failed to build the map')),
      });
    return 'stop';
  }

  private startPoll(datasetId: string): void {
    this.stopPoll();
    this.poll = pollUntil<ProjectionMeta>({
      fetch: () => this.projectionApi.getMeta(),
      apply: (meta) => this.handleMeta(datasetId, meta),
      onLostContact: () =>
        this.fail(datasetId, 'Lost contact with the server while building the map.'),
    });
  }

  private finish(datasetId: string): void {
    this.clear();
    this.router.navigate(['/browse', datasetId]);
  }

  private fail(datasetId: string, message: string): void {
    if (!this.isCurrent(datasetId)) return;
    this.stopPoll();
    this.patch({ phase: 'error', error: message });
  }

  // --- helpers ---

  private isCurrent(datasetId: string): boolean {
    const s = this.state();
    return !!s && s.datasetId === datasetId && s.phase !== 'error';
  }

  private patch(partial: Partial<BrowsePrepState>): void {
    const s = this.state();
    if (!s) return;
    this.state.set({ ...s, ...partial });
  }

  private clear(): void {
    this.stopPoll();
    this.state.set(null);
  }

  private stopPoll(): void {
    this.poll?.stop();
    this.poll = null;
  }

  private synthTask(
    datasetId: string,
    status: string,
    message: string,
    current: number,
    total: number,
    error: string,
  ): LoadingTask {
    return {
      status,
      message,
      current,
      total,
      task_id: `${SENTINEL_PREFIX}${datasetId}`,
      name: '',
      created_at: 0,
      dataset_id: datasetId,
      error,
    };
  }

  private errMessage(err: unknown, fallback: string): string {
    if (err instanceof HttpErrorResponse) {
      const body = err.error as { message?: string; error?: string } | undefined;
      return body?.message || body?.error || fallback;
    }
    return fallback;
  }
}
