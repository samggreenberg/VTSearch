import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { take } from 'rxjs/operators';
import { Router } from '@angular/router';
import { HttpErrorResponse } from '@angular/common/http';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { ProgressEventsService } from './progress-events.service';
import { ProjectionApiService } from './projection-api.service';
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
  error: string;
}

/** task_id prefix for the synthetic projection-phase row, so the dashboard
 *  can route its Cancel/Dismiss clicks back here instead of to a real
 *  dataset-load task. */
const SENTINEL_PREFIX = '__browseprep__';

/** Bin shape we pre-build. The 2-D UMAP layout is shared across shapes, so
 *  if the browse view later wants squares it re-bins the frozen layout in
 *  milliseconds rather than re-fitting UMAP. */
const PREP_SHAPE = 'hex' as const;

const MAX_POLL_ERRORS = 5;

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
  private readonly stateSubject = new BehaviorSubject<BrowsePrepState | null>(null);
  readonly state$ = this.stateSubject.asObservable();

  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private pollErrors = 0;

  constructor(
    private router: Router,
    private activeContext: ActiveContextService,
    private contextSwitch: ContextSwitchService,
    private progressEvents: ProgressEventsService,
    private projectionApi: ProjectionApiService,
  ) {}

  /** True while a browse preparation is in flight (not counting the
   *  terminal error state, which waits for the user to dismiss). */
  get preparing(): boolean {
    const s = this.stateSubject.value;
    return !!s && s.phase !== 'error';
  }

  /**
   * Begin preparing *datasetId* for browsing: ensure it's loaded and active,
   * then ensure its projection is built, then navigate to the browse view.
   * No-ops if a preparation is already running.
   */
  prepareAndBrowse(datasetId: string): void {
    if (this.preparing) return;
    this.clearTimer();
    this.stateSubject.next({
      datasetId,
      phase: 'loading',
      current: 0,
      total: 0,
      message: 'Loading dataset…',
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
          const loadFailed = this.progressEvents.loadingTasks.some(
            (t) => t.dataset_id === datasetId && !!t.error,
          );
          if (loadFailed) {
            this.clear();
            return;
          }
          this.startProjection(datasetId);
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
    const s = this.stateSubject.value;
    return s && s.datasetId === datasetId && s.phase !== 'loading' ? 'projection' : 'dataset';
  }

  /**
   * The synthetic loading-task row to show for *datasetId*, or ``null`` to
   * defer to the real SSE load task. We only own the row during the
   * projection phase (and a brief pre-load "Preparing…" flash before the SSE
   * task appears); the load phase is the real task's to render.
   */
  displayTask(datasetId: string): LoadingTask | null {
    const s = this.stateSubject.value;
    if (!s || s.datasetId !== datasetId) return null;

    if (s.phase === 'loading') {
      // Defer to the real SSE task once it shows; until then, a placeholder
      // so the row gives immediate feedback instead of flashing blank.
      const hasReal = this.progressEvents.loadingTasks.some(
        (t) => t.dataset_id === datasetId && t.status !== 'idle',
      );
      if (hasReal) return null;
      return this.synthTask(datasetId, 'loading', 'Loading dataset…', 0, 0, '');
    }
    if (s.phase === 'error') {
      return this.synthTask(datasetId, 'idle', '', 0, 0, s.error);
    }
    return this.synthTask(datasetId, 'building', s.message, s.current, s.total, '');
  }

  // --- projection phase ---

  private startProjection(datasetId: string): void {
    this.patch({ phase: 'projecting', current: 0, total: 0, message: 'Building projection…' });
    this.pollErrors = 0;
    this.projectionApi
      .getMeta(PREP_SHAPE)
      .pipe(take(1))
      .subscribe({
        next: (meta) => this.handleMeta(datasetId, meta),
        error: (err) => this.fail(datasetId, this.errMessage(err, 'Failed to load projection')),
      });
  }

  private handleMeta(datasetId: string, meta: ProjectionMeta): void {
    if (!this.isCurrent(datasetId)) return;

    if (meta.point_count > 0) {
      this.finish(datasetId);
      return;
    }
    if (meta.status === 'error') {
      this.fail(datasetId, meta.error || 'Projection build failed');
      return;
    }
    if (meta.status === 'building') {
      this.patch({
        current: meta.current ?? 0,
        total: meta.total ?? 0,
        message: meta.message || 'Building projection…',
      });
      this.schedulePoll(datasetId);
      return;
    }
    // status === "idle": nothing built yet. Kick off the build.
    this.projectionApi
      .build(PREP_SHAPE)
      .pipe(take(1))
      .subscribe({
        next: (resp) => {
          if (!this.isCurrent(datasetId)) return;
          if (resp.status === 'ready') {
            this.finish(datasetId);
            return;
          }
          this.schedulePoll(datasetId);
        },
        error: (err) => this.fail(datasetId, this.errMessage(err, 'Failed to start projection build')),
      });
  }

  private schedulePoll(datasetId: string): void {
    this.clearTimer();
    this.pollTimer = setTimeout(() => {
      if (!this.isCurrent(datasetId)) return;
      this.projectionApi
        .getMeta(PREP_SHAPE)
        .pipe(take(1))
        .subscribe({
          next: (meta) => {
            this.pollErrors = 0;
            this.handleMeta(datasetId, meta);
          },
          error: () => {
            this.pollErrors += 1;
            if (this.pollErrors >= MAX_POLL_ERRORS) {
              this.fail(datasetId, 'Lost contact with the server while building the projection.');
              return;
            }
            this.schedulePoll(datasetId);
          },
        });
    }, 1000);
  }

  private finish(datasetId: string): void {
    this.clear();
    this.router.navigate(['/browse', datasetId]);
  }

  private fail(datasetId: string, message: string): void {
    if (!this.isCurrent(datasetId)) return;
    this.clearTimer();
    this.patch({ phase: 'error', error: message });
  }

  // --- helpers ---

  private isCurrent(datasetId: string): boolean {
    const s = this.stateSubject.value;
    return !!s && s.datasetId === datasetId && s.phase !== 'error';
  }

  private patch(partial: Partial<BrowsePrepState>): void {
    const s = this.stateSubject.value;
    if (!s) return;
    this.stateSubject.next({ ...s, ...partial });
  }

  private clear(): void {
    this.clearTimer();
    this.stateSubject.next(null);
  }

  private clearTimer(): void {
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
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
