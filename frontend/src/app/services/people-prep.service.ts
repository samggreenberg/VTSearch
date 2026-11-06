import { Injectable, inject, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { Router } from '@angular/router';
import { take } from 'rxjs/operators';

import type { PeopleList } from '../generated/api-client/models/people-list';
import { LoadingTask } from '../models/api.models';
import { apiErrorMessage } from '../utils/api-error';
import type { ProgressKind } from '../utils/format-progress';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { PeopleApiService } from './people-api.service';
import { pollUntil, type PollHandle, type PollStep } from './poll-until';
import { ProgressEventsService } from './progress-events.service';

/** Which step of opening People we're on. */
type Phase = 'loading' | 'finding' | 'error';

interface PeoplePrepState {
  datasetId: string;
  phase: Phase;
  current: number;
  total: number;
  message: string;
  /** The fit's whole-job position, as the people list reports it while
   *  `building` (the same fields a map build reports). */
  step: number | null;
  totalSteps: number | null;
  overall: number | null;
  stepEnd: number | null;
  error: string;
}

/** task_id prefix of the synthetic finding-people row, so the Dashboard can
 *  route its Cancel/Dismiss clicks back here rather than to a real load. */
const SENTINEL_PREFIX = '__peopleprep__';

const FINDING = 'Finding people…';
const FAILED = 'Failed to find the people';

/**
 * The Dashboard's **People** menu item (#5590): load the Face dataset if it
 * needs it **and** find its people, with progress on the dataset's row, and
 * open `/people/:datasetId` only once both are done. The People twin of
 * {@link BrowsePrepService}, which waits for the map the same way: a dataset
 * that fails to load, or whose people can't be found, says so on the
 * Dashboard rather than after the user has left it.
 *
 * The load's progress rides the SSE loading-task channel (the row the Load
 * button shows); this service synthesises a row only while the people are
 * being found, which `GET /api/people` reports as `building` and this polls.
 * An uncalibrated face space (`not_calibrated`) opens the view, which says
 * why there are no people; it is an answer, not a failure.
 *
 * Browse's **People** button skips this: its dataset is loaded already, and
 * the view polls a fit still running itself.
 */
@Injectable({ providedIn: 'root' })
export class PeoplePrepService {
  private router = inject(Router);
  private activeContext = inject(ActiveContextService);
  private contextSwitch = inject(ContextSwitchService);
  private progressEvents = inject(ProgressEventsService);
  private peopleApi = inject(PeopleApiService);

  // A signal for the same reason as BrowsePrepService's: the Dashboard reads
  // it through `displayTask()` / `taskKind()` / `preparing` in its template.
  private readonly state = signal<PeoplePrepState | null>(null);

  private poll: PollHandle | null = null;

  /** True while opening People is in flight (not the terminal error state,
   *  which waits for the user to dismiss it). */
  get preparing(): boolean {
    const s = this.state();
    return !!s && s.phase !== 'error';
  }

  /**
   * Make *datasetId* the active dataset (loading it if need be), find its
   * people, then open the People view. A no-op while one is already running.
   */
  prepareAndOpen(datasetId: string): void {
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

    // Keep the active detector, as the route guard and Browse's prep do:
    // People needs none, and clearing it would clobber the user's choice.
    this.contextSwitch
      .applyActivePair(datasetId, this.activeContext.modelId || '')
      .pipe(take(1))
      .subscribe({
        next: () => {
          if (!this.isCurrent(datasetId)) return;
          // A failed load leaves its errored task on the row already.
          const loadFailed = this.progressEvents
            .loadingTasks()
            .some((t) => t.dataset_id === datasetId && !!t.error);
          if (loadFailed) {
            this.clear();
            return;
          }
          this.startFinding(datasetId);
        },
        complete: () => {
          // The switch was superseded before it emitted; don't stay latched.
          const s = this.state();
          if (s && s.datasetId === datasetId && s.phase === 'loading') this.clear();
        },
      });
  }

  /** Cancel (the row's Cancel). The fit, once started, finishes server-side
   *  and is kept: the next People ask reads it. */
  cancel(): void {
    this.clear();
  }

  /** Dismiss a preparation that ended in an error (the row's Dismiss). */
  dismiss(): void {
    this.clear();
  }

  /** True when *taskId* is this service's synthetic row. */
  ownsTask(taskId: string): boolean {
    return taskId.startsWith(SENTINEL_PREFIX);
  }

  /** Whether this service is preparing (or failed preparing) *datasetId*. */
  holds(datasetId: string): boolean {
    return this.state()?.datasetId === datasetId;
  }

  /** The progress vocabulary for *datasetId*'s row: People's own once the
   *  load is done, the load's before. */
  taskKind(datasetId: string): ProgressKind {
    const s = this.state();
    return s && s.datasetId === datasetId && s.phase !== 'loading' ? 'people' : 'dataset';
  }

  /**
   * The synthetic loading-task row for *datasetId*, or `null` to defer to the
   * real SSE load task (or to nothing). Owned while the people are found, and
   * as a brief "Loading dataset…" before the SSE task appears.
   */
  displayTask(datasetId: string): LoadingTask | null {
    const s = this.state();
    if (!s || s.datasetId !== datasetId) return null;

    if (s.phase === 'loading') {
      const hasReal = this.progressEvents
        .loadingTasks()
        .some((t) => t.dataset_id === datasetId && t.status !== 'idle');
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

  // --- finding the people ---

  private startFinding(datasetId: string): void {
    this.patch({ phase: 'finding', current: 0, total: 0, message: FINDING });
    this.peopleApi
      .list()
      .pipe(take(1))
      .subscribe({
        next: (list) => {
          if (this.handleList(datasetId, list) === 'continue') this.startPoll(datasetId);
        },
        error: (err) => this.fail(datasetId, errMessage(err)),
      });
  }

  /** Apply one people list, and say whether the fit is still running. */
  private handleList(datasetId: string, list: PeopleList): PollStep {
    if (!this.isCurrent(datasetId)) return 'stop';
    switch (list.status) {
      case 'ready':
      case 'not_calibrated':
        this.finish(datasetId);
        return 'stop';
      case 'error':
        this.fail(datasetId, list.error || FAILED);
        return 'stop';
      default:
        this.patch({
          current: list.current ?? 0,
          total: list.total ?? 0,
          message: list.message || FINDING,
          step: list.step ?? null,
          totalSteps: list.total_steps ?? null,
          overall: list.overall ?? null,
          stepEnd: list.overall_step_end ?? null,
        });
        return 'continue';
    }
  }

  private startPoll(datasetId: string): void {
    this.stopPoll();
    this.poll = pollUntil<PeopleList>({
      fetch: () => this.peopleApi.list(),
      apply: (list) => this.handleList(datasetId, list),
      onLostContact: () => this.fail(datasetId, 'Lost contact with the server while finding the people.'),
    });
  }

  private finish(datasetId: string): void {
    this.clear();
    this.router.navigate(['/people', datasetId]);
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

  private patch(partial: Partial<PeoplePrepState>): void {
    const s = this.state();
    if (s) this.state.set({ ...s, ...partial });
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
}

function errMessage(err: unknown): string {
  return err instanceof HttpErrorResponse ? apiErrorMessage(err, FAILED) : FAILED;
}
