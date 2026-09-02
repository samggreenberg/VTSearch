import { Injectable, OnDestroy, inject, signal } from '@angular/core';
import { MonoTypeOperatorFunction, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { MediaStateService } from './media-state.service';
import { SortStateService } from './sort-state.service';
import { SortingApiService } from './sorting-api.service';
import { VoteStateService } from './vote-state.service';

/**
 * The lifetime of the active (dataset, detector) pair, and the reset that ends
 * it. **Component-provided** (`providers: [PairScopeService]` on `vt-find-view`
 * / `vt-label-view`), never `providedIn: 'root'`: the scope it bounds is a
 * component's, and the two views must not share one subject.
 *
 * ## Why this is a service and not two copies of a comment
 *
 * Every request whose response writes *pair-scoped* state — the ranking, the
 * threshold, the inclusion slider, the dataset name, the vote cache — must be
 * piped through {@link scoped} rather than component teardown, so the work
 * started for the pair we are leaving is torn down the instant the pair
 * switches.
 *
 * Without it a scoring run — minutes long on a large dataset — outlives the
 * switch: whichever response lands last wins, so the *old* pair's ranking and
 * threshold get installed into the new context, and the auto-select lands on a
 * media id that may not exist in the new dataset (broken viewer, image 404s).
 * Even when the old response lands *first*, its `finalize()` drops the wait
 * overlay and re-enables voting while the new run is still going. `takeUntil`
 * also aborts the stale XHR client-side.
 *
 * The failure is silent — stale results repaint over fresh ones; nothing
 * throws — so the ordering rule below is **enforced here rather than described
 * in each view**: {@link resetForNewPair} is the only way to fire the scope,
 * and it supersedes before it touches any of the new pair's state.
 *
 * `ngOnDestroy` on a component-provided service runs when the host component is
 * destroyed, so the destroy-time teardown is automatic; a view must not (and
 * need not) fire the scope by hand.
 *
 * NOTE for callers: a subscription started from `ActiveContextService.modelId$`
 * — which emits *before* `pair$` — must stay on component teardown
 * (`takeUntilDestroyed`), or {@link resetForNewPair}'s teardown would kill the
 * request it just issued for the *new* pair.
 */
@Injectable()
export class PairScopeService implements OnDestroy {
  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private readonly sortingApi = inject(SortingApiService);
  private readonly mediaState = inject(MediaStateService);
  private readonly sortState = inject(SortStateService);
  private readonly voteState = inject(VoteStateService);

  /** Fires whenever the active pair changes — and on host destroy. Private:
   *  the ordering rule is only enforceable while `next()` has one caller. */
  private readonly scope$ = new Subject<void>();

  /** Display name of the active dataset, refreshed on entry and on every pair
   *  change. Signal-backed so the un-bound `getStatus` subscribe schedules
   *  change detection under zoneless (see `docs/FRONTEND.md` §5). */
  readonly datasetName = signal('');

  /**
   * Teardown operator for a pair-scoped request: `pipe(pairScope.scoped())`.
   *
   * This is *cancellation*, not lifetime teardown — the component is very much
   * still alive across a pair switch — so `takeUntilDestroyed()` cannot express
   * it (see `docs/FRONTEND.md` §7, "Subscription teardown vs. cancellation").
   */
  scoped<T>(): MonoTypeOperatorFunction<T> {
    return takeUntil(this.scope$);
  }

  /** Refresh {@link datasetName} for the active pair. */
  loadDatasetName(): void {
    this.datasetsRegistryApi.getStatus().pipe(this.scoped()).subscribe({
      next: (status) => this.datasetName.set(status.display_name || ''),
    });
  }

  /**
   * Seed the inclusion slider from the active detector's per-detector value.
   *
   * `GET /api/inclusion` resolves per-detector, falling back to the
   * user-settings default the first time it is read, so this keeps the slider
   * tracking the detector rather than a stale global.
   *
   * This lives here rather than on `SortStateService` (as #3448 first proposed)
   * because `SortStateService` is a `providedIn: 'root'` signal store that
   * injects nothing: hosting a pair-scoped HTTP call on it would mean either
   * reinventing pair-scoped cancellation inside a singleton or passing a
   * component's scope subject *into* the singleton. #3428 flags the same trap.
   */
  seedInclusion(): void {
    this.sortingApi
      .getInclusion()
      .pipe(this.scoped())
      .subscribe({ next: (resp) => this.sortState.setInclusion(resp.inclusion) });
  }

  /**
   * Drop the ephemeral state the pair we are leaving wrote.
   *
   * Split out of {@link resetForNewPair} for find-view's entry path, which
   * applies the same clear on a fresh navigation (Find and Train share
   * singleton sub-view state, so a Dashboard → Find entry would otherwise
   * render the previous session's ranking against a possibly smaller dataset).
   */
  clearPairState(): void {
    this.sortState.setSortResults([], 0);
    this.sortState.setSortStatus('');
    this.sortState.setSortProgress(0, 0);
    this.voteState.clear();
  }

  /**
   * The pair-change reset, in the one order that is correct.
   *
   * 1. **Supersede.** Every in-flight request scoped to the pair we are leaving
   *    dies here, *before* any of the new pair's state is installed, so no late
   *    response can write the old ranking/threshold into the new context. A
   *    scoping run's `finalize()` runs as part of this teardown, which is why it
   *    cannot clobber the fresh run a caller starts in its tail: that run begins
   *    after the teardown has already settled.
   * 2. **Quiesce.** Anything a superseded subscription owned but had no
   *    `finalize` to reset — a progress poll, a busy flag, a job id. Runs after
   *    the supersede (so nothing can re-set it) and before the reloads.
   * 3. **Clear**, then **reload** for the new pair.
   *
   * Callers run their view-specific tail (the vote load, a re-score) after this
   * returns.
   *
   * @param quiesce Optional step 2 hook, run inside the reset so its placement
   *                relative to the supersede is not a caller's to get wrong.
   */
  resetForNewPair(quiesce?: () => void): void {
    this.scope$.next();
    quiesce?.();
    this.clearPairState();
    this.mediaState.loadMedias();
    this.loadDatasetName();
    this.seedInclusion();
  }

  ngOnDestroy(): void {
    this.scope$.next();
    this.scope$.complete();
  }
}
