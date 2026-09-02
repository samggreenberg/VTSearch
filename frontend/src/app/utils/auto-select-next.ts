import type { SelectMode, SortedItem } from '../services/sort-state.service';

/**
 * What the auto-advance should do next, as data rather than as a side effect.
 *
 * `diversity` is deliberately *not* resolved here: the New pick is a server
 * round-trip (`GET /api/coverage-atlas/next`), so the rule reports that a
 * diversity probe is owed and the caller fires it. Keeping the request out
 * means the whole pick rule stays synchronous and pure — which is what makes
 * it unit-testable, and what lets `scripts/check-eval-app-sync.py` pin a
 * digest of the rule alone rather than of the component plumbing around it.
 */
export type AutoSelectPick =
  | { kind: 'media'; id: number }
  | { kind: 'diversity' }
  | { kind: 'none' };

export interface AutoSelectInput {
  /** The loaded sort window, descending by score. `null` before any sort. */
  sortOrder: SortedItem[] | null;
  selectMode: SelectMode;
  /** The *acquisition* cut (`SortStateService.acqThreshold`), not the
   *  reporting threshold the user is shown — see #2876. */
  acqThreshold: number | null;
  goodVotes: ReadonlySet<number>;
  badVotes: ReadonlySet<number>;
  /** Treated as already-voted. Set when advancing off the item just voted on,
   *  whose vote has not yet landed in `goodVotes` / `badVotes`. */
  excludeId?: number;
}

/**
 * Pick the next media to show, given the current sort window and select mode.
 *
 * This is the app's auto-advance rule, and the Python eval harness reproduces
 * it in `vtscore/eval/al_strategies.py` (`_hard_pick_by_index` for the `hard`
 * branch) so a simulated Autopilot user clicks the way a real one does. That
 * copy is pinned by the `autopilot.auto_select_next` mirror in
 * `scripts/check-eval-app-sync.py`: changing the rule here trips the gate and
 * asks whoever changed it to port the change. Keep the two in step.
 *
 * The three modes:
 *
 * - **top** — the highest-ranked unvoted item.
 * - **hard** — the unvoted item nearest the cutoff *by rank index*, not by
 *   score. Rank space is the point: a score-space `argmin |score - t|` biases
 *   toward whichever side of the line happens to be denser, which on a
 *   saturated cold-start model is exactly the failure being avoided. The
 *   threshold index spans voted and unvoted rows alike (the full window is
 *   ranked and voted rows are merely skipped), so the cutoff position does not
 *   drift as votes accumulate.
 * - **new** — defer to the coverage atlas; reported as `diversity`.
 */
export function autoSelectNext(input: AutoSelectInput): AutoSelectPick {
  const { sortOrder, selectMode, acqThreshold, goodVotes, badVotes, excludeId } = input;
  if (!sortOrder || sortOrder.length === 0) return { kind: 'none' };

  const isVoted = (id: number): boolean =>
    id === excludeId || goodVotes.has(id) || badVotes.has(id);

  if (selectMode === 'top') {
    const next = sortOrder.find((s) => !isVoted(s.id));
    return next ? { kind: 'media', id: next.id } : { kind: 'none' };
  }

  if (selectMode === 'hard') {
    if (acqThreshold === null) return { kind: 'none' };
    // The first position whose score is at or below the cut, in the descending
    // ranking; `length` when every row is above it.
    let thresholdIndex = sortOrder.length;
    for (let i = 0; i < sortOrder.length; i++) {
      if (sortOrder[i].score <= acqThreshold) {
        thresholdIndex = i;
        break;
      }
    }
    let best: SortedItem | null = null;
    let bestDist = Infinity;
    for (let i = 0; i < sortOrder.length; i++) {
      if (isVoted(sortOrder[i].id)) continue;
      const dist = Math.abs(i - thresholdIndex);
      if (dist < bestDist) {
        bestDist = dist;
        best = sortOrder[i];
      }
    }
    return best ? { kind: 'media', id: best.id } : { kind: 'none' };
  }

  if (selectMode === 'new') return { kind: 'diversity' };

  return { kind: 'none' };
}
