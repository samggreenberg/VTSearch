import { Injectable, inject } from '@angular/core';
import { AutopilotStateService } from './autopilot-state.service';
import { SortStateService } from './sort-state.service';

/**
 * Which UI flow drove a vote. Mirrors `FLOWS` in
 * `vtscore/datasets/vote_provenance.py`; the server rejects anything else.
 *
 * The values the client can originate are here; `import`, `seed_example` and
 * `unknown` are stamped server-side on paths that have no client at all.
 */
export type VoteFlow =
  | 'autopilot'
  | 'list_review'
  | 'find_verify'
  | 'labelset_review'
  | 'bulk'
  | 'undo';

/**
 * How a voted-on item came to be in front of the user, captured at click time.
 *
 * Recorded, never read back: nothing in the app changes behaviour on these
 * values. They exist because the surfacing context is *not re-derivable* —
 * `sortOrder` is ephemeral client state and the model behind `score` is
 * overwritten by the next retrain — so a vote not annotated now is annotated
 * never. See `docs/plans/provenance-partitioned-calibration.md`.
 *
 * Deliberately narrower than the generated `VoteProvenance` model, which
 * carries every value the wire format allows including the ones only the
 * server ever originates (`import`, `seed_example`, `unknown`). Structural
 * typing bridges the two where a request body is built.
 */
export interface VoteProvenance {
  flow?: VoteFlow;
  /** Autopilot phase; only meaningful when `flow` is `autopilot`. */
  phase?: 'good' | 'bad' | 'hard' | 'new';
  /** How the item was drawn off the ranking. */
  select_mode?: 'top' | 'hard' | 'new';
  /** Which ranking the user was looking at. */
  sort_kind?: 'learned' | 'text' | 'load';
  /** Zero-based position in that ranking. */
  rank_at_vote?: number;
  /** The item's model score when it was surfaced. */
  score_at_vote?: number;
}

/**
 * Assembles the {@link VoteProvenance} for a vote from the state that already
 * describes it.
 *
 * This is a single chokepoint on purpose. Every component vote funnels through
 * `VoteStateService`, which calls this — so no vote surface has to remember to
 * describe itself, and a new one is annotated correctly by default rather than
 * silently recording `unknown`.
 *
 * The four categorical fields are independent axes rather than one fused enum
 * (see the Python module for why): `flow` says who was driving, `select_mode`
 * says how the item was drawn off the ranking, and those two genuinely come
 * apart — a user can pick the `hard` select mode by hand and get autopilot's
 * exact margin-sampled draw.
 */
@Injectable({ providedIn: 'root' })
export class VoteProvenanceService {
  private readonly autopilot = inject(AutopilotStateService);
  private readonly sortState = inject(SortStateService);

  /**
   * Describe how *mediaId* was surfaced.
   *
   * @param mediaId  The media being voted on; used to look up its rank and
   *                 score in the ranking currently on screen.
   * @param flow     Overrides the derived flow for a surface that knows better
   *                 than the autopilot/sort state can tell (a Find-mode
   *                 verify, a bulk action, an undo replay).
   */
  forVote(mediaId: number, flow?: VoteFlow): VoteProvenance {
    const provenance: VoteProvenance = {};

    const phase = this.autopilot.state.phase;
    const labelingPhase =
      phase === 'good' || phase === 'bad' || phase === 'hard' || phase === 'new' ? phase : null;

    // Autopilot only claims the vote while it is actually surfacing items. Its
    // `done` / `exhausted` states leave the user labeling off the last sort by
    // hand, which is list review whatever the panel still says.
    provenance.flow = flow ?? (labelingPhase ? 'autopilot' : 'list_review');
    if (provenance.flow === 'autopilot' && labelingPhase) provenance.phase = labelingPhase;

    provenance.select_mode = this.sortState.selectMode;
    provenance.sort_kind = this.sortState.sortMode;

    const order = this.sortState.sortOrder;
    if (order) {
      // The loaded window always starts at rank 0 and grows by appending
      // pages, so the index into it is the true rank in the full ranking.
      const rank = order.findIndex((item) => item.id === mediaId);
      if (rank >= 0) {
        provenance.rank_at_vote = rank;
        const score = order[rank].score;
        if (Number.isFinite(score)) provenance.score_at_vote = score;
      }
    }

    return provenance;
  }
}
