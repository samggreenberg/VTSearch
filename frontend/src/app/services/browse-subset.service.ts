import { Injectable } from '@angular/core';

/**
 * Where a subset browse was launched from, and so where its **Back** returns:
 * the Test view, or the Find Results dialog of run *runId* (#4615), which
 * reopens on the Dashboard. Rides the browse URL's query (`from=results&run=…`)
 * rather than the in-memory handoff, so it survives a reload.
 */
export type BrowseSubsetOrigin = { kind: 'test' } | { kind: 'results'; runId: string };

/** The `/browse/:datasetId` query params that open a subset browse from *origin*. */
export function subsetBrowseQueryParams(origin: BrowseSubsetOrigin): Record<string, string | number> {
  return origin.kind === 'results' ? { subset: 1, from: 'results', run: origin.runId } : { subset: 1 };
}

/** A subset of media ids to browse as its own UMAP projection. */
export interface BrowseSubset {
  /** Dataset the ids belong to (guards against a stale handoff). */
  datasetId: string;
  /** Media ids to project (e.g. the positives of a Find run). */
  ids: number[];
}

/**
 * Carries the subset selection from the Test view (or the Find Results dialog)
 * to the Browse view across a route navigation. The launcher stashes the ids
 * here and navigates to `/browse/:datasetId?subset=1`; the Browse view reads
 * them back on init.
 *
 * In-memory only: a hard reload of the browse page loses the handoff (the
 * subset projection is ephemeral and recomputed on demand anyway), and the
 * browse view shows a "re-run Find" message in that case.
 */
@Injectable({ providedIn: 'root' })
export class BrowseSubsetService {
  private pending: BrowseSubset | null = null;
  private returningToFind = false;

  set(subset: BrowseSubset): void {
    this.pending = subset;
  }

  /** Read and clear the pending subset (single-shot handoff). */
  take(): BrowseSubset | null {
    const s = this.pending;
    this.pending = null;
    return s;
  }

  /**
   * Flag the reverse handoff: the user clicked "Back to Find" from the browse
   * view after verifying a selection (Verified Good / Verified Bad). The Find
   * view consumes this on init to SKIP its automatic re-run of detector
   * scoring — which would otherwise re-promote the just-verified items with the
   * unchanged model — and instead just refresh the (already-updated) vote
   * lists. Single-shot, like {@link take}.
   */
  markReturningToFind(): void {
    this.returningToFind = true;
  }

  /** Read and clear the returning-to-Find flag (single-shot). */
  consumeReturningToFind(): boolean {
    const r = this.returningToFind;
    this.returningToFind = false;
    return r;
  }
}
