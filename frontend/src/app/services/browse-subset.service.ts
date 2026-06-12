import { Injectable } from '@angular/core';

/** A subset of media ids to browse as its own UMAP projection. */
export interface BrowseSubset {
  /** Dataset the ids belong to (guards against a stale handoff). */
  datasetId: string;
  /** Media ids to project (e.g. the positives of a Find run). */
  ids: number[];
  /** Human label for the browse header, e.g. the detector name. */
  label: string;
}

/**
 * Carries the subset selection from the Find view to the Browse view across a
 * route navigation. The Find view stashes the positive ids here and navigates
 * to `/browse/:datasetId?subset=1`; the Browse view reads them back on init.
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
