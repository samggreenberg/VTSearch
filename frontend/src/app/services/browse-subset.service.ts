import { Injectable } from '@angular/core';
import type { AutoDetectResultsData } from '../models/api.models';

/** Which of a Find run's rows the Find Results dialog lists, and so browses. */
export type RunBrowseSides = 'good' | 'bad' | 'both';

/**
 * Where a subset browse was launched from, and so where its **Back** returns:
 * the Tune view, or the Find Results dialog of run *runId* (#4615), which
 * reopens on the Dashboard. Rides the browse URL's query
 * (`from=results&run=…&sides=…`) rather than the in-memory handoff, so it
 * survives a reload, and names enough for the browse view to rebuild a run's
 * map without the handoff (#5129).
 */
export type BrowseSubsetOrigin = { kind: 'test' } | { kind: 'results'; runId: string; sides: RunBrowseSides };

/** The `/browse/:datasetId` query params that open a subset browse from *origin*. */
export function subsetBrowseQueryParams(origin: BrowseSubsetOrigin): Record<string, string | number> {
  return origin.kind === 'results'
    ? { subset: 1, from: 'results', run: origin.runId, sides: origin.sides }
    : { subset: 1 };
}

/** The `sides` query param as a {@link RunBrowseSides}: Good, the dialog's
 *  default, when it is absent or unrecognised. */
export function runBrowseSidesParam(raw: string | null): RunBrowseSides {
  return raw === 'bad' || raw === 'both' ? raw : 'good';
}

/**
 * The media ids a Browse of *run*'s *sides* rows lays out, once each: a media
 * two detectors both called Good is one item on the map. The Find Results
 * dialog's Browse button and the browse view rebuilding that map from a link
 * both ask this, so they lay out the same set.
 */
export function runBrowseIds(run: AutoDetectResultsData, sides: RunBrowseSides): number[] {
  const ids = new Set<number>();
  for (const result of Object.values(run.results || {})) {
    const rows = [
      ...(sides === 'bad' ? [] : result.hits || []),
      ...(sides === 'good' ? [] : result.negative_hits || []),
    ];
    for (const hit of rows) {
      if (typeof hit.id === 'number') ids.add(hit.id);
    }
  }
  return [...ids];
}

/** A subset of media ids to browse as its own UMAP projection. */
export interface BrowseSubset {
  /** Dataset the ids belong to (guards against a stale handoff). */
  datasetId: string;
  /** Media ids to project (e.g. the positives of a Find run). */
  ids: number[];
}

/**
 * Carries the subset selection from the Tune view (or the Find Results dialog)
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
