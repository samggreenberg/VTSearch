import type { AutoDetectDetectorResult, AutoDetectHit, AutoDetectResultsData } from '../models/api.models';
import { BALANCE_PRESETS, nearestBalancePreset } from './line-balance';

/**
 * A Find run's results at another notch of the Threshold (#5540).
 *
 * A run cannot retrain, but the balance only moves the cut: the server sends
 * each detector's cut at every preset (`lines`, re-derived from what the train
 * cached), and the run already holds every item's score (`hits` above its own
 * cut, `negative_hits` below). So a notch's Good is the items scoring at or
 * above that notch's cut, and its Bad the rest. The detector's own notch, or a
 * notch it carries no cut for, is split as the run split it.
 */

/** One detector's results split at a notch. */
export interface NotchSplit {
  good: AutoDetectHit[];
  bad: AutoDetectHit[];
  /** The cut the split was made at. */
  threshold: number | undefined;
  /** Whether the split is another notch's, not the run's own. */
  moved: boolean;
}

/** The notch a detector's run was cut at: the preset nearest its balance, or `null` when it names none. */
export function ownNotch(result: AutoDetectDetectorResult): number | null {
  return result.beta != null ? nearestBalancePreset(result.beta).value : null;
}

/** *result*'s cut at *notch*, or `null` when it carries none there. */
export function cutAt(result: AutoDetectDetectorResult, notch: number): number | null {
  return result.lines?.find((line) => line.beta === notch)?.threshold ?? null;
}

/** *result*'s Good and Bad at *notch*; `null`, its own notch, or one it has no cut for split as the run did. */
export function splitAtNotch(result: AutoDetectDetectorResult, notch: number | null): NotchSplit {
  const good = result.hits || [];
  const bad = result.negative_hits || [];
  const cut = notch == null || notch === ownNotch(result) ? null : cutAt(result, notch);
  if (cut == null) return { good, bad, threshold: result.threshold, moved: false };
  // Both lists run highest score first, Good above Bad, so their join does too.
  const scored = (hit: AutoDetectHit) => (hit.score ?? -Infinity) >= cut;
  const all = [...good, ...bad];
  return { good: all.filter(scored), bad: all.filter((hit) => !scored(hit)), threshold: cut, moved: true };
}

/** The notch every detector of *run* was cut at, or `null` when they differ or name none. */
export function runNotch(run: AutoDetectResultsData): number | null {
  const notches = new Set(Object.values(run.results || {}).map(ownNotch));
  return notches.size === 1 ? [...notches][0] : null;
}

/** Whether another notch would list anything differently: some detector's cut moves with the balance. */
export function notchMoves(run: AutoDetectResultsData): boolean {
  return Object.values(run.results || {}).some((result) => {
    const cuts = new Set(BALANCE_PRESETS.map((preset) => cutAt(result, preset.value)));
    return !cuts.has(null) && cuts.size > 1;
  });
}

/** A notch as a URL's query param reads it back: a preset, else `null` (the run's own). */
export function notchParam(raw: string | null): number | null {
  const value = raw == null || raw === '' ? NaN : Number(raw);
  return BALANCE_PRESETS.some((preset) => preset.value === value) ? value : null;
}
