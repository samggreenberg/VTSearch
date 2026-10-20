import { LoadingTask } from '../models/api.models';
import { ProgressBarState, progressBarState } from './format-progress';

/**
 * The bar under the Dashboard's Tune button (#5098): from the click until the
 * Tune view opens, how far along are the loads the click is waiting on?
 *
 * The click navigates, and the route guard (`activeContextGuard`, through
 * `ContextSwitchService`) holds the route until the pair's dataset and detector
 * are loaded. Those loads publish on the two loading-task channels like any
 * other, so the bar reads them there rather than asking the switch: one bar
 * for the whole click, filling across both loads. With nothing to load (or
 * before the guard's loads have shown up) it is indeterminate, which still
 * covers the route's own chunk download.
 *
 * Kept free of Angular so the rule is a pure function of the two snapshots.
 */

/** A task row's identity across reuse. Loads of one dataset or detector share
 *  a task id (`_regload_<id>`, `_detload_<id>`), and a finished row lingers
 *  on the channel for seconds afterwards (5s, or 30s with an error), so the
 *  id alone cannot tell this load's row from the last one's; `created_at`
 *  can. */
export function rowKey(task: LoadingTask): string {
  return `${task.task_id}@${task.created_at}`;
}

/** Keys of the rows on *tasks* that have already settled: leftovers of
 *  earlier loads, which a waiter must not read as its own load finishing. */
export function settledRowKeys(tasks: readonly LoadingTask[]): ReadonlySet<string> {
  return new Set(tasks.filter((t) => t.status === 'idle').map(rowKey));
}

/** The pair a click is loading, and the rows that were already settled when it was made. */
export interface PairLoadWatch {
  datasetId: string;
  detectorId: string;
  leftovers: ReadonlySet<string>;
}

/** Start watching *datasetId* / *detectorId*: snapshot both channels' settled
 *  rows, before the click's own loads can publish any. */
export function watchPairLoad(
  datasetId: string,
  detectorId: string,
  datasetTasks: readonly LoadingTask[],
  detectorTasks: readonly LoadingTask[],
): PairLoadWatch {
  return {
    datasetId,
    detectorId,
    leftovers: new Set([...settledRowKeys(datasetTasks), ...settledRowKeys(detectorTasks)]),
  };
}

/**
 * One bar for the click's loads: each row of the pair's that is not a leftover
 * counts as one equal step, a finished row as full and a running row at its own
 * fraction. A running row with no fraction yet counts as empty and makes the
 * bar pulse. No rows, or none with a fraction, is an indeterminate bar.
 */
export function pairLoadBar(
  watch: PairLoadWatch,
  datasetTasks: readonly LoadingTask[],
  detectorTasks: readonly LoadingTask[],
): ProgressBarState {
  const rows = [
    ...datasetTasks.filter((t) => t.dataset_id === watch.datasetId),
    ...detectorTasks.filter((t) => t.detector_id === watch.detectorId),
  ].filter((t) => !watch.leftovers.has(rowKey(t)));

  let sum = 0;
  let measured = 0;
  let pulsing = false;
  for (const row of rows) {
    if (row.status === 'idle') {
      sum += 1;
      measured += 1;
      continue;
    }
    const bar = progressBarState(row);
    if (bar.indeterminate) {
      pulsing = true;
      continue;
    }
    sum += bar.max > 0 ? Math.min(1, bar.value / bar.max) : 0;
    measured += 1;
    pulsing ||= !!bar.pulsing;
  }
  if (measured === 0) return { value: 0, max: 1, indeterminate: true };
  return { value: sum / rows.length, max: 1, indeterminate: false, pulsing };
}
