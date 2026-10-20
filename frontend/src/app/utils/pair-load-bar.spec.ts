import { LoadingTask } from '../models/api.models';
import { pairLoadBar, rowKey, settledRowKeys, watchPairLoad } from './pair-load-bar';

function row(fields: Partial<LoadingTask>): LoadingTask {
  return { task_id: 't', name: '', message: '', status: 'loading', current: 0, total: 0, created_at: 1, ...fields };
}

const datasetLoad = (fields: Partial<LoadingTask> = {}): LoadingTask =>
  row({ task_id: '_regload_d1', dataset_id: 'd1', created_at: 10, ...fields });
const detectorLoad = (fields: Partial<LoadingTask> = {}): LoadingTask =>
  row({ task_id: '_detload_m1', detector_id: 'm1', created_at: 10, ...fields });

describe('settledRowKeys', () => {
  it('keys only the rows that have settled, by id and start time', () => {
    const tasks = [datasetLoad({ status: 'idle' }), detectorLoad()];
    expect([...settledRowKeys(tasks)]).toEqual([rowKey(tasks[0])]);
    expect(rowKey(tasks[0])).toBe('_regload_d1@10');
  });
});

describe('pairLoadBar', () => {
  const fresh = watchPairLoad('d1', 'm1', [], []);

  it('is indeterminate before any load of the pair has published', () => {
    expect(pairLoadBar(fresh, [], [])).toEqual({ value: 0, max: 1, indeterminate: true });
  });

  it("reads one load's own fraction", () => {
    expect(pairLoadBar(fresh, [datasetLoad({ overall: 0.4, current: 2, total: 5 })], [])).toEqual({
      value: 0.4,
      max: 1,
      indeterminate: false,
      pulsing: false,
    });
    expect(pairLoadBar(fresh, [], [detectorLoad({ current: 3, total: 4 })]).value).toBe(0.75);
  });

  it('counts each load as an equal step, a finished one as full', () => {
    const bar = pairLoadBar(fresh, [datasetLoad({ status: 'idle' })], [detectorLoad({ current: 1, total: 2 })]);
    expect(bar.value).toBe(0.75);
    expect(bar.indeterminate).toBe(false);
  });

  it('counts a load with no fraction yet as empty, and pulses', () => {
    const bar = pairLoadBar(fresh, [datasetLoad({ overall: 0.5 })], [detectorLoad()]);
    expect(bar).toEqual({ value: 0.25, max: 1, indeterminate: false, pulsing: true });
  });

  it('is indeterminate while every running load has no fraction', () => {
    expect(pairLoadBar(fresh, [datasetLoad()], [detectorLoad()]).indeterminate).toBe(true);
  });

  it("ignores other datasets' and detectors' loads", () => {
    const others = pairLoadBar(
      fresh,
      [row({ task_id: '_regload_d2', dataset_id: 'd2', overall: 0.9 })],
      [row({ task_id: '_detload_m2', detector_id: 'm2', overall: 0.9 })],
    );
    expect(others.indeterminate).toBe(true);
  });

  it("ignores a finished row left over from an earlier load of the same dataset", () => {
    const earlier = datasetLoad({ status: 'idle', created_at: 5 });
    const watch = watchPairLoad('d1', 'm1', [earlier], []);
    expect(pairLoadBar(watch, [earlier], []).indeterminate).toBe(true);
    // This click's own load reuses the task id but not the start time.
    expect(pairLoadBar(watch, [datasetLoad({ overall: 0.2 })], []).value).toBe(0.2);
  });
});
