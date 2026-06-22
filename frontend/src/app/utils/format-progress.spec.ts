import {
  formatProgressHeader,
  isProgressIndeterminate,
  progressBarState,
} from './format-progress';

describe('progressBarState', () => {
  it('prefers the whole-job overall fraction', () => {
    const state = progressBarState({ overall: 0.4, current: 5, total: 10 });
    expect(state).toEqual({ value: 0.4, max: 1, indeterminate: false });
  });

  it('clamps the overall fraction to [0, 1]', () => {
    expect(progressBarState({ overall: 1.5 }).value).toBe(1);
    expect(progressBarState({ overall: -0.2 }).value).toBe(0);
  });

  it('falls back to current/total when overall is absent', () => {
    expect(progressBarState({ current: 3, total: 12 })).toEqual({
      value: 3,
      max: 12,
      indeterminate: false,
    });
  });

  it('is indeterminate when neither overall nor a positive total is known', () => {
    expect(progressBarState({ current: 0, total: 0 }).indeterminate).toBe(true);
    expect(progressBarState({}).indeterminate).toBe(true);
    expect(progressBarState(null).indeterminate).toBe(true);
  });

  it('treats overall=0 as a determinate bar at 0%, not a spinner', () => {
    // A multi-step job in an indeterminate first phase sits at the step floor
    // (0%) as a real whole-job position rather than reverting to a spinner.
    expect(progressBarState({ overall: 0 })).toEqual({
      value: 0,
      max: 1,
      indeterminate: false,
    });
  });
});

describe('isProgressIndeterminate', () => {
  it('is false when an overall fraction is present', () => {
    expect(isProgressIndeterminate({ overall: 0 })).toBe(false);
  });

  it('is true when nothing is known', () => {
    expect(isProgressIndeterminate({})).toBe(true);
  });
});

describe('formatProgressHeader step count', () => {
  it('surfaces "step S of T" in the header for multi-step jobs', () => {
    const { header } = formatProgressHeader(
      { status: 'embedding', message: 'Embedding files', step: 3, total_steps: 4 },
      'dataset',
    );
    expect(header).toContain('step 3 of 4');
    expect(header).toContain('Loading dataset');
    expect(header).toContain('embedding files');
  });

  it('omits the step count for single-step jobs', () => {
    const { header } = formatProgressHeader(
      { status: 'downloading', message: 'Fetching', step: 1, total_steps: 1 },
      'dataset',
    );
    expect(header).not.toContain('step 1 of 1');
  });
});
