import {
  formatEta,
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
  it('surfaces a capitalized "Step S of T" in the header for multi-step jobs', () => {
    const { header } = formatProgressHeader(
      { status: 'embedding', message: 'Embedding files', step: 3, total_steps: 4 },
      'dataset',
    );
    expect(header).toBe('Loading dataset · Step 3 of 4 · Analyzing files');
  });

  it('omits the step count for single-step jobs', () => {
    const { header } = formatProgressHeader(
      { status: 'downloading', message: 'Fetching', step: 1, total_steps: 1 },
      'dataset',
    );
    expect(header).not.toContain('Step 1 of 1');
  });
});

describe('formatProgressHeader detail line', () => {
  it('drops the parentheses around the count and strips the redundant verb', () => {
    // Header already says "· Embedding files", so the per-item detail keeps only
    // the count and the filename — no "(…)" and no repeated "Embedding".
    const { header, detail } = formatProgressHeader(
      { status: 'embedding', message: 'Embedding cats/img.png', current: 12, total: 345 },
      'dataset',
    );
    expect(header).toContain('Analyzing files');
    expect(detail).toBe('12/345 cats/img.png');
  });

  it('shows the bare count when there is no per-item identifier', () => {
    const { detail } = formatProgressHeader(
      { status: 'embedding', message: 'Embedding files', current: 12, total: 345 },
      'dataset',
    );
    // "Embedding files" → verb stripped → "files"; the count carries the signal.
    expect(detail).toBe('12/345 files');
  });

  it('keeps a message that has no leading action verb intact', () => {
    const { detail } = formatProgressHeader(
      { status: 'loading', message: 'cats/img.png', current: 3, total: 9 },
      'dataset',
    );
    expect(detail).toBe('3/9 cats/img.png');
  });
});

describe('formatProgressHeader step-4 finalize phases', () => {
  // The serialize → zip → write window of step 4 used to match no phase, so
  // the header collapsed to a bare "Step 4 of 4" with no descriptor.
  it.each([
    ['Saving to registry…', 'saving dataset'],
    ['Serializing dataset…', 'saving dataset'],
    ['Packaging dataset…', 'saving dataset'],
    ['Building coverage atlas…', 'building coverage atlas'],
    ['Building 2-D projection…', 'building map'],
    ['Building tile pyramid…', 'building map'],
    ['Dropped 3 item(s) with failed embedding…', 'cleaning up'],
  ])('labels %j as "%s"', (message, expectedPhase) => {
    const { header, subtitle } = formatProgressHeader(
      { status: 'loading', message, step: 4, total_steps: 4 },
      'dataset',
    );
    expect(header).toBe(`Loading dataset · Step 4 of 4 · ${capitalizeFirst(expectedPhase)}`);
    expect(subtitle).not.toBe('');
  });
});

function capitalizeFirst(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

describe('formatEta', () => {
  it('returns empty for null, non-finite, or non-positive values', () => {
    expect(formatEta(null)).toBe('');
    expect(formatEta(undefined)).toBe('');
    expect(formatEta(0)).toBe('');
    expect(formatEta(-5)).toBe('');
    expect(formatEta(Infinity)).toBe('');
  });

  it('never claims finer than 10-second granularity sub-minute', () => {
    // A few seconds left snaps below the 10s floor → "< 10 sec", never "< 5 sec".
    expect(formatEta(3)).toBe('< 10 sec left?');
    expect(formatEta(4)).toBe('< 10 sec left?');
    // Just over the floor rounds to the nearest 10s, not 5s.
    expect(formatEta(12)).toBe('10 sec left?');
    expect(formatEta(18)).toBe('20 sec left?');
    expect(formatEta(34)).toBe('30 sec left?');
  });

  it('switches to minutes and hours for larger estimates', () => {
    expect(formatEta(330)).toBe('5.5 min left?');
    expect(formatEta(7200)).toBe('2 hr left?');
  });
});
