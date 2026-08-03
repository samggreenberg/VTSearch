import {
  formatEta,
  formatProgressHeader,
  isProgressIndeterminate,
  progressBarState,
} from './format-progress';

describe('progressBarState', () => {
  it('prefers the whole-job overall fraction', () => {
    const state = progressBarState({ overall: 0.4, current: 5, total: 10 });
    expect(state).toEqual({ value: 0.4, max: 1, indeterminate: false, pulsing: false });
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
      pulsing: true,
    });
  });

  it('pulses when the current phase of a whole-job bar reports no total', () => {
    // Model load: parked overall fraction, 0/0 within the phase (issue #2621).
    const state = progressBarState({
      status: 'loading',
      message: 'Loading embedding model…',
      overall: 0.35,
      current: 0,
      total: 0,
    });
    expect(state.indeterminate).toBe(false);
    expect(state.pulsing).toBe(true);
  });

  it('does not pulse while the phase reports real counts', () => {
    expect(progressBarState({ overall: 0.5, current: 3, total: 10 }).pulsing).toBe(false);
  });

  it('does not pulse once the job is complete', () => {
    expect(progressBarState({ overall: 1, current: 0, total: 0 }).pulsing).toBe(false);
  });

  it('does not pulse on an errored job', () => {
    expect(progressBarState({ overall: 0.4, total: 0, error: 'boom' }).pulsing).toBe(false);
  });

  it('leaves single-phase and indeterminate bars without a pulsing flag', () => {
    expect(progressBarState({ current: 3, total: 12 }).pulsing).toBeUndefined();
    expect(progressBarState({}).pulsing).toBeUndefined();
  });

  it('bounds the pulse with pulseTo when the slice end is known', () => {
    // The motivating case: steps weighted 50/30/20, step 2 count-less. The
    // bar parks at 0.5 and the backend says the slice ends at 0.8 — the job
    // is somewhere in between, and the bar shades exactly that span.
    const state = progressBarState({ overall: 0.5, current: 0, total: 0, overall_step_end: 0.8 });
    expect(state.pulsing).toBe(true);
    expect(state.pulseTo).toBe(0.8);
  });

  it('omits pulseTo when the phase reports real counts', () => {
    expect(
      progressBarState({ overall: 0.5, current: 3, total: 10, overall_step_end: 0.8 }).pulseTo,
    ).toBeUndefined();
  });

  it('omits pulseTo when the slice end does not extend past the fill', () => {
    expect(
      progressBarState({ overall: 0.5, current: 0, total: 0, overall_step_end: 0.5 }).pulseTo,
    ).toBeUndefined();
  });

  it('clamps pulseTo to 1', () => {
    expect(
      progressBarState({ overall: 0.5, current: 0, total: 0, overall_step_end: 1.2 }).pulseTo,
    ).toBe(1);
  });

  it('renders a whole-bar zone exactly as a plain indeterminate bar', () => {
    // A single count-less step ("Building coverage atlas…" reports 0/0 with
    // step 1 of 1) knows nothing about anywhere, which is precisely what an
    // indeterminate bar says. It must not animate differently from an
    // identical job that happens to declare no step structure at all.
    const oneStep = progressBarState({ overall: 0, current: 0, total: 0, overall_step_end: 1 });
    const noStructure = progressBarState({ current: 0, total: 0 });
    expect(oneStep).toEqual(noStructure);
    expect(oneStep.indeterminate).toBe(true);
    expect(oneStep.pulseTo).toBeUndefined();
  });

  it('still bounds a zone that starts at 0 but stops short of the end', () => {
    // Step 1 of 4 with no counts: unknown, but only within the first quarter.
    const state = progressBarState({ overall: 0, current: 0, total: 0, overall_step_end: 0.25 });
    expect(state.indeterminate).toBe(false);
    expect(state.pulseTo).toBe(0.25);
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

  it('hedges every estimate with "About"', () => {
    // The backend already snapped this to a coarse rung and will hold it there
    // (ProgressTracker._humble_eta); the wording has to match that promise
    // rather than implying a number anyone should check.
    expect(formatEta(15)).toBe('About 15 sec left');
    expect(formatEta(600)).toBe('About 10 min left');
    expect(formatEta(7200)).toBe('About 2 hr left');
  });

  it('picks the largest unit that keeps the value readable', () => {
    expect(formatEta(45)).toBe('About 45 sec left');
    expect(formatEta(90)).toBe('About 1.5 min left');
    expect(formatEta(2700)).toBe('About 45 min left');
    expect(formatEta(3600)).toBe('About 1 hr left');
    expect(formatEta(5400)).toBe('About 1.5 hr left');
    expect(formatEta(86400)).toBe('About 24 hr left');
  });

  it('renders an off-ladder value sanely instead of pretending to precision', () => {
    // Nothing on the wire should be off-ladder, but a caller passing a raw
    // estimate must not produce "About 5.516666666666667 min left".
    expect(formatEta(331)).toBe('About 5.5 min left');
    expect(formatEta(12.4)).toBe('About 12 sec left');
  });
});
