import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

import { AUDIO_DWELL_MS, BrowseAudioAudition, type NowPlaying } from './browse-audio-audition';

/**
 * Unit coverage for the audition state machine the three Browse surfaces share
 * (issue #3436): the dwell debounce, the stale-event guard, the stop edges and
 * the `dwellMs: 0` variant the bin popup uses.
 *
 * **jsdom has no media pipeline**, so what is asserted here is the machine's
 * bookkeeping — which emissions happen, on which edges, carrying what — never
 * that sound actually came out. The parts that need a real pipeline (the rAF
 * playhead sweep advancing a real `progress`, the buffering tri-state driven by
 * real `waiting`/`playing` events, the clip window enforced against a real
 * clock, and playback from an element that is never mounted) were verified in
 * chromium against a served WAV; see the issue #3436 PR for that transcript.
 */
describe('BrowseAudioAudition', () => {
  let emitted: (NowPlaying | null)[];

  function make(dwellMs?: number): BrowseAudioAudition {
    return new BrowseAudioAudition({
      mediaUrl: (path) => `/ctx${path}`,
      lookup: () => undefined,
      ensureLoaded: () => {},
      emit: (state) => emitted.push(state),
      ...(dwellMs === undefined ? {} : { dwellMs }),
    });
  }

  beforeEach(() => {
    emitted = [];
    vi.useFakeTimers();
    // jsdom implements neither; the machine only needs them not to throw.
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('waits out the dwell before the first emission', () => {
    const a = make();
    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS - 1);
    expect(emitted).toEqual([]);

    vi.advanceTimersByTime(1);
    expect(emitted).toEqual([
      { mediaId: 7, waveUrl: '/ctx/api/medias/7/thumbnail', loading: true, progress: null },
    ]);
    expect(a.element.src).toContain('/ctx/api/medias/7/audio');
  });

  it('re-arms the dwell on each hover, so only the settled item plays', () => {
    const a = make();
    a.hover(1);
    vi.advanceTimersByTime(AUDIO_DWELL_MS - 20);
    a.hover(2);
    vi.advanceTimersByTime(AUDIO_DWELL_MS - 20);
    a.hover(3);
    expect(emitted).toEqual([]);

    vi.advanceTimersByTime(AUDIO_DWELL_MS);
    expect(emitted.map((e) => e?.mediaId)).toEqual([3]);
    expect(a.isTargeting(3)).toBe(true);
    expect(a.isTargeting(1)).toBe(false);
  });

  it('treats a re-hover of the auditioning clip as a no-op', () => {
    const a = make();
    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS);
    const after = emitted.length;

    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS * 2);
    expect(emitted.length).toBe(after);
  });

  it('reports a media whose dwell is armed but not yet playing as targeted', () => {
    const a = make();
    a.hover(7);
    expect(a.isTargeting(7)).toBe(true);
    expect(emitted).toEqual([]);
  });

  it('flips the loading flag from the element buffering events', () => {
    const a = make();
    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS);

    a.element.dispatchEvent(new Event('playing'));
    expect(emitted.at(-1)?.loading).toBe(false);

    a.element.dispatchEvent(new Event('waiting'));
    expect(emitted.at(-1)?.loading).toBe(true);

    a.element.dispatchEvent(new Event('canplay'));
    expect(emitted.at(-1)?.loading).toBe(false);
  });

  it('emits null exactly once on stop, and stays silent when stopped again', () => {
    const a = make();
    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS);
    emitted.length = 0;

    a.stop();
    expect(emitted).toEqual([null]);

    a.stop();
    expect(emitted).toEqual([null]);
  });

  it('emits nothing when stopping an audition that never started', () => {
    const a = make();
    a.hover(7);
    a.stop();
    vi.advanceTimersByTime(AUDIO_DWELL_MS * 2);
    expect(emitted).toEqual([]);
    expect(a.isTargeting(7)).toBe(false);
  });

  it('ignores buffering events that arrive after a stop', () => {
    const a = make();
    a.hover(7);
    vi.advanceTimersByTime(AUDIO_DWELL_MS);
    a.stop();
    emitted.length = 0;

    a.element.dispatchEvent(new Event('playing'));
    a.element.dispatchEvent(new Event('waiting'));
    expect(emitted).toEqual([]);
  });

  it('plays synchronously when the dwell is zero (the bin popup)', () => {
    const a = make(0);
    a.hover(42);
    expect(emitted).toEqual([
      { mediaId: 42, waveUrl: '/ctx/api/medias/42/thumbnail', loading: true, progress: null },
    ]);
  });

  it('applies the volume to the element, and to a clip started later', () => {
    const a = make(0);
    a.setVolume(0.25);
    expect(a.element.volume).toBe(0.25);

    a.hover(7);
    expect(a.element.volume).toBe(0.25);
  });

  it('emits the final null on destroy', () => {
    const a = make(0);
    a.hover(7);
    emitted.length = 0;

    a.destroy();
    expect(emitted).toEqual([null]);
    expect(a.element.paused).toBe(true);
  });

  it('keeps its element out of the document', () => {
    const a = make(0);
    a.hover(7);
    expect(a.element.isConnected).toBe(false);
    expect(document.querySelector('audio')).toBeNull();
  });
});
