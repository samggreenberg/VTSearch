import { describe, expect, it } from 'vitest';

import { applyClipWindow, clearClipWindow, clipProgress } from './clip-window';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';

/**
 * Unit coverage for the hover-preview clip-window helper (VTSearch Bug 1): the
 * two hover players (browse canvas + bin popup) must play only a windowed
 * clip's [clip_start, clip_end] rather than the whole archive member the server
 * serves. Driving the element handlers directly keeps this deterministic —
 * jsdom's HTMLMediaElement never fires real loadedmetadata/timeupdate.
 */
describe('applyClipWindow', () => {
  function media(partial: Partial<MediaBatchResponse>): MediaBatchResponse {
    return {
      id: 1,
      media_type: 'audio',
      filename: 'clip.wav',
      md5: 'x',
      custom_metadata: {},
      ...partial,
    } as MediaBatchResponse;
  }

  it('seeks to clip_start and disables native loop for a windowed clip', () => {
    const el = document.createElement('audio');
    el.loop = true;
    applyClipWindow(el, () => media({ clip_start: 40, clip_end: 50 }));

    el.onloadedmetadata!(new Event('loadedmetadata'));

    expect(el.currentTime).toBe(40);
    expect(el.loop).toBe(false);
  });

  it('loops within the window: resets to clip_start once currentTime reaches clip_end', () => {
    const el = document.createElement('audio');
    applyClipWindow(el, () => media({ clip_start: 40, clip_end: 50 }));
    el.onloadedmetadata!(new Event('loadedmetadata'));

    el.currentTime = 50; // reached the end of the window
    el.ontimeupdate!(new Event('timeupdate'));
    expect(el.currentTime).toBe(40);

    el.currentTime = 45; // inside the window — left alone
    el.ontimeupdate!(new Event('timeupdate'));
    expect(el.currentTime).toBe(45);
  });

  it('snaps back into the window when currentTime falls before clip_start', () => {
    const el = document.createElement('audio');
    applyClipWindow(el, () => media({ clip_start: 40, clip_end: 50 }));
    el.onloadedmetadata!(new Event('loadedmetadata'));

    el.currentTime = 10; // before the window
    el.ontimeupdate!(new Event('timeupdate'));
    expect(el.currentTime).toBe(40);
  });

  it('uses native full-file looping for non-windowed media (no clip_start)', () => {
    const el = document.createElement('audio');
    applyClipWindow(el, () => media({}));

    el.onloadedmetadata!(new Event('loadedmetadata'));
    expect(el.loop).toBe(true);
    expect(el.currentTime).toBe(0);

    // timeupdate is a no-op without a clip window — playback position is left be.
    el.currentTime = 5;
    el.ontimeupdate!(new Event('timeupdate'));
    expect(el.currentTime).toBe(5);
  });

  it('reads the clip lazily, so extents that hydrate after wiring still take effect', () => {
    const el = document.createElement('audio');
    let current: MediaBatchResponse | undefined; // not yet hydrated
    applyClipWindow(el, () => current);

    // loadedmetadata before hydration: nothing to seek to, native loop stands.
    el.onloadedmetadata!(new Event('loadedmetadata'));
    expect(el.loop).toBe(true);

    // Metadata lands; the timeupdate handler now enforces the window.
    current = media({ clip_start: 40, clip_end: 50 });
    el.currentTime = 55;
    el.ontimeupdate!(new Event('timeupdate'));
    expect(el.currentTime).toBe(40);
  });

  it('clearClipWindow detaches both handlers', () => {
    const el = document.createElement('audio');
    applyClipWindow(el, () => media({ clip_start: 40, clip_end: 50 }));
    clearClipWindow(el);

    expect(el.onloadedmetadata).toBeNull();
    expect(el.ontimeupdate).toBeNull();
  });
});

/**
 * Unit coverage for the now-playing playhead fraction: the sweeping line on the
 * VTSBrowse Now-Playing waveform maps playback position across the same window
 * the thumbnail PNG depicts (whole file for a plain clip, [clip_start, clip_end]
 * for a windowed archive member). jsdom leaves `duration` as NaN, so each test
 * stamps a finite value onto the element.
 */
describe('clipProgress', () => {
  function media(partial: Partial<MediaBatchResponse>): MediaBatchResponse {
    return {
      id: 1,
      media_type: 'audio',
      filename: 'clip.wav',
      md5: 'x',
      custom_metadata: {},
      ...partial,
    } as MediaBatchResponse;
  }

  function audioWithDuration(duration: number): HTMLAudioElement {
    const el = document.createElement('audio');
    Object.defineProperty(el, 'duration', { value: duration, configurable: true });
    return el;
  }

  it('returns null before a finite, positive duration is known', () => {
    const el = document.createElement('audio'); // jsdom duration is NaN
    expect(clipProgress(el, () => undefined)).toBeNull();

    const zero = audioWithDuration(0);
    expect(clipProgress(zero, () => undefined)).toBeNull();
  });

  it('maps currentTime across the whole file for a non-windowed clip', () => {
    const el = audioWithDuration(100);
    el.currentTime = 25;
    expect(clipProgress(el, () => media({}))).toBeCloseTo(0.25);
  });

  it('maps currentTime across the [clip_start, clip_end] window for a windowed clip', () => {
    const el = audioWithDuration(100);
    el.currentTime = 45; // 5s into a [40, 50] window → halfway
    expect(clipProgress(el, () => media({ clip_start: 40, clip_end: 50 }))).toBeCloseTo(0.5);
  });

  it('falls back to the duration as the window end when only clip_start is set', () => {
    const el = audioWithDuration(100);
    el.currentTime = 70; // 10s into a [50, 100] span → 0.2
    expect(clipProgress(el, () => media({ clip_start: 50 }))).toBeCloseTo(0.2);
  });

  it('clamps to [0, 1] when currentTime strays outside the window', () => {
    const el = audioWithDuration(100);
    el.currentTime = 30; // before a [40, 50] window
    expect(clipProgress(el, () => media({ clip_start: 40, clip_end: 50 }))).toBe(0);
    el.currentTime = 60; // past the window end
    expect(clipProgress(el, () => media({ clip_start: 40, clip_end: 50 }))).toBe(1);
  });

  it('returns null for a degenerate zero-length window', () => {
    const el = audioWithDuration(100);
    el.currentTime = 40;
    expect(clipProgress(el, () => media({ clip_start: 40, clip_end: 40 }))).toBeNull();
  });
});
