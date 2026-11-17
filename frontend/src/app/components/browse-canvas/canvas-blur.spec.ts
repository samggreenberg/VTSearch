import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BLUR_RADIUS_MIN_PX, blurRadiusFor } from '../../utils/blur-radius';
import { BlurPeek, PEEK_DELAY_MS, PEEK_FADE_MS, cellBlurFraction, stillPaint } from './canvas-blur';

/**
 * The Browse canvas's blur rules (#5689): how hard a cell is blurred, what a
 * cell paints with, and the hover peek's clock. The canvas itself only wires
 * these to its draw loop, so this is where the behaviour is pinned.
 */

describe('cellBlurFraction', () => {
  it('gives the smallest cell the default size draws the DOM’s least radius', () => {
    // Level selection keeps a cell within √2 of the target radius.
    const smallest = (2 * 28) / Math.SQRT2;
    expect(cellBlurFraction(28) * smallest).toBeCloseTo(BLUR_RADIUS_MIN_PX);
  });

  it('blurs every cell of the level band at least as hard as a DOM image of its size, to within its rounding', () => {
    for (const target of [16, 28, 48, 96, 200]) {
      const fraction = cellBlurFraction(target);
      for (const scale of [Math.SQRT1_2, 0.85, 1, 1.2, Math.SQRT2]) {
        const side = 2 * target * scale;
        // blurRadiusFor rounds to whole pixels, at both ends of the comparison.
        expect(fraction * side).toBeGreaterThanOrEqual(blurRadiusFor(side, side) - 1 - 1e-9);
      }
    }
  });

  it('is no blur for a cell of no size', () => {
    expect(cellBlurFraction(0)).toBe(0);
  });
});

describe('stillPaint', () => {
  const sharp = 'sharp';
  const copy = 'blurred-copy';

  it('paints a sharp item with its image, without building a copy', () => {
    const build = vi.fn(() => copy);
    expect(stillPaint(sharp, false, 0, build)).toEqual({ base: sharp, over: null, overAlpha: 0 });
    expect(build).not.toHaveBeenCalled();
  });

  it('paints a blurred item with its blurred copy', () => {
    expect(stillPaint(sharp, true, 0, () => copy)).toEqual({ base: copy, over: null, overAlpha: 0 });
  });

  it('fades the sharp image in over the copy while the peek runs', () => {
    expect(stillPaint(sharp, true, 0.4, () => copy)).toEqual({ base: copy, over: sharp, overAlpha: 0.4 });
  });

  it('paints a finished peek sharp, without building a copy', () => {
    const build = vi.fn(() => copy);
    expect(stillPaint(sharp, true, 1, build)).toEqual({ base: sharp, over: null, overAlpha: 0 });
    expect(build).not.toHaveBeenCalled();
  });

  it('paints nothing, never the sharp image, while a blurred item’s copy is not built', () => {
    expect(stillPaint(sharp, true, 0, () => null)).toBeNull();
    expect(stillPaint(sharp, true, 0.5, () => null)).toBeNull();
  });
});

describe('BlurPeek', () => {
  let repaint: ReturnType<typeof vi.fn<() => void>>;
  let peek: BlurPeek;

  beforeEach(() => {
    vi.useFakeTimers();
    repaint = vi.fn<() => void>();
    peek = new BlurPeek({ now: () => Date.now(), repaint });
  });

  afterEach(() => {
    peek.stop();
    vi.useRealTimers();
  });

  it('holds a rested cell blurred for the delay, then fades it to sharp', () => {
    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS - 1);
    expect(peek.progress('a')).toBe(0);
    expect(repaint).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(repaint).toHaveBeenCalledTimes(1);
    expect(peek.progress('a')).toBe(0);
    expect(peek.fading()).toBe(true);

    // Gradual: still mostly blurred a quarter of the way in, half way at half time.
    vi.advanceTimersByTime(PEEK_FADE_MS / 4);
    const quarter = peek.progress('a');
    expect(quarter).toBeGreaterThan(0);
    expect(quarter).toBeLessThan(0.25);
    vi.advanceTimersByTime(PEEK_FADE_MS / 4);
    expect(peek.progress('a')).toBeCloseTo(0.5);

    vi.advanceTimersByTime(PEEK_FADE_MS / 2);
    expect(peek.progress('a')).toBe(1);
    expect(peek.fading()).toBe(false);
  });

  it('carries on when told the same cell frame after frame', () => {
    for (let t = 0; t < PEEK_DELAY_MS; t += 100) {
      peek.rest('a');
      vi.advanceTimersByTime(100);
    }
    expect(repaint).toHaveBeenCalledTimes(1);
    expect(peek.fading()).toBe(true);
  });

  it('cancels the delay when the pointer leaves', () => {
    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS / 2);
    peek.rest(null);
    vi.advanceTimersByTime(PEEK_DELAY_MS * 2);
    expect(repaint).not.toHaveBeenCalled();
    expect(peek.progress('a')).toBe(0);
  });

  it('starts the delay over when the pointer moves to another cell', () => {
    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS * 0.6);
    peek.rest('b');
    vi.advanceTimersByTime(PEEK_DELAY_MS * 0.6);
    expect(repaint).not.toHaveBeenCalled();

    vi.advanceTimersByTime(PEEK_DELAY_MS * 0.4);
    expect(repaint).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(PEEK_FADE_MS);
    expect(peek.progress('b')).toBe(1);
    expect(peek.progress('a')).toBe(0);
  });

  it('snaps back to blurred at once when the hover ends mid-fade, and waits the delay again on return', () => {
    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS + PEEK_FADE_MS / 2);
    expect(peek.progress('a')).toBeGreaterThan(0);

    peek.rest(null);
    expect(peek.progress('a')).toBe(0);
    expect(peek.fading()).toBe(false);

    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS - 1);
    expect(peek.progress('a')).toBe(0);
  });

  it('turns sharp at once when the delay is up, with the fade skipped', () => {
    peek.rest('a');
    vi.advanceTimersByTime(PEEK_DELAY_MS - 1);
    expect(peek.progress('a', true)).toBe(0);
    vi.advanceTimersByTime(1);
    expect(peek.progress('a', true)).toBe(1);
    expect(peek.fading(true)).toBe(false);
  });

  it('paints nothing after it is stopped', () => {
    peek.rest('a');
    peek.stop();
    vi.advanceTimersByTime(PEEK_DELAY_MS * 2);
    expect(repaint).not.toHaveBeenCalled();
  });
});
