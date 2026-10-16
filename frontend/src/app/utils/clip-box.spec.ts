import { describe, expect, it } from 'vitest';

import { clampClipBox, clipBoxViewBox } from './clip-box';

/**
 * The player's half of a video unit's letterbox crop (#4993). The expected
 * boxes follow ``clamp_clip_box`` in ``vtscore/media/video/crop.py``: the
 * player has to frame the same picture the embedder and the thumbnail did.
 */
describe('clipBoxViewBox', () => {
  it('crops a box smaller than the frame, inset from each edge', () => {
    // 640×480 frame, picture at x 100..580, y 120..420: top, right, bottom, left.
    expect(clipBoxViewBox([100, 120, 580, 420], 640, 480)).toBe('inset(120px 60px 60px 100px)');
  });

  it('crops a letterbox-only box on the vertical edges alone', () => {
    expect(clipBoxViewBox([0, 60, 640, 420], 640, 480)).toBe('inset(60px 0px 60px 0px)');
  });

  it('gives no style for a box equal to the frame', () => {
    expect(clipBoxViewBox([0, 0, 640, 480], 640, 480)).toBeNull();
  });

  it('gives no style for a missing box', () => {
    expect(clipBoxViewBox(undefined, 640, 480)).toBeNull();
    expect(clipBoxViewBox(null, 640, 480)).toBeNull();
  });

  it('gives no style for a malformed box', () => {
    expect(clipBoxViewBox([], 640, 480)).toBeNull();
    expect(clipBoxViewBox([0, 60, 640], 640, 480)).toBeNull();
    expect(clipBoxViewBox([0, 60, 640, 420, 1], 640, 480)).toBeNull();
    expect(clipBoxViewBox([0, Number.NaN, 640, 420], 640, 480)).toBeNull();
    expect(clipBoxViewBox([0, 60, Number.POSITIVE_INFINITY, 420], 640, 480)).toBeNull();
    // A null coordinate can ride the wire inside a List(Float) field.
    expect(clipBoxViewBox([0, null, 640, 420] as unknown as number[], 640, 480)).toBeNull();
  });

  it('gives no style for a box that is degenerate once clamped', () => {
    expect(clipBoxViewBox([100, 100, 100, 200], 640, 480)).toBeNull();
    expect(clipBoxViewBox([200, 100, 100, 200], 640, 480)).toBeNull();
    // Wholly outside the frame: clamps to a zero-width sliver on the right edge.
    expect(clipBoxViewBox([700, 0, 800, 480], 640, 480)).toBeNull();
  });

  it('gives no style before the frame size is known', () => {
    // videoWidth / videoHeight read 0 until (loadedmetadata).
    expect(clipBoxViewBox([0, 60, 640, 420], 0, 0)).toBeNull();
  });

  it('clamps a box past the frame to the frame', () => {
    expect(clampClipBox([-10, 40, 700, 440], 640, 480)).toEqual({ x0: 0, y0: 40, x1: 640, y1: 440 });
    expect(clipBoxViewBox([-10, 40, 700, 440], 640, 480)).toBe('inset(40px 0px 40px 0px)');
    // Past the frame on every side clamps to the whole frame, which is no crop.
    expect(clipBoxViewBox([-10, -5, 700, 500], 640, 480)).toBeNull();
  });

  it('truncates fractional coordinates toward zero, as int(float(v)) does', () => {
    expect(clampClipBox([10.9, 20.5, 600.7, 460.2], 640, 480)).toEqual({ x0: 10, y0: 20, x1: 600, y1: 460 });
    expect(clampClipBox([-0.9, 30, 640, 450], 640, 480)).toEqual({ x0: 0, y0: 30, x1: 640, y1: 450 });
  });
});
