import { describe, expect, it } from 'vitest';

import { clampClipBox, clipBoxViewBox, scaleClipBox } from './clip-box';

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

/**
 * An anamorphic video (#5032): the server measures the box on decoded frames,
 * which are stored pixels, while the browser reports its natural size in
 * display pixels. A 720×480 DVD rip with sample aspect ratio 32:27 plays at
 * 853×480, so the box comes across scaled by the recorded `clip_box_frame`.
 */
describe('clipBoxViewBox on an anamorphic video', () => {
  it('scales a letterbox box measured on stored pixels to the full display width', () => {
    expect(clipBoxViewBox([0, 60, 720, 420], 853, 480, [720, 480])).toBe('inset(60px 0px 60px 0px)');
  });

  it('scales a pillarbox box on the horizontal axis', () => {
    // x 90..630 of 720 stored pixels is x 106.6..746.4 of 853 display pixels.
    expect(clipBoxViewBox([90, 0, 630, 480], 853, 480, [720, 480])).toBe('inset(0px 107px 0px 107px)');
  });

  it('scales each axis by its own ratio, which covers a quarter-turned video', () => {
    // Upright on both sides: decoded 480×720, played 480×853.
    expect(clipBoxViewBox([60, 0, 420, 720], 480, 853, [480, 720])).toBe('inset(0px 60px 0px 60px)');
    expect(scaleClipBox([0, 60, 480, 660], [480, 720], 480, 853)).toEqual([0, 71, 480, 782]);
  });

  it('leaves a box recorded without a frame in stored pixels, as before', () => {
    // The right 133 display pixels are lost: the behaviour the frame fixes.
    expect(clipBoxViewBox([0, 60, 720, 420], 853, 480)).toBe('inset(60px 133px 60px 0px)');
    expect(clipBoxViewBox([0, 60, 720, 420], 853, 480, null)).toBe('inset(60px 133px 60px 0px)');
  });

  it('is a no-op when the frame matches the played size', () => {
    expect(scaleClipBox([100, 120, 580, 420], [640, 480], 640, 480)).toEqual([100, 120, 580, 420]);
  });

  it('ignores an unusable frame', () => {
    const box = [0, 60, 720, 420];
    for (const frame of [[], [720], [720, 480, 1], [0, 480], [-720, 480], [Number.NaN, 480], [720, Number.POSITIVE_INFINITY]]) {
      expect(scaleClipBox(box, frame, 853, 480)).toBe(box);
    }
    expect(scaleClipBox(box, [720, null] as unknown as number[], 853, 480)).toBe(box);
  });

  it('does not scale before the frame size is known', () => {
    expect(scaleClipBox([0, 60, 720, 420], [720, 480], 0, 0)).toEqual([0, 60, 720, 420]);
    expect(clipBoxViewBox([0, 60, 720, 420], 0, 0, [720, 480])).toBeNull();
  });

  it('still rejects a malformed box', () => {
    // Scaling must not turn a null coordinate into a 0.
    const malformed = [0, null, 720, 420] as unknown as number[];
    expect(scaleClipBox(malformed, [720, 480], 853, 480)).toBe(malformed);
    expect(clipBoxViewBox(malformed, 853, 480, [720, 480])).toBeNull();
    expect(clipBoxViewBox([0, 60, 720], 853, 480, [720, 480])).toBeNull();
  });
});
