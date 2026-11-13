import {
  FaceFrame,
  FaceLandmarks,
  PaneView,
  faceAnchor,
  faceFrame,
  faceShape,
  faceViewOf,
  paneViewOf,
  sameFaceView,
  viewOfRect,
} from './face-link';

/**
 * The face-space math behind the Face compare layout's registration (#5647):
 * two faces shown with the same view land with their eye midpoints in the same
 * place and their eye-to-mouth distances the same length, by a resize and a
 * shift alone.
 */
describe('face-link', () => {
  /** Eyes at (0.4, 0.4) and (0.6, 0.4), mouth corners at (0.45, 0.6) and (0.55, 0.6). */
  const centred: FaceLandmarks = [
    [0.4, 0.4],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.45, 0.6],
    [0.55, 0.6],
  ];
  /** A small face up in the top-left corner. */
  const cornered: FaceLandmarks = [
    [0.2, 0.2],
    [0.4, 0.2],
    [0.3, 0.25],
    [0.25, 0.3],
    [0.35, 0.3],
  ];

  /** Where image point (*u*, *v*) of a *w* × *h* image is drawn, from the pane's centre. */
  function onScreen(pane: PaneView, rotation: number, w: number, h: number, u: number, v: number): [number, number] {
    const rad = (rotation * Math.PI) / 180;
    const x = (u - 0.5) * w;
    const y = (v - 0.5) * h;
    return [
      pane.panX + pane.zoom * (x * Math.cos(rad) - y * Math.sin(rad)),
      pane.panY + pane.zoom * (x * Math.sin(rad) + y * Math.cos(rad)),
    ];
  }

  function expectPoint(actual: [number, number], expected: [number, number]): void {
    expect(actual[0]).toBeCloseTo(expected[0], 9);
    expect(actual[1]).toBeCloseTo(expected[1], 9);
  }

  describe('faceAnchor', () => {
    it('takes the midpoints of the eyes and of the mouth corners', () => {
      const anchor = faceAnchor(centred)!;
      expect(anchor.eyes[0]).toBeCloseTo(0.5, 12);
      expect(anchor.eyes[1]).toBeCloseTo(0.4, 12);
      expect(anchor.mouth[0]).toBeCloseTo(0.5, 12);
      expect(anchor.mouth[1]).toBeCloseTo(0.6, 12);
    });

    it('keeps points that lie past the crop edge', () => {
      expect(faceAnchor([[-0.1, 0.3], [0.5, 0.3], [0.2, 0.5], [0, 0.8], [0.4, 1.2]])).not.toBeNull();
    });

    it('has nothing to go on without five finite points, or with the mouth on the eyes', () => {
      expect(faceAnchor(null)).toBeNull();
      expect(faceAnchor(undefined)).toBeNull();
      expect(faceAnchor(centred.slice(0, 4))).toBeNull();
      expect(faceAnchor([...centred.slice(0, 4), [Number.NaN, 0.6]])).toBeNull();
      expect(faceAnchor([...centred.slice(0, 4), [0.5]])).toBeNull();
      expect(faceAnchor([[0.4, 0.4], [0.6, 0.4], [0.5, 0.5], [0.4, 0.4], [0.6, 0.4]])).toBeNull();
    });
  });

  describe('faceFrame', () => {
    it('measures the face in the pane, its eye midpoint from the image centre', () => {
      // A 2:1 image drawn 200 × 100.
      const frame = faceFrame(faceAnchor(centred)!, 200, 100)!;
      expect(frame.x).toBeCloseTo(0, 12);
      expect(frame.y).toBeCloseTo(-10, 12);
      expect(frame.unit).toBeCloseTo(20, 12);
    });

    it('has no frame before the image has a size', () => {
      expect(faceFrame(faceAnchor(centred)!, 0, 100)).toBeNull();
    });
  });

  describe('views', () => {
    const frame: FaceFrame = { x: -12, y: 7, unit: 15 };

    it('round-trips a pane view through face space at every quarter turn', () => {
      const pane: PaneView = { zoom: 2.5, panX: 31, panY: -18 };
      for (const rotation of [0, 90, 180, 270, -90, 450]) {
        const back = paneViewOf(frame, faceViewOf(frame, pane, rotation), rotation);
        expect(back.zoom).toBeCloseTo(pane.zoom, 12);
        expect(back.panX).toBeCloseTo(pane.panX, 9);
        expect(back.panY).toBeCloseTo(pane.panY, 9);
      }
    });

    it('keeps the view through a turn that turns the pan with the image', () => {
      // `rotateAboutCentre` turns the pan clockwise: (x, y) -> (-y, x).
      const before = faceViewOf(frame, { zoom: 2, panX: 10, panY: 4 }, 0);
      const after = faceViewOf(frame, { zoom: 2, panX: -4, panY: 10 }, 90);
      expect(after.cx).toBeCloseTo(before.cx, 12);
      expect(after.cy).toBeCloseTo(before.cy, 12);
      expect(after.scale).toBeCloseTo(before.scale, 12);
    });
  });

  describe('registration', () => {
    // The lead: a square image filling a 100 × 100 pane. The follower: a 1:2
    // image, drawn 50 × 100 in the same pane, its face small and off in a corner.
    const lead = { frame: faceFrame(faceAnchor(centred)!, 100, 100)!, w: 100, h: 100 };
    const follower = { frame: faceFrame(faceAnchor(cornered)!, 50, 100)!, w: 50, h: 100 };
    const leadPane: PaneView = { zoom: 1, panX: 0, panY: 0 };

    it('puts the eye midpoints in the same place and matches the eye-to-mouth distance', () => {
      const view = faceViewOf(lead.frame, leadPane, 0);
      const pane = paneViewOf(follower.frame, view, 0);
      // A resize, then a shift.
      expect(pane.zoom).toBeCloseTo(2, 12);
      expectPoint(onScreen(pane, 0, follower.w, follower.h, 0.3, 0.2), onScreen(leadPane, 0, lead.w, lead.h, 0.5, 0.4));
      expectPoint(onScreen(pane, 0, follower.w, follower.h, 0.3, 0.3), onScreen(leadPane, 0, lead.w, lead.h, 0.5, 0.6));
    });

    it('keeps them together as the lead zooms and pans', () => {
      const zoomed: PaneView = { zoom: 3.2, panX: -40, panY: 25 };
      const pane = paneViewOf(follower.frame, faceViewOf(lead.frame, zoomed, 0), 0);
      expectPoint(onScreen(pane, 0, follower.w, follower.h, 0.3, 0.2), onScreen(zoomed, 0, lead.w, lead.h, 0.5, 0.4));
      expectPoint(onScreen(pane, 0, follower.w, follower.h, 0.3, 0.3), onScreen(zoomed, 0, lead.w, lead.h, 0.5, 0.6));
      // Any other point of face space too: the left eye is left of the midpoint
      // by half the eye gap, which is the same share of eye-to-mouth on both faces.
      expectPoint(onScreen(pane, 0, follower.w, follower.h, 0.2, 0.2), onScreen(zoomed, 0, lead.w, lead.h, 0.4, 0.4));
    });

    it('shows the same face point in the middle of a pane turned on its own', () => {
      const zoomed: PaneView = { zoom: 2, panX: 30, panY: -10 };
      const view = faceViewOf(lead.frame, zoomed, 0);
      const pane = paneViewOf(follower.frame, view, 90);
      // The follower's pane centre is the face point the lead's is.
      const centre = faceViewOf(follower.frame, pane, 90);
      expect(centre.cx).toBeCloseTo(view.cx, 12);
      expect(centre.cy).toBeCloseTo(view.cy, 12);
      // And the eyes-to-mouth line, which runs down the lead's pane, runs
      // across the turned one at the same length.
      const eyes = onScreen(pane, 90, follower.w, follower.h, 0.3, 0.2);
      const mouth = onScreen(pane, 90, follower.w, follower.h, 0.3, 0.3);
      expect(mouth[0] - eyes[0]).toBeCloseTo(-view.scale, 9);
      expect(mouth[1] - eyes[1]).toBeCloseTo(0, 9);
    });
  });

  it('puts a face and its image in face space', () => {
    // 200 x 200: eyes 40 px apart round the centre, mouth 40 px under them.
    const shape = faceShape(centred, 200, 200)!;
    const expected = [
      [-0.5, 0],
      [0.5, 0],
      [0, 0.5],
      [-0.25, 1],
      [0.25, 1],
    ];
    shape.points.forEach((p, i) => expectPoint([p[0], p[1]], expected[i] as [number, number]));
    // The eye midpoint is 100 px from either side, 80 from the top and 120 from the bottom.
    expect(shape.bounds.x0).toBeCloseTo(-2.5, 9);
    expect(shape.bounds.y0).toBeCloseTo(-2, 9);
    expect(shape.bounds.x1).toBeCloseTo(2.5, 9);
    expect(shape.bounds.y1).toBeCloseTo(3, 9);
    expect(faceShape(null, 200, 200)).toBeNull();
    expect(faceShape(centred, 0, 200)).toBeNull();
  });

  it('fits a rectangle of the face in the pane, its centre in the middle', () => {
    const rect = { x0: -1, y0: 0, x1: 1, y1: 1 };
    // Twice as wide as tall in a square pane: the width is what fits.
    expect(viewOfRect(rect, 100, 100, 0)).toEqual({ cx: 0, cy: 0.5, scale: 50 });
    // On its side, the rectangle's height runs across the pane.
    expect(viewOfRect(rect, 100, 300, 90)).toEqual({ cx: 0, cy: 0.5, scale: 100 });
    expect(viewOfRect(rect, 100, 300, -90)).toEqual({ cx: 0, cy: 0.5, scale: 100 });
    expect(viewOfRect(rect, 100, 300, 180)).toEqual({ cx: 0, cy: 0.5, scale: 50 });
    expect(viewOfRect({ x0: 1, y0: 0, x1: 1, y1: 1 }, 100, 100, 0)).toBeNull();
    expect(viewOfRect(rect, 0, 100, 0)).toBeNull();
  });

  it('tells a view from one a little off it', () => {
    const view = { cx: 0.2, cy: -0.4, scale: 80 };
    expect(sameFaceView(view, { cx: 0.2 + 1e-9, cy: -0.4, scale: 80 * (1 + 1e-9) })).toBe(true);
    expect(sameFaceView(view, { ...view, cx: 0.21 })).toBe(false);
    expect(sameFaceView(view, { ...view, scale: 81 })).toBe(false);
  });
});
