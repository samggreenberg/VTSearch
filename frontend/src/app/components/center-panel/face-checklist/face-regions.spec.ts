import { FaceRect, FaceShape, faceShape } from '../image-viewer/face-link';
import { FaceRegion, landmarkTurn, regionViews, sharedViews } from './face-regions';

/**
 * Where the checklist's rows zoom to (#5647): each part's rectangle of face
 * space, placed from the five landmarks, and which of them two faces share.
 */
describe('face-regions', () => {
  type Point = readonly [number, number];
  const ROOMY: FaceRect = { x0: -2.5, y0: -2.5, x1: 2.5, y1: 3 };

  /** A face looking at the camera: eyes level either side of the origin, mouth one unit down. */
  function face(bounds: FaceRect = ROOMY, noseX = 0): FaceShape {
    return {
      points: [
        [-0.5, 0],
        [0.5, 0],
        [noseX, 0.55],
        [-0.4, 1],
        [0.4, 1],
      ],
      bounds,
    };
  }

  function holds(rect: FaceRect, [x, y]: Point): boolean {
    return x > rect.x0 && x < rect.x1 && y > rect.y0 && y < rect.y1;
  }

  function only(shape: FaceShape, region: FaceRegion): FaceRect {
    const views = regionViews(shape, region);
    expect(views.length).toBe(1);
    return views[0]!;
  }

  it('frames each part round the landmarks it is placed from', () => {
    const frontal = face();
    const [leftEye, rightEye, nose, leftMouth, rightMouth] = frontal.points;

    const eyes = only(frontal, 'eyes');
    expect(holds(eyes, leftEye) && holds(eyes, rightEye)).toBe(true);
    // Brows above the eyes; the mouth is not part of it.
    expect(holds(eyes, [0, -0.4])).toBe(true);
    expect(holds(eyes, leftMouth)).toBe(false);

    const noseView = only(frontal, 'nose');
    // Tip, and the bridge between the eyes.
    expect(holds(noseView, nose) && holds(noseView, [0, 0])).toBe(true);
    expect(holds(noseView, leftMouth)).toBe(false);

    const lower = only(frontal, 'lower');
    // Mouth and, well under it, the chin.
    expect(holds(lower, leftMouth) && holds(lower, rightMouth) && holds(lower, [0, 1.7])).toBe(true);
    expect(holds(lower, leftEye)).toBe(false);

    const marks = only(frontal, 'marks');
    for (const p of frontal.points) expect(holds(marks, p)).toBe(true);
    // Forehead to chin.
    expect(holds(marks, [0, -1]) && holds(marks, [0, 1.7])).toBe(true);
  });

  it('gives the ears as two views, either side of a face that looks ahead', () => {
    const [left, right] = regionViews(face(), 'ears');
    expect(left!.x1).toBeLessThan(-0.5);
    expect(right!.x0).toBeGreaterThan(0.5);
    // Mirror images of each other.
    expect(left!.x0).toBeCloseTo(-right!.x1, 9);
    expect(left!.x1).toBeCloseTo(-right!.x0, 9);
  });

  it('drops the ear a turned head hides, and moves the other out', () => {
    // The nose well to the image's left of the eyes-to-mouth line: turned that way.
    const [left, right] = regionViews(face(ROOMY, -0.3), 'ears');
    expect(left).toBeNull();
    const [, ahead] = regionViews(face(), 'ears');
    expect(right!.x0).toBeGreaterThan(ahead!.x0);
    // And the other way round.
    const [left2, right2] = regionViews(face(ROOMY, 0.3), 'ears');
    expect(right2).toBeNull();
    expect(left2).not.toBeNull();
  });

  it('shares one view holding the part on both faces', () => {
    const a = face();
    // The same face a little to one side of its eyes: its nose view sits further left.
    const b = face(ROOMY, -0.1);
    const [shared] = sharedViews(a, b, 'nose');
    const noseA = only(a, 'nose');
    const noseB = only(b, 'nose');
    expect(shared.x0).toBe(Math.min(noseA.x0, noseB.x0));
    expect(shared.x1).toBe(Math.max(noseA.x1, noseB.x1));
    expect(sharedViews(a, a, 'ears').length).toBe(2);
  });

  it('leaves out a view either image holds less than half of', () => {
    // A crop cut close at the cheeks: no room for either ear.
    const tight = face({ x0: -1, y0: -2.5, x1: 1, y1: 3 });
    expect(sharedViews(face(), tight, 'ears')).toEqual([]);
    expect(sharedViews(tight, face(), 'ears')).toEqual([]);
    // The parts in the middle of the face are still on it.
    expect(sharedViews(face(), tight, 'nose').length).toBe(1);
    // A crop that ends under the mouth has no chin.
    const chinless = face({ x0: -2.5, y0: -2.5, x1: 2.5, y1: 1.1 });
    expect(sharedViews(face(), chinless, 'lower')).toEqual([]);
  });

  it('shares only the ear both faces show', () => {
    const turned = face(ROOMY, -0.3);
    const shared = sharedViews(face(), turned, 'ears');
    expect(shared.length).toBe(1);
    // The right-hand one, wide enough for both faces' ears.
    expect(shared[0].x0).toBeGreaterThan(0.5);
  });
});

/**
 * Which way a head is turned, from its five points alone (#5670): what the
 * reference pane orders the Good pile by, so the reference shows the ear the
 * candidate does.
 */
describe('landmarkTurn', () => {
  /**
   * The landmarks, as crop fractions, of a head turned by the angle whose sine
   * is *turn*, by the head regionViews reads: the nose stands 0.45 face units
   * out of the face and the eyes are 0.9 apart, in a crop *w* × *h* face units.
   */
  function head(turn: number, w = 5, h = 5): number[][] {
    const across = Math.sqrt(1 - turn * turn);
    const face: [number, number][] = [
      [-0.45 * across, 0],
      [0.45 * across, 0],
      [0.45 * turn, 0.55],
      [-0.4 * across, 1],
      [0.4 * across, 1],
    ];
    // The eye midpoint a little above the crop's middle.
    return face.map(([x, y]) => [0.5 + x / w, 0.4 + y / h]);
  }

  it('reads a head that looks ahead as no turn', () => {
    expect(landmarkTurn(head(0))).toBeCloseTo(0, 9);
  });

  it('reads the turn of the head regionViews reads, with its sign', () => {
    for (const turn of [-0.6, -0.25, 0.1, 0.4]) {
      const landmarks = head(turn);
      expect(landmarkTurn(landmarks)).toBeCloseTo(turn, 9);
      // The side regionViews hides an ear on: a nose to the image's left is a
      // negative turn, and past the cut that hides the left-hand ear.
      const shape = faceShape(landmarks, 1, 1)!;
      const [left, right] = regionViews(shape, 'ears');
      if (turn < -0.5) expect(left).toBeNull();
      if (turn > -0.5) expect(left).not.toBeNull();
      expect(right).not.toBeNull();
    }
  });

  it('reads the same turn off a crop of any shape', () => {
    const square = landmarkTurn(head(0.3))!;
    // Cut short at a photo's edge, or tall and thin.
    expect(landmarkTurn(head(0.3, 3, 5))).toBeCloseTo(square, 9);
    expect(landmarkTurn(head(0.3, 5, 8))).toBeCloseTo(square, 9);
  });

  it('caps a head seen side on, as regionViews does', () => {
    expect(landmarkTurn(head(0.99))).toBe(0.9);
    expect(landmarkTurn(head(-1))).toBe(-0.9);
  });

  it('has no turn to read without five usable points', () => {
    expect(landmarkTurn(null)).toBeNull();
    expect(landmarkTurn(undefined)).toBeNull();
    expect(landmarkTurn(head(0).slice(0, 4))).toBeNull();
    expect(landmarkTurn([[0.4, 0.4], [0.6, 0.4], [0.5, NaN], [0.45, 0.6], [0.55, 0.6]])).toBeNull();
  });
});
