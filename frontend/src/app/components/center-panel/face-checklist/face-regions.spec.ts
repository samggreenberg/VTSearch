import { FaceRect, FaceShape } from '../image-viewer/face-link';
import { FaceRegion, regionViews, sharedViews } from './face-regions';

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
