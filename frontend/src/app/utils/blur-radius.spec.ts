import { BLUR_RADIUS_MAX_PX, BLUR_RADIUS_MIN_PX, blurRadiusFor } from './blur-radius';

describe('blurRadiusFor', () => {
  it('scales with the shorter side, so a wide strip is blurred by its height', () => {
    expect(blurRadiusFor(200, 200)).toBe(16);
    expect(blurRadiusFor(1000, 200)).toBe(16);
    expect(blurRadiusFor(200, 1000)).toBe(16);
  });

  it('keeps a small chip blurred enough to hide it', () => {
    expect(blurRadiusFor(40, 40)).toBe(BLUR_RADIUS_MIN_PX);
    expect(blurRadiusFor(10, 10)).toBe(BLUR_RADIUS_MIN_PX);
  });

  it('caps the radius of a large image', () => {
    expect(blurRadiusFor(2000, 1500)).toBe(BLUR_RADIUS_MAX_PX);
  });

  it('gives a box with no size yet the least radius', () => {
    expect(blurRadiusFor(0, 0)).toBe(BLUR_RADIUS_MIN_PX);
    expect(blurRadiusFor(300, 0)).toBe(BLUR_RADIUS_MIN_PX);
    expect(blurRadiusFor(Number.NaN, 300)).toBe(BLUR_RADIUS_MIN_PX);
  });

  it('answers whole pixels', () => {
    expect(Number.isInteger(blurRadiusFor(123, 457))).toBe(true);
  });
});
