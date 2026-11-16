/**
 * The blur radius for a blurred image, as a share of its rendered size (#5689).
 *
 * One fixed radius cannot serve every site: what hides a full-screen image
 * barely softens it at 40px, and what hides a 40px chip leaves a full-screen
 * image recognisable. So the radius is a fraction of the image's shorter side,
 * which blurs every site to about the same degree, clamped so a tiny chip keeps
 * enough blur to hide it and a huge viewer does not pay for a radius no one
 * needs.
 */

/** Share of the image's shorter side the blur's radius takes. */
export const BLUR_RADIUS_FRACTION = 0.08;
/** The least radius, in CSS pixels: what a chip of a few dozen pixels gets. */
export const BLUR_RADIUS_MIN_PX = 4;
/** The greatest radius, in CSS pixels: reached at about 750px. */
export const BLUR_RADIUS_MAX_PX = 60;

/**
 * The radius, in whole CSS pixels, for an image drawn *width* × *height*. A box
 * with no size yet (not laid out, or hidden) gets the least radius; it is
 * measured again once it has one.
 */
export function blurRadiusFor(width: number, height: number): number {
  const side = Math.min(width, height);
  if (!(side > 0)) return BLUR_RADIUS_MIN_PX;
  return Math.min(BLUR_RADIUS_MAX_PX, Math.max(BLUR_RADIUS_MIN_PX, Math.round(side * BLUR_RADIUS_FRACTION)));
}
