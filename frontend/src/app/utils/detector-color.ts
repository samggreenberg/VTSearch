/**
 * Deterministic hue for a detector, derived from its display name.
 *
 * The hue (0-359) feeds the `--detector-hue` CSS custom property; theme SCSS
 * picks saturation/lightness so the color renders well in dark and light
 * modes. Renaming a detector intentionally produces a new color - the hue
 * tracks the user's mental label, not the opaque registry id.
 */
export function detectorHue(name: string): number {
  let hash = 5381;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) + hash + name.charCodeAt(i)) | 0;
  }
  return ((hash % 360) + 360) % 360;
}
