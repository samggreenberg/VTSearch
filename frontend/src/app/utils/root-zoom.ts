/**
 * The app applies an interface-scale factor as `html { zoom: <n> }`, which
 * renders the whole page `n×` larger on screen without changing any layout-px
 * value in CSS. That split is the source of a recurring class of bug: pointer
 * coordinates (`MouseEvent.clientX/Y`) and `getBoundingClientRect()` boxes come
 * back in *visual* px (already multiplied by the zoom), while CSS widths, canvas
 * backing-store sizes, and other layout dimensions are in *layout* px (the
 * pre-zoom unit). Mixing the two makes a dragged divider or resize handle ride
 * ~`(zoom-1)` away from the pointer.
 *
 * `readRootZoom()` is the single place that reads the current factor so callers
 * can convert between the two spaces (divide a visual-px delta by it to get
 * layout px; multiply a layout-px size by it to get the on-screen size). Falls
 * back to 1 where the value is unset, unparseable, or the DOM is unavailable
 * (SSR, jsdom tests).
 */
export function readRootZoom(): number {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') {
    return 1;
  }
  return parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
}
