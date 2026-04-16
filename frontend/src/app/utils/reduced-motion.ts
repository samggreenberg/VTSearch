/**
 * Returns true when the user (or OS) has asked for reduced motion.
 * Falls back to false in non-browser contexts (SSR, tests without matchMedia).
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
