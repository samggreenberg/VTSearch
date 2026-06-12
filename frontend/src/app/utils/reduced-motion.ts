/**
 * CSS class added to `<html>` by `SettingsStateService` when the user turns
 * the "Show Animations" setting off. Kept in sync with the selector in
 * `styles.scss` that mirrors the OS reduce-motion blanket rule.
 */
export const ANIMATIONS_OFF_CLASS = 'animations-off';

/**
 * Returns true when motion should be suppressed — either because the user (or
 * OS) asked for reduced motion, or because the app's "Show Animations" setting
 * is off (reflected as the `animations-off` class on `<html>`). Callers that
 * drive motion from JS (swipe transition, smooth scrolling, projection-browser
 * zoom tweens) gate on this so the single setting turns them all off together.
 * Falls back to false in non-browser contexts (SSR, tests without matchMedia).
 */
export function prefersReducedMotion(): boolean {
  if (
    typeof document !== 'undefined' &&
    document.documentElement.classList.contains(ANIMATIONS_OFF_CLASS)
  ) {
    return true;
  }
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
