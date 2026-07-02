/**
 * CSS class added to `<html>` by `SettingsStateService` when the "Show
 * Animations" setting is "Hide". Kept in sync with the selector in
 * `styles.scss` that mirrors the OS reduce-motion blanket rule.
 */
export const ANIMATIONS_OFF_CLASS = 'animations-off';

/**
 * CSS class added to `<html>` by `SettingsStateService` when the "Show
 * Animations" setting is "Show". It forces motion on even when the OS asks
 * for reduced motion: `styles.scss` exempts `html.animations-on` from the
 * `prefers-reduced-motion` blanket rule, and `prefersReducedMotion()` reports
 * `false` while it is present.
 */
export const ANIMATIONS_ON_CLASS = 'animations-on';

/** The media query both the CSS blanket rule and the JS gates key off. */
export const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/**
 * Returns true when the browser/OS itself is suppressing motion via the
 * `prefers-reduced-motion: reduce` media query — independent of the app's
 * "Show Animations" setting. This is the gate that *overrides* the in-app
 * toggle: when it matches, motion stays off even with "Show Animations" on,
 * which is why the toggle appears to do nothing. The Settings → Appearance
 * status line reports this so a user whose animations vanished can see the
 * block is coming from their browser/OS, not from VTSearch.
 * Falls back to false where matchMedia is unavailable (SSR, jsdom tests).
 */
export function browserPrefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia(REDUCED_MOTION_QUERY).matches;
}

/**
 * Subscribe to changes in the browser/OS `prefers-reduced-motion` preference.
 * Invokes `onChange` with the new value whenever the user flips their OS or
 * browser reduce-motion setting while the app is open, so live UI (the Settings
 * status line) updates without a reload. Returns a teardown that removes the
 * listener; a no-op where matchMedia is unavailable.
 */
export function onBrowserReducedMotionChange(
  onChange: (reduced: boolean) => void,
): () => void {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return () => {};
  }
  const mql = window.matchMedia(REDUCED_MOTION_QUERY);
  const handler = (event: MediaQueryListEvent) => onChange(event.matches);
  mql.addEventListener('change', handler);
  return () => mql.removeEventListener('change', handler);
}

/**
 * Returns true when motion should be suppressed. The app's "Show Animations"
 * setting wins over the OS: "Hide" (the `animations-off` class on `<html>`)
 * always suppresses, "Show" (the `animations-on` class) always allows motion
 * even against an OS reduce-motion request, and "OS Setting" (neither class)
 * defers to `prefers-reduced-motion`. Callers that drive motion from JS (swipe
 * transition, smooth scrolling, projection-browser zoom tweens) gate on this so
 * the single setting turns them all off together. Falls back to the OS
 * preference in non-browser contexts (SSR, tests without matchMedia).
 */
export function prefersReducedMotion(): boolean {
  if (typeof document !== 'undefined') {
    const classes = document.documentElement.classList;
    if (classes.contains(ANIMATIONS_OFF_CLASS)) {
      return true;
    }
    if (classes.contains(ANIMATIONS_ON_CLASS)) {
      return false;
    }
  }
  return browserPrefersReducedMotion();
}
