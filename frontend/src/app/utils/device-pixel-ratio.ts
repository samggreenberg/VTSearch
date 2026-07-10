/**
 * Subscribe to changes in `window.devicePixelRatio`. Invokes `onChange`
 * whenever the display density changes — most commonly when the window is
 * dragged onto a different-density monitor, but also on browser zoom or an OS
 * scaling change. A `ResizeObserver` does *not* catch this on its own: dragging
 * to a Retina display leaves the element's CSS box unchanged, so nothing fires
 * and the canvas keeps rendering at the stale `dpr` (and stale thumbnail-
 * resolution tier) until the next actual resize.
 *
 * There is no dedicated "dpr changed" event, so we lean on the standard
 * `matchMedia('(resolution: Ndppx)')` trick: a query pinned to the *current*
 * dpr matches now and stops matching the instant dpr moves, firing a `change`
 * event. Because the query is pinned to one value it is one-shot, so the
 * handler re-establishes a fresh listener at the new dpr each time before
 * invoking the callback — keeping it live across any number of density changes.
 *
 * Returns a teardown that removes the active listener; a no-op where matchMedia
 * is unavailable (SSR, jsdom tests).
 */
export function onDevicePixelRatioChange(onChange: () => void): () => void {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return () => {};
  }
  let mql: MediaQueryList | null = null;
  const handler = () => {
    // Re-pin to the new dpr first so a rapid second change (e.g. dragging
    // across three monitors) is still observed, then notify the caller.
    subscribe();
    onChange();
  };
  const subscribe = () => {
    if (mql) mql.removeEventListener('change', handler);
    const dpr = window.devicePixelRatio || 1;
    mql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    mql.addEventListener('change', handler);
  };
  subscribe();
  return () => {
    if (mql) mql.removeEventListener('change', handler);
    mql = null;
  };
}
