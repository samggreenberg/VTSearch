/**
 * Coercers shared by the per-media-type settings preferences.
 *
 * `SettingsStateService.perMediaType`'s `coerce` hook is where a stored value is
 * checked before it is trusted: a settings file is hand-editable and an older
 * server can hold a shape the current app never writes, so a preference has to
 * be able to say "that isn't one of mine" and fall back. The same three checks
 * were being written out at each consumer; they live here so a fix lands once.
 *
 * Every coercer returns `undefined` to reject, which is what
 * `perMediaType` reads as "use the fallback".
 */

/** The two focus modes a panel can be in: click-to-vote or hover-to-preview. */
export type FocusMode = 'click' | 'hover';

/** Accept only the two real focus modes; anything else falls back. */
export function coerceFocusMode(raw: unknown): FocusMode | undefined {
  return raw === 'click' || raw === 'hover' ? raw : undefined;
}

/** Accept only a finite number of pixels; anything else falls back to "unset". */
export function coercePx(raw: unknown): number | undefined {
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

/**
 * Accept a non-empty string. Used for the named icon sizes, whose ladders
 * differ per surface (`XS…XL` in the panels, `XS…5XL` on the browse canvas) —
 * `iconSizeToGoalWidth` already maps an unknown name onto the default, so the
 * only value worth rejecting here is the empty one.
 */
export function coerceNonEmptyString(raw: unknown): string | undefined {
  return typeof raw === 'string' && raw !== '' ? raw : undefined;
}
