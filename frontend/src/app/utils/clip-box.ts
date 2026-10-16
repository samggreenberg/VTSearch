/**
 * A video unit's ``clip_box`` — the pixel region of each frame a letterbox
 * cleaner kept — turned into the CSS that shows only that region in the player.
 *
 * The box is honoured, never baked in: the served file is the parent's untouched
 * bytes, and the server crops each frame to the box when it embeds the unit or
 * renders its grid thumbnail (``vtscore/media/video/crop.py``). The player does
 * the same with ``object-view-box``, which crops the picture of a replaced
 * element *and* makes the box its natural size, so the ``<video>`` lays out at
 * the cropped aspect and its native control bar runs along the cropped
 * picture's bottom edge.
 */

/** A crop box inside its frame, in source pixels: ``[x0, x1)`` × ``[y0, y1)``. */
export interface ClipBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Return *raw* as a usable box within a *width* × *height* frame, or ``null``.
 *
 * Mirrors ``clamp_clip_box`` in ``vtscore/media/video/crop.py``, so the player
 * frames exactly the picture the embedder and the thumbnail saw: each
 * coordinate is truncated to an integer and clamped into the frame, and
 * ``null`` means "no crop applies" — the box is missing or malformed, it is
 * degenerate once clamped, or it already covers the whole frame.
 */
export function clampClipBox(raw: readonly number[] | null | undefined, width: number, height: number): ClipBox | null {
  if (!Array.isArray(raw) || raw.length !== 4 || !raw.every((v) => Number.isFinite(v))) return null;
  // Python's int(float(v)) truncates toward zero.
  const [rx0, ry0, rx1, ry1] = raw.map((v) => Math.trunc(v));
  const clamp = (v: number, hi: number) => Math.max(0, Math.min(v, hi));
  const box = { x0: clamp(rx0, width), y0: clamp(ry0, height), x1: clamp(rx1, width), y1: clamp(ry1, height) };
  if (box.x1 - box.x0 < 1 || box.y1 - box.y0 < 1) return null;
  if (box.x0 === 0 && box.y0 === 0 && box.x1 === width && box.y1 === height) return null;
  return box;
}

/**
 * The ``object-view-box`` value that crops a *width* × *height* video to *raw*,
 * or ``null`` when {@link clampClipBox} finds no crop to apply.
 */
export function clipBoxViewBox(raw: readonly number[] | null | undefined, width: number, height: number): string | null {
  const box = clampClipBox(raw, width, height);
  if (!box) return null;
  return `inset(${box.y0}px ${width - box.x1}px ${height - box.y1}px ${box.x0}px)`;
}
