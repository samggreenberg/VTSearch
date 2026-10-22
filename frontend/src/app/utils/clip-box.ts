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
 *
 * The server measures the box on decoded frames, which are *stored* pixels;
 * the browser reports a video's natural size in *display* pixels. The two
 * differ on an anamorphic video (non-square pixels), so the box is first
 * scaled into the played frame by {@link scaleClipBox}.
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
 * Return *raw* scaled from the *frame* it was measured on to a *width* × *height* one.
 *
 * A 720×480 DVD rip with sample aspect ratio 32:27 decodes at 720×480 on the
 * server but plays at 853×480, so its box, unscaled, would crop about 15% off
 * the right of the picture. The letterbox cleaner records the frame size beside
 * the box as ``clip_box_frame``, and each axis is scaled by its own ratio. That
 * also covers a rotated video, whose decoded frames and natural size are both
 * upright. Coordinates are rounded to the nearest pixel. *raw* comes back as is
 * when it or *frame* is not usable, so a box recorded without a frame keeps the
 * stored-pixel reading and a malformed box is still rejected by the clamp.
 */
export function scaleClipBox(
  raw: readonly number[] | null | undefined,
  frame: readonly number[] | null | undefined,
  width: number,
  height: number,
): readonly number[] | null | undefined {
  if (!Array.isArray(raw) || raw.length !== 4 || !raw.every((v) => Number.isFinite(v))) return raw;
  if (!Array.isArray(frame) || frame.length !== 2) return raw;
  const [frameWidth, frameHeight] = frame;
  if (!(frameWidth > 0 && frameHeight > 0 && Number.isFinite(frameWidth) && Number.isFinite(frameHeight))) return raw;
  if (!(width > 0 && height > 0)) return raw;
  const [x0, y0, x1, y1] = raw;
  const sx = width / frameWidth;
  const sy = height / frameHeight;
  return [Math.round(x0 * sx), Math.round(y0 * sy), Math.round(x1 * sx), Math.round(y1 * sy)];
}

/**
 * The ``object-view-box`` value that crops a *width* × *height* video to *raw*,
 * or ``null`` when {@link clampClipBox} finds no crop to apply. *frame* is the
 * unit's ``clip_box_frame``, the frame size *raw* was measured on (see
 * {@link scaleClipBox}).
 */
export function clipBoxViewBox(
  raw: readonly number[] | null | undefined,
  width: number,
  height: number,
  frame?: readonly number[] | null,
): string | null {
  const box = clampClipBox(scaleClipBox(raw, frame, width, height), width, height);
  if (!box) return null;
  return `inset(${box.y0}px ${width - box.x1}px ${height - box.y1}px ${box.x0}px)`;
}
