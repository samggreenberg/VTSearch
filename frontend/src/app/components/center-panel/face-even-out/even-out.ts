import { FaceLandmarks, faceAnchor, faceFrame, faceShape } from '../image-viewer/face-link';
import { skinPatches } from '../face-checklist/face-regions';

/**
 * Evening out how two faces look in the Face compare layout (#5674, #5675;
 * `docs/plans/face-vote-helper.md`, "Matching how the two faces look").
 *
 * Two photos of one person taken under different light look less alike than
 * the person is. Each pane gets one global per-channel transfer, so the two
 * faces' exposure, contrast and colour cast agree, and both are shown in
 * greyscale when either is. Everything here is display: the stored images are
 * untouched, and nothing changes what the face embedder sees.
 *
 * The standards set the limits. FISWG's *Image Processing Techniques for
 * Facial Image Comparison* (v1.0) lists "Brightness/Contrast", "Color
 * balance", "Levels", "Curves" and "Grayscale" (5.1), wants "the minimum
 * amount of image processing" (4.4), and every operation "over the whole image
 * and not in localized areas" (4.5). So the map is one curve per channel over
 * the whole picture, measured where it should be and capped so neither face
 * is pushed far from how it was taken.
 *
 * Everything is measured on the face's skin and nowhere else: small patches
 * on the forehead, the bridge of the nose and each visible cheek, placed from
 * the landmarks ({@link skinBoxes}). The wall behind a person, white or black,
 * their hair, and their eyes, brows and lips never count.
 *
 * - **Exposure** brings the skin's median luminance to {@link EXPOSURE_TARGET}
 *   by a gamma curve, which lifts a dark face without clipping its
 *   highlights: darkness hides a mole.
 * - **Contrast** brings the skin's spread, its 5th to 95th percentile (how
 *   unevenly the light falls across the face), to {@link CONTRAST_TARGET} by a
 *   stretch about the median.
 * - **White balance** moves the skin's mean colour toward
 *   {@link TYPICAL_SKIN} at its own luminance ({@link skinBalance}), a gain per
 *   channel. Read from the skin, it cannot tell a cast from a person's own
 *   colour, so it draws two people's skin toward one colour as far as its
 *   caps let it.
 *
 * Both faces move, each toward the same fixed targets, rather than one toward
 * the other: a dark candidate is lifted too, and neither face becomes the
 * template for the other. So each pane's map is its own pixels' alone, and the
 * one thing the two panes share is the greyscale rule ({@link greyPair}).
 *
 * The arithmetic is pure, over a pixel array, so Vitest covers it without a
 * canvas; the reading of a picture's pixels is `FaceEvenOutService`'s.
 */

/** The on/off choices, one per standardization (the `face_even_out` setting). */
export interface EvenOutChoices {
  /** Bring each face's skin to a common median luminance. */
  readonly exposure: boolean;
  /** Bring each face's skin to a common spread of luminance. */
  readonly contrast: boolean;
  /** Move each face's skin toward a typical skin colour, taking a cast out. */
  readonly white_balance: boolean;
  /** Show both faces in greyscale when either is a greyscale photo. */
  readonly greyscale_either: boolean;
  /** Show both faces in greyscale, whatever they are. */
  readonly greyscale_both: boolean;
  /** Put the Good faces turned the way the candidate is first (#5670). */
  readonly turn_matched: boolean;
}

export type EvenOutChoice = keyof EvenOutChoices;

/** The defaults, which `FaceEvenOut` in `vtsearch/settings/models.py` mirrors:
 *  every standardization on, but the manual greyscale. */
export const EVEN_OUT_DEFAULTS: EvenOutChoices = {
  exposure: true,
  contrast: true,
  white_balance: true,
  greyscale_either: true,
  greyscale_both: false,
  turn_matched: true,
};

/** A stored choice set over the defaults: a key it lacks, or one that is not a
 *  boolean, reads as its default, so a standardization added later starts on
 *  its default for everyone. */
export function evenOutChoices(stored: Partial<Record<EvenOutChoice, unknown>> | null | undefined): EvenOutChoices {
  const out: Record<string, boolean> = { ...EVEN_OUT_DEFAULTS };
  for (const key of Object.keys(EVEN_OUT_DEFAULTS) as EvenOutChoice[]) {
    const value = stored?.[key];
    if (typeof value === 'boolean') out[key] = value;
  }
  return out as unknown as EvenOutChoices;
}

/*
 * The targets and the caps, all in sRGB-encoded units (0 to 1), the space the
 * pixels are read in and the filter is applied in. The targets are the median
 * face's skin, measured on the 253 face crops of the screenshot harness's
 * Celebrities in Context slice (skin luminance 0.64, spread 0.21, colour
 * 1.37 : 1 : 0.85), so a typical face is barely changed (FISWG 4.4's
 * minimum). The exposure and colour caps reach about the 10th and 90th
 * percentiles of what those crops would need. The stretch's are tighter: the
 * skin's spread runs from 0.11 to 0.50 across them, and stretching a flat face
 * twice over to match would clip its hair and its background. All were
 * checked by eye in Chromium.
 */

/** The skin's median luminance that exposure brings each face to. */
export const EXPOSURE_TARGET = 0.64;
/** The gamma curve's exponent is kept in here: under 1 lifts, over 1 darkens. */
export const GAMMA_MIN = 0.5;
export const GAMMA_MAX = 2;
/** The skin's spread, 5th to 95th percentile, that contrast brings each face to. */
export const CONTRAST_TARGET = 0.21;
/** The contrast stretch about the median is kept in here. */
export const STRETCH_MIN = 0.75;
export const STRETCH_MAX = 1.5;
/** Each channel's white-balance gain is kept in here. */
export const BALANCE_MIN = 0.83;
export const BALANCE_MAX = 1.2;
/**
 * The colour white balance moves each face's skin toward, at the skin's own
 * luminance: the median skin colour of those crops, in proportion only (its
 * scale does not matter).
 */
export const TYPICAL_SKIN: Rgb = [1.37, 1, 0.85];
/**
 * Where a face sits in a face crop, as fractions of it, in `face_landmarks`'
 * order: the median of those crops' landmarks. A face with none of its own is
 * read where these place it, since the face converter centres every crop on
 * its face; an example picture framed otherwise is read in the wrong place.
 */
export const TYPICAL_LANDMARKS: FaceLandmarks = [
  [0.34, 0.43],
  [0.65, 0.43],
  [0.5, 0.56],
  [0.37, 0.66],
  [0.64, 0.66],
];
/**
 * A picture is greyscale when the 99th percentile of its pixels' channel
 * disagreement, max(|R−G|, |G−B|), is at most this many levels of 255: a
 * black-and-white photo saved as colour JPEG carries a few levels of chroma
 * noise, most of it at edges, and a colour face has far more than this.
 */
export const GREY_TOLERANCE = 6;
/** The share of pixels the greyscale test lets disagree by more (see {@link GREY_TOLERANCE}). */
const GREY_OUTLIERS = 0.01;

/** The percentiles the skin's luminance is read at. */
const LOW_PERCENTILE = 0.05;
const HIGH_PERCENTILE = 0.95;
/**
 * The skin's colour is the mean of its pixels between these luminance
 * percentiles, so a shine, a mole or a stray hair does not tint it.
 */
const COLOUR_LOW = 0.1;
const COLOUR_HIGH = 0.9;

/** Rec. 709 luma weights, on the encoded values. */
const LUMA: readonly [number, number, number] = [0.2126, 0.7152, 0.0722];

/** A pixel with less alpha than this is not part of the picture. */
const MIN_ALPHA = 128;

export type Rgb = readonly [number, number, number];

/** Where a luminance distribution sits: its 5th percentile, median and 95th, 0 to 1. */
export interface Levels {
  readonly low: number;
  readonly median: number;
  readonly high: number;
}

/**
 * What a picture's pixels say, read once when it loads.
 *
 * `face` is the skin's luminance as taken and with {@link balance} applied,
 * so the exposure and contrast parts of the map can be fitted with white
 * balance ticked or not; null when no skin patch lands on the picture.
 */
export interface PixelMeasure {
  readonly grey: boolean;
  readonly balance: Rgb;
  readonly face: { readonly raw: Levels; readonly balanced: Levels } | null;
}

/** A rectangle of whole pixels, `x0 <= x < x1`, `y0 <= y < y1`. */
export interface PixelBox {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

/**
 * The skin patches of the face on a picture *width* × *height* pixels
 * ({@link skinPatches}: forehead, nose bridge, each visible cheek), cut to the
 * picture. Placed from *landmarks*, or from {@link TYPICAL_LANDMARKS} when
 * they are not five usable points.
 */
export function skinBoxes(landmarks: FaceLandmarks | null | undefined, width: number, height: number): readonly PixelBox[] {
  const points = faceAnchor(landmarks) ? landmarks : TYPICAL_LANDMARKS;
  const anchor = faceAnchor(points);
  const frame = anchor && faceFrame(anchor, width, height);
  const shape = frame && faceShape(points, width, height);
  if (!frame || !shape) return [];
  // Face space back to pixels: the inverse of `faceShape`'s placement.
  const px = (x: number) => x * frame.unit + frame.x + width / 2;
  const py = (y: number) => y * frame.unit + frame.y + height / 2;
  const boxes: PixelBox[] = [];
  for (const patch of skinPatches(shape)) {
    const box = {
      x0: Math.max(0, Math.floor(px(patch.x0))),
      y0: Math.max(0, Math.floor(py(patch.y0))),
      x1: Math.min(width, Math.ceil(px(patch.x1))),
      y1: Math.min(height, Math.ceil(py(patch.y1))),
    };
    if (box.x1 > box.x0 && box.y1 > box.y0) boxes.push(box);
  }
  return boxes;
}

/**
 * Read *data* (RGBA, *width* × *height*, as `getImageData` gives it): whether
 * it is greyscale, from all of it, and from the *skin* patches alone, the
 * skin's luminance and the white-balance gains its colour asks for. Nothing
 * outside the patches, the wall behind the face included, counts toward
 * either. Transparent pixels are left out.
 */
export function measurePixels(
  data: ArrayLike<number>,
  width: number,
  height: number,
  skin: readonly PixelBox[],
): PixelMeasure {
  const end = Math.min(data.length, width * height * 4);
  let count = 0;
  const disagreement = new Array<number>(256).fill(0);
  for (let i = 0; i + 3 < end; i += 4) {
    if (data[i + 3] < MIN_ALPHA) continue;
    count++;
    disagreement[Math.min(255, Math.max(Math.abs(data[i] - data[i + 1]), Math.abs(data[i + 1] - data[i + 2])))]++;
  }
  const grey = count > 0 && percentileOf(disagreement, count, 1 - GREY_OUTLIERS) <= GREY_TOLERANCE;
  const pixels = skinPixels(data, width, skin);
  if (!pixels.length) return { grey, balance: NO_BALANCE, face: null };
  // A greyscale photo has no colour to balance.
  const balance = grey ? NO_BALANCE : skinBalance(skinColour(data, pixels));
  return { grey, balance, face: skinLevels(data, pixels, balance) };
}

const NO_BALANCE: Rgb = [1, 1, 1];

/** The offsets into *data* of the opaque pixels inside *boxes*. */
function skinPixels(data: ArrayLike<number>, width: number, boxes: readonly PixelBox[]): number[] {
  const offsets: number[] = [];
  for (const box of boxes) {
    for (let y = box.y0; y < box.y1; y++) {
      for (let x = box.x0; x < box.x1; x++) {
        const i = (y * width + x) * 4;
        if (i + 3 < data.length && data[i + 3] >= MIN_ALPHA) offsets.push(i);
      }
    }
  }
  return offsets;
}

function lumaAt(data: ArrayLike<number>, i: number): number {
  return Math.round(LUMA[0] * data[i] + LUMA[1] * data[i + 1] + LUMA[2] * data[i + 2]);
}

/** The skin's colour, 0 to 1 per channel: the mean of its pixels between {@link COLOUR_LOW} and {@link COLOUR_HIGH} in luminance. */
export function skinColour(data: ArrayLike<number>, pixels: readonly number[]): Rgb {
  const histogram = new Array<number>(256).fill(0);
  for (const i of pixels) histogram[lumaAt(data, i)]++;
  const lo = percentileOf(histogram, pixels.length, COLOUR_LOW);
  const hi = percentileOf(histogram, pixels.length, COLOUR_HIGH);
  const sums = [0, 0, 0];
  let n = 0;
  for (const i of pixels) {
    const l = lumaAt(data, i);
    if (l < lo || l > hi) continue;
    sums[0] += data[i];
    sums[1] += data[i + 1];
    sums[2] += data[i + 2];
    n++;
  }
  return [sums[0] / n / 255, sums[1] / n / 255, sums[2] / n / 255];
}

/**
 * The per-channel gains that move *skin*'s colour to {@link TYPICAL_SKIN}'s
 * at *skin*'s own luminance, each kept within the caps. A channel with next to
 * nothing in it gets no gain, rather than a huge one.
 *
 * It reads the light from the skin, so it cannot tell a colour cast from a
 * person's own colour: two people's skin is drawn toward one colour, as far
 * as the caps let it. The owner's call (#5675), so that nothing behind the
 * face, a white wall or a black one, sways it; the checklist says not to
 * weigh skin tone.
 */
export function skinBalance(skin: Rgb): Rgb {
  const luma = LUMA[0] * skin[0] + LUMA[1] * skin[1] + LUMA[2] * skin[2];
  const typicalLuma = LUMA[0] * TYPICAL_SKIN[0] + LUMA[1] * TYPICAL_SKIN[1] + LUMA[2] * TYPICAL_SKIN[2];
  return skin.map((m, c) =>
    m > 1e-3 && luma > 1e-3 ? clamp((TYPICAL_SKIN[c] * luma) / (typicalLuma * m), BALANCE_MIN, BALANCE_MAX) : 1,
  ) as unknown as Rgb;
}

/** The skin's luminance levels, as taken and under *balance*. */
function skinLevels(data: ArrayLike<number>, pixels: readonly number[], balance: Rgb): { raw: Levels; balanced: Levels } {
  const raw = new Array<number>(256).fill(0);
  const balanced = new Array<number>(256).fill(0);
  const [wr, wg, wb] = [LUMA[0] * balance[0], LUMA[1] * balance[1], LUMA[2] * balance[2]];
  for (const i of pixels) {
    raw[lumaAt(data, i)]++;
    balanced[Math.min(255, Math.round(wr * data[i] + wg * data[i + 1] + wb * data[i + 2]))]++;
  }
  return { raw: levelsOf(raw, pixels.length), balanced: levelsOf(balanced, pixels.length) };
}

function levelsOf(histogram: readonly number[], count: number): Levels {
  return {
    low: percentileOf(histogram, count, LOW_PERCENTILE) / 255,
    median: percentileOf(histogram, count, 0.5) / 255,
    high: percentileOf(histogram, count, HIGH_PERCENTILE) / 255,
  };
}

/** The value under which a share *q* of a 256-bin histogram's *count* entries lie. */
function percentileOf(histogram: readonly number[], count: number, q: number): number {
  const rank = q * count;
  let seen = 0;
  for (let v = 0; v < histogram.length; v++) {
    seen += histogram[v];
    if (seen >= rank && seen > 0) return v;
  }
  return histogram.length - 1;
}

/**
 * One channel's curve, as an SVG `<feFuncR type="gamma">` and its siblings
 * draw it: `amplitude · C^exponent + offset`, clamped to 0..1 by the filter.
 */
export interface ChannelCurve {
  readonly amplitude: number;
  readonly exponent: number;
  readonly offset: number;
}

/** The three channels' curves: one pane's whole map. */
export interface ToneMap {
  readonly r: ChannelCurve;
  readonly g: ChannelCurve;
  readonly b: ChannelCurve;
}

/** A channel value through *curve*, as the filter computes it. */
export function applyCurve(curve: ChannelCurve, c: number): number {
  return clamp(curve.amplitude * Math.pow(c, curve.exponent) + curve.offset, 0, 1);
}

/**
 * The map for a picture measured as *measure*, under *choices*; null when it
 * would change nothing (every part off, or nothing to correct).
 *
 * Its three parts compose into one gamma curve per channel. With *w* the
 * channel's white-balance gain, *γ* the exposure exponent and *k* the
 * contrast stretch about the median *p* (after exposure), a value *C* goes
 * to `p + k · ((w · C)^γ − p)`, which is `k·w^γ · C^γ + (1 − k)·p`. Unticking
 * a part sets its own number to "no change" (*w* = 1, *γ* = 1, *k* = 1) and
 * leaves the others to reach their own targets.
 */
export function toneMap(measure: PixelMeasure, choices: EvenOutChoices): ToneMap | null {
  const balance = choices.white_balance ? measure.balance : NO_BALANCE;
  const levels = measure.face ? (choices.white_balance ? measure.face.balanced : measure.face.raw) : null;
  let exponent = 1;
  let stretch = 1;
  let pivot = 0;
  if (levels) {
    const median = clamp(levels.median, 0.02, 0.98);
    if (choices.exposure) exponent = clamp(Math.log(EXPOSURE_TARGET) / Math.log(median), GAMMA_MIN, GAMMA_MAX);
    if (choices.contrast) {
      const spread = Math.pow(levels.high, exponent) - Math.pow(levels.low, exponent);
      stretch = clamp(CONTRAST_TARGET / Math.max(spread, 1e-3), STRETCH_MIN, STRETCH_MAX);
      pivot = Math.pow(median, exponent);
    }
  }
  const curve = (w: number): ChannelCurve => ({
    amplitude: stretch * Math.pow(w, exponent),
    exponent,
    offset: (1 - stretch) * pivot,
  });
  const map: ToneMap = { r: curve(balance[0]), g: curve(balance[1]), b: curve(balance[2]) };
  return isIdentity(map) ? null : map;
}

function isIdentity(map: ToneMap): boolean {
  const flat = (c: ChannelCurve) =>
    Math.abs(c.amplitude - 1) < 1e-4 && Math.abs(c.exponent - 1) < 1e-4 && Math.abs(c.offset) < 1e-4;
  return flat(map.r) && flat(map.g) && flat(map.b);
}

/**
 * Whether the compare layout shows both faces in greyscale (#5674): always
 * with **Both in greyscale** ticked, and with **Greyscale when either is**
 * ticked when either picture is a greyscale photo. A face not measured yet
 * (null) is not counted as greyscale. It greys both panes or neither, never
 * one: a mixed pair makes people more likely to declare a match (Bobak,
 * Mileva & Hancock 2019).
 */
export function greyPair(candidate: boolean | null, reference: boolean | null, choices: EvenOutChoices): boolean {
  return choices.greyscale_both || (choices.greyscale_either && (candidate === true || reference === true));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
