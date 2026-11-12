import { FaceLandmarks, faceAnchor, faceFrame, faceShape } from '../image-viewer/face-link';
import { innerFace } from '../face-checklist/face-regions';

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
 * - **Exposure** is measured on the inner face ({@link innerFace}), where the
 *   features are: darkness hides a mole. Its median luminance is brought to
 *   {@link EXPOSURE_TARGET} by a gamma curve, which lifts a dark face without
 *   clipping its highlights.
 * - **Contrast** is the inner face's spread, its 5th to 95th percentile,
 *   brought to {@link CONTRAST_TARGET} by a stretch about the median.
 * - **White balance** is measured on the whole crop, the scene, never on the
 *   skin alone: two people's skin should keep its difference. It is a gain
 *   per channel that makes the light grey, keeping the crop's luminance, with
 *   the light's colour read from the crop's brightest pixels
 *   ({@link lightOf}).
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
  /** Bring each face's inner-face median luminance to a common target. */
  readonly exposure: boolean;
  /** Bring each face's inner-face spread to a common target. */
  readonly contrast: boolean;
  /** Take each picture's colour cast out, from its own scene. */
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
 * face's, measured on the 253 face crops of the screenshot harness's
 * Celebrities in Context slice (inner-face median 0.52, spread 0.50), so a
 * typical face is barely changed (FISWG 4.4's minimum). The caps reach about
 * the 10th and 90th percentiles of what those crops would need, and were
 * checked by eye in Chromium.
 */

/** The inner face's median luminance that exposure brings each face to. */
export const EXPOSURE_TARGET = 0.5;
/** The gamma curve's exponent is kept in here: under 1 lifts, over 1 darkens. */
export const GAMMA_MIN = 0.55;
export const GAMMA_MAX = 1.8;
/** The inner face's spread, 5th to 95th percentile, that contrast brings each face to. */
export const CONTRAST_TARGET = 0.5;
/** The contrast stretch about the median is kept in here. */
export const STRETCH_MIN = 0.75;
export const STRETCH_MAX = 1.5;
/** Each channel's white-balance gain is kept in here. */
export const BALANCE_MIN = 0.82;
export const BALANCE_MAX = 1.22;
/**
 * The share of a picture's pixels, the brightest, whose mean colour is taken
 * as the light's ({@link lightOf}).
 */
export const BRIGHT_SHARE = 0.05;
/** A pixel with a channel at or over this is clipped: its colour is not the light's. */
const CLIPPED = 250;
/**
 * A picture is greyscale when the 99th percentile of its pixels' channel
 * disagreement, max(|R−G|, |G−B|), is at most this many levels of 255: a
 * black-and-white photo saved as colour JPEG carries a few levels of chroma
 * noise, most of it at edges, and a colour face has far more than this.
 */
export const GREY_TOLERANCE = 6;
/** The share of pixels the greyscale test lets disagree by more (see {@link GREY_TOLERANCE}). */
const GREY_OUTLIERS = 0.01;

/** The percentiles the inner face's luminance is read at. */
const LOW_PERCENTILE = 0.05;
const HIGH_PERCENTILE = 0.95;

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
 * `face` is the inner face's luminance as taken and with {@link balance}
 * applied, so the exposure and contrast parts of the map can be fitted with
 * white balance ticked or not; null with no landmarks to place it from.
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
 * The inner face of *landmarks* on a picture *width* × *height* pixels, cut to
 * the picture; null with no usable landmarks, or when the box misses it.
 */
export function innerFaceBox(landmarks: FaceLandmarks | null | undefined, width: number, height: number): PixelBox | null {
  const anchor = faceAnchor(landmarks);
  const frame = anchor && faceFrame(anchor, width, height);
  const shape = frame && faceShape(landmarks, width, height);
  if (!frame || !shape) return null;
  // Face space back to pixels: the inverse of `faceShape`'s placement.
  const rect = innerFace(shape);
  const px = (x: number) => x * frame.unit + frame.x + width / 2;
  const py = (y: number) => y * frame.unit + frame.y + height / 2;
  const box = {
    x0: Math.max(0, Math.floor(px(rect.x0))),
    y0: Math.max(0, Math.floor(py(rect.y0))),
    x1: Math.min(width, Math.ceil(px(rect.x1))),
    y1: Math.min(height, Math.ceil(py(rect.y1))),
  };
  return box.x1 > box.x0 && box.y1 > box.y0 ? box : null;
}

/**
 * Read *data* (RGBA, *width* × *height*, as `getImageData` gives it): whether
 * it is greyscale, its white-balance gains, and the luminance of the inner
 * face *faceBox* when there is one. Transparent pixels are left out.
 */
export function measurePixels(
  data: ArrayLike<number>,
  width: number,
  height: number,
  faceBox: PixelBox | null,
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
  // A greyscale photo has no cast to take out, and its light reads grey anyway.
  const light = grey ? null : lightOf(data, end);
  const balance = light ? balanceFor(light) : NO_BALANCE;
  return { grey, balance, face: faceBox ? faceLevels(data, width, faceBox, balance) : null };
}

const NO_BALANCE: Rgb = [1, 1, 1];

/**
 * The colour of the light *data* (RGBA, its first *end* entries) was taken
 * under, 0 to 1 per channel: the mean of its brightest {@link BRIGHT_SHARE}
 * of pixels, leaving out clipped ones. Null with no pixel to read.
 *
 * Not the mean of every pixel (grey-world), which a face crop biases: it is
 * mostly skin, so grey-world reads every face as lit warm and turns its skin
 * grey. On the 253 crops the targets were measured on, grey-world cooled the
 * median crop by 19% in red; its brightest pixels (highlights, the whites of
 * the eyes, a shirt) left it within 7%, and still followed a cast laid over
 * every crop to within 2%. That is the "bright pixels" estimate (Joze, Drew,
 * Finlayson & Rey 2012, *The role of bright pixels in illumination
 * estimation*); a clipped pixel says the sensor's limit, not the light.
 */
export function lightOf(data: ArrayLike<number>, end: number = data.length): Rgb | null {
  const histogram = new Array<number>(256).fill(0);
  let count = 0;
  let unclipped = true;
  const level = (i: number) => Math.round(LUMA[0] * data[i] + LUMA[1] * data[i + 1] + LUMA[2] * data[i + 2]);
  const counts = (i: number) =>
    data[i + 3] >= MIN_ALPHA && (!unclipped || Math.max(data[i], data[i + 1], data[i + 2]) < CLIPPED);
  for (let pass = 0; pass < 2 && count === 0; pass++) {
    // Every pixel clipped (a blown-out picture): read them all rather than none.
    unclipped = pass === 0;
    for (let i = 0; i + 3 < end; i += 4) {
      if (!counts(i)) continue;
      histogram[level(i)]++;
      count++;
    }
  }
  if (!count) return null;
  const threshold = percentileOf(histogram, count, 1 - BRIGHT_SHARE);
  const sums = [0, 0, 0];
  let n = 0;
  for (let i = 0; i + 3 < end; i += 4) {
    if (!counts(i) || level(i) < threshold) continue;
    sums[0] += data[i];
    sums[1] += data[i + 1];
    sums[2] += data[i + 2];
    n++;
  }
  return [sums[0] / n / 255, sums[1] / n / 255, sums[2] / n / 255];
}

/**
 * The per-channel gains that make *light* grey at its own luminance, each
 * kept within the caps. A channel with next to nothing in it gets no gain,
 * rather than a huge one.
 */
export function balanceFor(light: Rgb): Rgb {
  const luma = LUMA[0] * light[0] + LUMA[1] * light[1] + LUMA[2] * light[2];
  return light.map((m) => (m > 1e-3 && luma > 1e-3 ? clamp(luma / m, BALANCE_MIN, BALANCE_MAX) : 1)) as unknown as Rgb;
}

/** The inner face's luminance levels, as taken and under *balance*; null when the box holds no pixel. */
function faceLevels(
  data: ArrayLike<number>,
  width: number,
  box: PixelBox,
  balance: Rgb,
): { raw: Levels; balanced: Levels } | null {
  const raw = new Array<number>(256).fill(0);
  const balanced = new Array<number>(256).fill(0);
  let count = 0;
  const [wr, wg, wb] = [LUMA[0] * balance[0], LUMA[1] * balance[1], LUMA[2] * balance[2]];
  for (let y = box.y0; y < box.y1; y++) {
    for (let x = box.x0; x < box.x1; x++) {
      const i = (y * width + x) * 4;
      if (i + 3 >= data.length || data[i + 3] < MIN_ALPHA) continue;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      raw[Math.round(LUMA[0] * r + LUMA[1] * g + LUMA[2] * b)]++;
      balanced[Math.min(255, Math.round(wr * r + wg * g + wb * b))]++;
      count++;
    }
  }
  if (!count) return null;
  return { raw: levelsOf(raw, count), balanced: levelsOf(balanced, count) };
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
