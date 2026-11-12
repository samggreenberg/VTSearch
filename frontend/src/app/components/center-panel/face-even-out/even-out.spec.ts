import {
  BALANCE_MAX,
  BALANCE_MIN,
  CONTRAST_TARGET,
  EVEN_OUT_DEFAULTS,
  EXPOSURE_TARGET,
  EvenOutChoices,
  GAMMA_MIN,
  PixelBox,
  Rgb,
  STRETCH_MAX,
  ToneMap,
  applyCurve,
  evenOutChoices,
  greyPair,
  balanceFor,
  lightOf,
  innerFaceBox,
  measurePixels,
  toneMap,
} from './even-out';

/**
 * The arithmetic of evening out two faces (#5674, #5675): the greyscale test,
 * the measurements, the caps and the map, on pixel arrays built here, since
 * jsdom has no real canvas.
 */
describe('even-out', () => {
  const W = 100;
  const H = 120;
  /** A face looking at the camera, as fractions of a W × H crop. */
  const LANDMARKS = [
    [0.35, 0.42],
    [0.65, 0.42],
    [0.5, 0.55],
    [0.4, 0.68],
    [0.6, 0.68],
  ];
  const ALL_ON: EvenOutChoices = { ...EVEN_OUT_DEFAULTS };
  const ALL_OFF: EvenOutChoices = {
    exposure: false,
    contrast: false,
    white_balance: false,
    greyscale_either: false,
    greyscale_both: false,
    turn_matched: false,
  };

  type Paint = (x: number, y: number) => Rgb;

  /** An RGBA picture W × H painted by *paint*, channel values 0..255. */
  function picture(paint: Paint, width = W, height = H): Uint8ClampedArray {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const [r, g, b] = paint(x, y);
        const i = (y * width + x) * 4;
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        data[i + 3] = 255;
      }
    }
    return data;
  }

  /** A seeded pseudo-random source, so a noisy picture is the same every run. */
  function seeded(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
  }

  /**
   * A crop with a face in it: *skin* over the inner face, shaded from 0.7× at
   * the top to 1.3× at the bottom so it has a spread to measure, and
   * *background* round it; every channel times *light*, the scene's cast.
   */
  function faceCrop(skin: Rgb, background: Rgb, light: Rgb = [1, 1, 1]): { data: Uint8ClampedArray; box: PixelBox } {
    const box = innerFaceBox(LANDMARKS, W, H)!;
    const data = picture((x, y) => {
      const inFace = x >= box.x0 && x < box.x1 && y >= box.y0 && y < box.y1;
      const shade = inFace ? 0.7 + (0.6 * (y - box.y0)) / (box.y1 - box.y0) : 1;
      const base = inFace ? skin : background;
      return [base[0] * shade * light[0], base[1] * shade * light[1], base[2] * shade * light[2]];
    });
    return { data, box };
  }

  /** *data* through *map*, as the filter draws it. */
  function through(map: ToneMap | null, data: Uint8ClampedArray): Uint8ClampedArray {
    if (!map) return data;
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i += 4) {
      out[i] = Math.round(255 * applyCurve(map.r, data[i] / 255));
      out[i + 1] = Math.round(255 * applyCurve(map.g, data[i + 1] / 255));
      out[i + 2] = Math.round(255 * applyCurve(map.b, data[i + 2] / 255));
      out[i + 3] = data[i + 3];
    }
    return out;
  }

  /** The mean colour of *box* on *data*, 0..1. */
  function averageOf(data: Uint8ClampedArray, box: PixelBox): Rgb {
    const sums = [0, 0, 0];
    let n = 0;
    for (let y = box.y0; y < box.y1; y++) {
      for (let x = box.x0; x < box.x1; x++) {
        const i = (y * W + x) * 4;
        sums[0] += data[i];
        sums[1] += data[i + 1];
        sums[2] += data[i + 2];
        n++;
      }
    }
    return [sums[0] / n / 255, sums[1] / n / 255, sums[2] / n / 255];
  }

  describe('the greyscale test', () => {
    it('reads a black-and-white photo as greyscale, JPEG chroma noise and all', () => {
      const rand = seeded(1);
      const data = picture((x, y) => {
        const v = 40 + ((x * 7 + y * 3) % 180);
        // A few levels of disagreement on most pixels, and a stray handful far off.
        const noise = rand() < 0.005 ? 40 : Math.round((rand() - 0.5) * 6);
        return [v + noise, v, v - noise];
      });
      expect(measurePixels(data, W, H, null).grey).toBe(true);
    });

    it('reads a colour face as colour', () => {
      const { data } = faceCrop([200, 150, 120], [90, 110, 140]);
      expect(measurePixels(data, W, H, null).grey).toBe(false);
    });

    it('reads a faintly tinted picture with real colour in it as colour', () => {
      // A grey scene with one coloured object over a tenth of it.
      const data = picture((x, y) => (x < 10 ? [180, 60, 60] : [120 + (y % 5), 120 + (y % 5), 120 + (y % 5)]));
      expect(measurePixels(data, W, H, null).grey).toBe(false);
    });

    it('leaves transparent pixels out', () => {
      const data = picture(() => [128, 128, 128]);
      // A colourful border, all of it transparent.
      for (let i = 0; i < W * 4 * 10; i += 4) {
        data[i] = 255;
        data[i + 1] = 0;
        data[i + 3] = 0;
      }
      expect(measurePixels(data, W, H, null).grey).toBe(true);
    });
  });

  describe('the inner face', () => {
    it('holds the eyes and the mouth, and leaves the hair and the background out', () => {
      const box = innerFaceBox(LANDMARKS, W, H)!;
      const inside = (fx: number, fy: number) => {
        const x = fx * W;
        const y = fy * H;
        return x >= box.x0 && x < box.x1 && y >= box.y0 && y < box.y1;
      };
      for (const [fx, fy] of LANDMARKS) expect(inside(fx, fy)).toBe(true);
      // The crown and the corners of the crop are not the face.
      expect(inside(0.5, 0.1)).toBe(false);
      expect(inside(0.05, 0.5)).toBe(false);
      expect(inside(0.95, 0.5)).toBe(false);
      expect(inside(0.5, 0.97)).toBe(false);
    });

    it('is none without five usable landmarks', () => {
      expect(innerFaceBox(null, W, H)).toBeNull();
      expect(innerFaceBox(LANDMARKS.slice(0, 4), W, H)).toBeNull();
    });

    it('is cut to the picture', () => {
      // A face low in a crop cut short at the photo's edge: the box would run off its foot.
      const low = LANDMARKS.map(([x, y]) => [x, y + 0.25]);
      const box = innerFaceBox(low, W, H)!;
      expect(box.y1).toBe(H);
      expect(box.y0).toBeGreaterThan(0);
    });
  });

  describe('measuring', () => {
    it('reads the inner face, not the background, for exposure and contrast', () => {
      const { data, box } = faceCrop([60, 60, 60], [230, 230, 230]);
      const measure = measurePixels(data, W, H, box);
      // The skin is 60 shaded from 0.7× to 1.3×: about 60 in the middle.
      expect(measure.face!.raw.median).toBeCloseTo(60 / 255, 1);
      expect(measure.face!.raw.low).toBeLessThan(measure.face!.raw.median);
      expect(measure.face!.raw.high).toBeGreaterThan(measure.face!.raw.median);
    });

    it('has no face levels without a box', () => {
      const { data } = faceCrop([60, 60, 60], [230, 230, 230]);
      expect(measurePixels(data, W, H, null).face).toBeNull();
    });
  });

  describe('white balance', () => {
    it('makes the light grey at its own luminance', () => {
      const light: Rgb = [0.55, 0.5, 0.42];
      const gains = balanceFor(light);
      const balanced = light.map((m, c) => m * gains[c]);
      expect(balanced[0]).toBeCloseTo(balanced[1], 6);
      expect(balanced[1]).toBeCloseTo(balanced[2], 6);
      const luma = (c: readonly number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      expect(luma(balanced)).toBeCloseTo(luma(light), 6);
    });

    it('caps every gain', () => {
      const gains = balanceFor([0.9, 0.4, 0.1]);
      for (const g of gains) {
        expect(g).toBeGreaterThanOrEqual(BALANCE_MIN);
        expect(g).toBeLessThanOrEqual(BALANCE_MAX);
      }
      expect(gains[0]).toBe(BALANCE_MIN);
      expect(gains[2]).toBe(BALANCE_MAX);
    });

    it('gives a black light no gain', () => {
      expect(balanceFor([0, 0, 0])).toEqual([1, 1, 1]);
    });

    it('reads the light from the brightest pixels, not the skin, and leaves clipped ones out', () => {
      // Mostly skin; a tenth a lit wall, the light's colour; a little blown out to white.
      const data = picture((x, y) => {
        if (y < 3) return [255, 255, 255];
        if (y < 15) return [220, 205, 180];
        return [150 + (x % 7), 100, 80];
      });
      const light = lightOf(data)!;
      expect(light[0]).toBeCloseTo(220 / 255, 2);
      expect(light[1]).toBeCloseTo(205 / 255, 2);
      expect(light[2]).toBeCloseTo(180 / 255, 2);
    });

    it('reads a picture blown out everywhere rather than nothing', () => {
      expect(lightOf(picture(() => [255, 255, 255]))).toEqual([1, 1, 1]);
      expect(lightOf(new Uint8ClampedArray(0))).toBeNull();
    });

    it("leaves a greyscale photo's balance alone", () => {
      const data = picture((x, y) => {
        const v = 40 + ((x * 7 + y * 3) % 180);
        return [v, v, v];
      });
      expect(measurePixels(data, W, H, null).balance).toEqual([1, 1, 1]);
    });

    /** A crop of *skin* against a darker grey scene, with a lit pale wall across its top, all under *light*. */
    function litCrop(skin: Rgb, light: Rgb = [1, 1, 1]): { data: Uint8ClampedArray; box: PixelBox } {
      const crop = faceCrop(skin, [90, 90, 90], light);
      for (let y = 0; y < 12; y++) {
        for (let x = 0; x < W; x++) {
          const i = (y * W + x) * 4;
          [crop.data[i], crop.data[i + 1], crop.data[i + 2]] = [210 * light[0], 210 * light[1], 210 * light[2]];
        }
      }
      return crop;
    }

    it('takes most of a warm cast out, and two people keep their skin', () => {
      const warm: Rgb = [1.15, 1, 0.82];
      const lightSkin: Rgb = [200, 150, 125];
      const darkSkin: Rgb = [115, 75, 58];
      const ratio = (c: Rgb) => c[0] / c[2];
      const shown = (crop: { data: Uint8ClampedArray; box: PixelBox }) =>
        through(toneMap(measurePixels(crop.data, W, H, crop.box), ALL_OFF_BUT('white_balance')), crop.data);

      // One person by daylight and under a lamp: the cast between the two photos is mostly gone.
      const daylight = litCrop(lightSkin);
      const indoor = litCrop(lightSkin, warm);
      const box = daylight.box;
      const castBefore = ratio(averageOf(indoor.data, box)) / ratio(averageOf(daylight.data, box));
      const castAfter = ratio(averageOf(shown(indoor), box)) / ratio(averageOf(shown(daylight), box));
      expect(castBefore).toBeGreaterThan(1.3);
      expect(Math.abs(castAfter - 1)).toBeLessThan(0.25 * Math.abs(castBefore - 1));

      // Two people in the same light: their skins differ as much after as before.
      const before = ratio(averageOf(litCrop(darkSkin).data, box)) - ratio(averageOf(daylight.data, box));
      const after = ratio(averageOf(shown(litCrop(darkSkin)), box)) - ratio(averageOf(shown(daylight), box));
      expect(after).toBeCloseTo(before, 1);
      expect(after).toBeGreaterThan(0.2);
    });
  });

  function ALL_OFF_BUT(...on: (keyof EvenOutChoices)[]): EvenOutChoices {
    const choices: Record<string, boolean> = { ...ALL_OFF };
    for (const key of on) choices[key] = true;
    return choices as unknown as EvenOutChoices;
  }

  /** The inner face's median luminance on *data*, 0..1. */
  function faceMedian(data: Uint8ClampedArray, box: PixelBox): number {
    return measurePixels(data, W, H, box).face!.raw.median;
  }

  describe('the map', () => {
    const grey: Rgb = [128, 128, 128];

    it('brings a dark face and a bright one to medians close together', () => {
      const darkCrop = faceCrop([70, 70, 70], grey);
      const brightCrop = faceCrop([175, 175, 175], grey);
      const exposure = ALL_OFF_BUT('exposure');
      const dark = through(toneMap(measurePixels(darkCrop.data, W, H, darkCrop.box), exposure), darkCrop.data);
      const bright = through(toneMap(measurePixels(brightCrop.data, W, H, brightCrop.box), exposure), brightCrop.data);
      const gapBefore = faceMedian(brightCrop.data, brightCrop.box) - faceMedian(darkCrop.data, darkCrop.box);
      const gapAfter = Math.abs(faceMedian(bright, brightCrop.box) - faceMedian(dark, darkCrop.box));
      expect(gapAfter).toBeLessThan(0.05);
      expect(gapBefore).toBeGreaterThan(0.35);
      expect(faceMedian(dark, darkCrop.box)).toBeCloseTo(EXPOSURE_TARGET, 1);
    });

    it('leaves each face as taken with exposure unticked', () => {
      const crop = faceCrop([45, 45, 45], grey);
      expect(toneMap(measurePixels(crop.data, W, H, crop.box), ALL_OFF)).toBeNull();
    });

    it('caps the exposure curve, so a near-black face is lifted only so far', () => {
      const crop = faceCrop([8, 8, 8], grey);
      const map = toneMap(measurePixels(crop.data, W, H, crop.box), ALL_OFF_BUT('exposure'))!;
      expect(map.r.exponent).toBe(GAMMA_MIN);
      expect(faceMedian(through(map, crop.data), crop.box)).toBeLessThan(EXPOSURE_TARGET - 0.1);
    });

    it('brings a flat face toward the target spread, and caps the stretch', () => {
      // Shaded 0.7× to 1.3× about 128: a spread well under the target.
      const crop = faceCrop([128, 128, 128], grey);
      const measure = measurePixels(crop.data, W, H, crop.box);
      const spreadBefore = measure.face!.raw.high - measure.face!.raw.low;
      const out = through(toneMap(measure, ALL_OFF_BUT('contrast')), crop.data);
      const after = measurePixels(out, W, H, crop.box).face!.raw;
      expect(after.high - after.low).toBeGreaterThan(spreadBefore);
      expect(Math.abs(after.high - after.low - CONTRAST_TARGET)).toBeLessThan(Math.abs(spreadBefore - CONTRAST_TARGET));
      // The stretch keeps the median where it was.
      expect(after.median).toBeCloseTo(measure.face!.raw.median, 1);

      // A face with no spread at all is stretched only as far as the cap.
      const flat = faceCrop([128, 128, 128], grey);
      const box = flat.box;
      for (let y = box.y0; y < box.y1; y++) {
        for (let x = box.x0; x < box.x1; x++) {
          const i = (y * W + x) * 4;
          flat.data[i] = flat.data[i + 1] = flat.data[i + 2] = 128;
        }
      }
      const flatMap = toneMap(measurePixels(flat.data, W, H, box), ALL_OFF_BUT('contrast'))!;
      expect(flatMap.g.amplitude).toBeCloseTo(STRETCH_MAX, 6);
    });

    it('removes only its own part of the map when a box is unticked', () => {
      const crop = faceCrop([60, 50, 40], [140, 120, 100], [1.1, 1, 0.85]);
      const measure = measurePixels(crop.data, W, H, crop.box);
      const all = toneMap(measure, ALL_ON)!;

      // Exposure off: no exponent; the balance and the stretch are still there.
      const noExposure = toneMap(measure, { ...ALL_ON, exposure: false })!;
      expect(noExposure.r.exponent).toBe(1);
      expect(noExposure.r.amplitude).not.toBeCloseTo(noExposure.b.amplitude, 3);
      expect(noExposure.g.offset).not.toBe(0);

      // White balance off: the three channels get the same curve.
      const noBalance = toneMap(measure, { ...ALL_ON, white_balance: false })!;
      expect(noBalance.r).toEqual(noBalance.g);
      expect(noBalance.g).toEqual(noBalance.b);
      expect(noBalance.r.exponent).not.toBe(1);

      // Contrast off: no stretch, so no offset, and each amplitude is the balance gain through the curve.
      const noContrast = toneMap(measure, { ...ALL_ON, contrast: false })!;
      expect(noContrast.r.offset).toBe(0);
      expect(noContrast.r.exponent).toBeCloseTo(all.r.exponent, 9);
      expect(noContrast.r.amplitude).toBeCloseTo(Math.pow(measure.balance[0], noContrast.r.exponent), 9);
    });

    it('gives a face without landmarks white balance only', () => {
      const crop = faceCrop([60, 50, 40], [140, 120, 100], [1.1, 1, 0.85]);
      const map = toneMap(measurePixels(crop.data, W, H, null), ALL_ON)!;
      for (const channel of [map.r, map.g, map.b]) {
        expect(channel.exponent).toBe(1);
        expect(channel.offset).toBe(0);
      }
      expect(map.r.amplitude).toBeLessThan(1);
      expect(map.b.amplitude).toBeGreaterThan(1);
    });

    it('is the same curve on every pixel: nothing is applied to part of an image', () => {
      // The face is striped, half its pixels the background's own value.
      const box = innerFaceBox(LANDMARKS, W, H)!;
      const inFace = (x: number, y: number) => x >= box.x0 && x < box.x1 && y >= box.y0 && y < box.y1;
      const data = picture((x, y): Rgb => (inFace(x, y) && x % 2 === 0 ? [40, 35, 30] : [100, 90, 80]));
      const out = through(toneMap(measurePixels(data, W, H, box), ALL_ON), data);
      const inside = ((box.y0 + 2) * W + (box.x0 % 2 === 0 ? box.x0 + 1 : box.x0)) * 4;
      const outside = 0;
      expect(data.slice(inside, inside + 3)).toEqual(data.slice(outside, outside + 3));
      expect(out.slice(inside, inside + 3)).toEqual(out.slice(outside, outside + 3));
      expect(out.slice(outside, outside + 3)).not.toEqual(data.slice(outside, outside + 3));
    });
  });

  describe('the greyscale pair', () => {
    it('greys both when either is greyscale, and neither otherwise', () => {
      expect(greyPair(true, false, ALL_ON)).toBe(true);
      expect(greyPair(false, true, ALL_ON)).toBe(true);
      expect(greyPair(false, false, ALL_ON)).toBe(false);
      // Not measured yet is not greyscale.
      expect(greyPair(null, null, ALL_ON)).toBe(false);
    });

    it('shows a mixed pair as taken with the rule unticked', () => {
      expect(greyPair(true, false, { ...ALL_ON, greyscale_either: false })).toBe(false);
    });

    it('greys two colour faces under Both in greyscale', () => {
      expect(greyPair(false, false, { ...ALL_OFF, greyscale_both: true })).toBe(true);
    });
  });

  describe('the stored choices', () => {
    it('fills what the setting lacks from the defaults, and ignores what is not a boolean', () => {
      expect(evenOutChoices(null)).toEqual(EVEN_OUT_DEFAULTS);
      expect(evenOutChoices({ exposure: false, contrast: 'no' })).toEqual({ ...EVEN_OUT_DEFAULTS, exposure: false });
    });
  });
});
