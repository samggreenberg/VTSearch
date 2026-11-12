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
  TYPICAL_LANDMARKS,
  TYPICAL_SKIN,
  ToneMap,
  applyCurve,
  evenOutChoices,
  greyPair,
  measurePixels,
  skinBalance,
  skinBoxes,
  skinColour,
  toneMap,
} from './even-out';

/**
 * The arithmetic of evening out two faces (#5674, #5675): the greyscale test,
 * the skin patches, the measurements, the caps and the map, on pixel arrays
 * built here, since jsdom has no real canvas.
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

  const SKIN = skinBoxes(LANDMARKS, W, H);
  /** The face: every skin patch, and a margin round them. */
  const FACE: PixelBox = {
    x0: Math.min(...SKIN.map((b) => b.x0)) - 4,
    y0: Math.min(...SKIN.map((b) => b.y0)) - 4,
    x1: Math.max(...SKIN.map((b) => b.x1)) + 4,
    y1: Math.max(...SKIN.map((b) => b.y1)) + 4,
  };
  const inFace = (x: number, y: number) => x >= FACE.x0 && x < FACE.x1 && y >= FACE.y0 && y < FACE.y1;

  /**
   * A crop with a face in it: *skin* over the face, shaded from *shade*[0]
   * times at its top to *shade*[1] times at its foot so it has a spread to
   * measure, and *wall* round it; every channel times *light*, the scene's cast.
   */
  function faceCrop(skin: Rgb, wall: Rgb, light: Rgb = [1, 1, 1], shade: [number, number] = [0.85, 1.15]): Uint8ClampedArray {
    return picture((x, y) => {
      const base = inFace(x, y) ? skin : wall;
      const s = inFace(x, y) ? shade[0] + ((shade[1] - shade[0]) * (y - FACE.y0)) / (FACE.y1 - FACE.y0) : 1;
      return [base[0] * s * light[0], base[1] * s * light[1], base[2] * s * light[2]];
    });
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

  /** The mean colour of the skin patches on *data*, 0..1. */
  function skinAverage(data: Uint8ClampedArray): Rgb {
    const sums = [0, 0, 0];
    let n = 0;
    for (const box of SKIN) {
      for (let y = box.y0; y < box.y1; y++) {
        for (let x = box.x0; x < box.x1; x++) {
          const i = (y * W + x) * 4;
          sums[0] += data[i];
          sums[1] += data[i + 1];
          sums[2] += data[i + 2];
          n++;
        }
      }
    }
    return [sums[0] / n / 255, sums[1] / n / 255, sums[2] / n / 255];
  }

  /** The map for *data*, measured on the skin. */
  function mapOf(data: Uint8ClampedArray, choices: EvenOutChoices = ALL_ON): ToneMap | null {
    return toneMap(measurePixels(data, W, H, SKIN), choices);
  }

  /** The skin's median luminance on *data*, 0..1. */
  function skinMedian(data: Uint8ClampedArray): number {
    return measurePixels(data, W, H, SKIN).face!.raw.median;
  }

  function ALL_OFF_BUT(...on: (keyof EvenOutChoices)[]): EvenOutChoices {
    const choices: Record<string, boolean> = { ...ALL_OFF };
    for (const key of on) choices[key] = true;
    return choices as unknown as EvenOutChoices;
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
      expect(measurePixels(data, W, H, []).grey).toBe(true);
    });

    it('reads a colour face as colour', () => {
      expect(measurePixels(faceCrop([200, 150, 120], [90, 110, 140]), W, H, []).grey).toBe(false);
    });

    it('reads a faintly tinted picture with real colour in it as colour', () => {
      // A grey scene with one coloured object over a tenth of it.
      const data = picture((x, y) => (x < 10 ? [180, 60, 60] : [120 + (y % 5), 120 + (y % 5), 120 + (y % 5)]));
      expect(measurePixels(data, W, H, []).grey).toBe(false);
    });

    it('leaves transparent pixels out', () => {
      const data = picture(() => [128, 128, 128]);
      // A colourful border, all of it transparent.
      for (let i = 0; i < W * 4 * 10; i += 4) {
        data[i] = 255;
        data[i + 1] = 0;
        data[i + 3] = 0;
      }
      expect(measurePixels(data, W, H, []).grey).toBe(true);
    });
  });

  describe('the skin patches', () => {
    const holds = (box: PixelBox, fx: number, fy: number) => {
      const x = fx * W;
      const y = fy * H;
      return x >= box.x0 && x < box.x1 && y >= box.y0 && y < box.y1;
    };

    it('sit on the forehead, the nose and both cheeks, clear of the eyes, the mouth and the wall', () => {
      expect(SKIN.length).toBe(4);
      for (const [fx, fy] of LANDMARKS) {
        for (const box of SKIN) expect(holds(box, fx, fy)).toBe(false);
      }
      const eyeY = LANDMARKS[0][1] * H;
      const mouthY = LANDMARKS[3][1] * H;
      // The forehead over the eyes, the cheeks under them and either side of the nose.
      expect(SKIN.filter((b) => b.y1 <= eyeY).length).toBe(1);
      const cheeks = SKIN.filter((b) => b.y0 > eyeY && (b.x1 < W / 2 - 3 || b.x0 > W / 2 + 3));
      expect(cheeks.length).toBe(2);
      for (const cheek of cheeks) expect(cheek.y1).toBeLessThan(mouthY);
      // Well inside the crop, where the wall is not.
      for (const box of SKIN) {
        expect(box.x0).toBeGreaterThan(10);
        expect(box.x1).toBeLessThan(W - 10);
        expect(box.y0).toBeGreaterThan(10);
      }
    });

    it('leave out the far cheek of a turned head', () => {
      // The nose off toward the image's left: the head is turned that way, its left cheek away.
      const turned = LANDMARKS.map((p, i) => (i === 2 ? [0.4, p[1]] : p));
      const boxes = skinBoxes(turned, W, H);
      expect(boxes.length).toBe(3);
      const cheeks = boxes.filter((b) => b.y0 > LANDMARKS[0][1] * H && (b.x1 < W / 2 - 3 || b.x0 > W / 2 + 3));
      expect(cheeks.length).toBe(1);
      expect(cheeks[0].x0).toBeGreaterThan(W / 2);
    });

    it('are read where a crop\'s face usually sits for a face with no landmarks', () => {
      const typical = skinBoxes(TYPICAL_LANDMARKS, W, H);
      expect(typical.length).toBe(4);
      expect(skinBoxes(null, W, H)).toEqual(typical);
      expect(skinBoxes(LANDMARKS.slice(0, 4), W, H)).toEqual(typical);
    });

    it('are cut to the picture', () => {
      // A face against the crop's right edge, cut short at its photo's.
      const edge = LANDMARKS.map(([x, y]) => [x + 0.3, y]);
      const boxes = skinBoxes(edge, W, H);
      expect(boxes.some((b) => b.x1 === W)).toBe(true);
      for (const box of boxes) expect(box.x1).toBeLessThanOrEqual(W);
    });
  });

  describe('measuring', () => {
    it('reads the skin alone: a white wall and a black wall behind the same face give the same map', () => {
      const skin: Rgb = [170, 120, 95];
      const white = measurePixels(faceCrop(skin, [250, 250, 245]), W, H, SKIN);
      const black = measurePixels(faceCrop(skin, [8, 8, 10]), W, H, SKIN);
      expect(white).toEqual(black);
      expect(toneMap(white, ALL_ON)).toEqual(toneMap(black, ALL_ON));
      // And a coloured wall, which once swayed the white balance.
      expect(measurePixels(faceCrop(skin, [60, 90, 200]), W, H, SKIN)).toEqual(white);
    });

    it('never counts the eyes, the brows or the lips', () => {
      const plain = faceCrop([170, 120, 95], [128, 128, 128]);
      const featured = plain.slice();
      const paint = (fx: number, fy: number, half: number, colour: Rgb) => {
        for (let y = Math.round(fy * H) - half; y <= Math.round(fy * H) + half; y++) {
          for (let x = Math.round(fx * W) - half; x <= Math.round(fx * W) + half; x++) {
            const i = (y * W + x) * 4;
            [featured[i], featured[i + 1], featured[i + 2]] = colour;
          }
        }
      };
      for (const eye of LANDMARKS.slice(0, 2)) {
        paint(eye[0], eye[1], 3, [20, 15, 10]);
        // The brow over it.
        paint(eye[0], eye[1] - 0.06, 2, [40, 30, 20]);
      }
      paint(0.5, 0.68, 3, [180, 40, 50]);
      expect(measurePixels(featured, W, H, SKIN)).toEqual(measurePixels(plain, W, H, SKIN));
    });

    it("reads the skin's brightness and spread", () => {
      const measure = measurePixels(faceCrop([60, 60, 60], [230, 230, 230]), W, H, SKIN);
      // The skin is 60, shaded from 0.85× to 1.15×: about 60 in the middle.
      expect(measure.face!.raw.median).toBeCloseTo(60 / 255, 1);
      expect(measure.face!.raw.low).toBeLessThan(measure.face!.raw.median);
      expect(measure.face!.raw.high).toBeGreaterThan(measure.face!.raw.median);
    });

    it('reads nothing but greyscale from a picture with no skin patch on it', () => {
      const measure = measurePixels(faceCrop([200, 150, 120], [90, 110, 140]), W, H, []);
      expect(measure.face).toBeNull();
      expect(measure.balance).toEqual([1, 1, 1]);
      expect(toneMap(measure, ALL_ON)).toBeNull();
    });
  });

  describe('white balance', () => {
    const luma = (c: readonly number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

    it('moves the skin to the typical skin colour at its own luminance', () => {
      const skin: Rgb = [0.5 * TYPICAL_SKIN[0] * 1.06, 0.5, 0.5 * TYPICAL_SKIN[2] * 0.94];
      const gains = skinBalance(skin);
      const balanced = skin.map((m, c) => m * gains[c]);
      expect(balanced[0] / balanced[1]).toBeCloseTo(TYPICAL_SKIN[0] / TYPICAL_SKIN[1], 6);
      expect(balanced[2] / balanced[1]).toBeCloseTo(TYPICAL_SKIN[2] / TYPICAL_SKIN[1], 6);
      expect(luma(balanced)).toBeCloseTo(luma(skin), 6);
    });

    it('leaves a typical skin alone', () => {
      const gains = skinBalance([TYPICAL_SKIN[0] * 0.4, TYPICAL_SKIN[1] * 0.4, TYPICAL_SKIN[2] * 0.4]);
      for (const g of gains) expect(g).toBeCloseTo(1, 9);
    });

    it('caps every gain', () => {
      const gains = skinBalance([0.9, 0.4, 0.1]);
      for (const g of gains) {
        expect(g).toBeGreaterThanOrEqual(BALANCE_MIN);
        expect(g).toBeLessThanOrEqual(BALANCE_MAX);
      }
      expect(gains[0]).toBe(BALANCE_MIN);
      expect(gains[2]).toBe(BALANCE_MAX);
    });

    it('gives a black reading no gain', () => {
      expect(skinBalance([0, 0, 0])).toEqual([1, 1, 1]);
    });

    it("reads the skin's colour without its shine or a mole", () => {
      const rand = seeded(7);
      const data = picture(() => {
        const r = rand();
        if (r < 0.05) return [255, 255, 255];
        if (r < 0.1) return [40, 25, 20];
        return [170, 120, 95];
      });
      const pixels = Array.from({ length: W * H }, (_, k) => k * 4);
      const colour = skinColour(data, pixels);
      expect(colour[0]).toBeCloseTo(170 / 255, 2);
      expect(colour[1]).toBeCloseTo(120 / 255, 2);
      expect(colour[2]).toBeCloseTo(95 / 255, 2);
    });

    it("leaves a greyscale photo's balance alone", () => {
      const data = picture((x, y) => {
        const v = 40 + ((x * 7 + y * 3) % 180);
        return [v, v, v];
      });
      expect(measurePixels(data, W, H, SKIN).balance).toEqual([1, 1, 1]);
    });

    it('takes most of a warm cast out of the skin', () => {
      const skin: Rgb = [180, 135, 112];
      const warm: Rgb = [1.12, 1, 0.86];
      const ratio = (c: Rgb) => c[0] / c[2];
      const balance = ALL_OFF_BUT('white_balance');
      const daylight = faceCrop(skin, [128, 128, 128]);
      const indoor = faceCrop(skin, [128, 128, 128], warm);
      const castBefore = ratio(skinAverage(indoor)) / ratio(skinAverage(daylight));
      const castAfter =
        ratio(skinAverage(through(mapOf(indoor, balance), indoor))) / ratio(skinAverage(through(mapOf(daylight, balance), daylight)));
      expect(castBefore).toBeGreaterThan(1.25);
      expect(Math.abs(castAfter - 1)).toBeLessThan(0.25 * Math.abs(castBefore - 1));
    });

    it("draws two people's skin toward one colour, only as far as the caps let it", () => {
      const balance = ALL_OFF_BUT('white_balance');
      const ratio = (c: Rgb) => c[0] / c[1];
      const near: Rgb = [180, 135, 112];
      const far: Rgb = [215, 85, 55];
      const before = ratio(skinAverage(faceCrop(far, [128, 128, 128]))) - ratio(skinAverage(faceCrop(near, [128, 128, 128])));
      const shown = (skin: Rgb) => {
        const data = faceCrop(skin, [128, 128, 128]);
        return skinAverage(through(mapOf(data, balance), data));
      };
      const after = ratio(shown(far)) - ratio(shown(near));
      // Read from the skin, it cannot tell a person's colour from a cast: the gap narrows...
      expect(after).toBeLessThan(before);
      // ...but the far skin's red is held at the cap, so some of it stays.
      expect(mapOf(faceCrop(far, [128, 128, 128]), balance)!.r.amplitude).toBeCloseTo(BALANCE_MIN, 6);
      expect(after).toBeGreaterThan(0.1);
    });
  });

  describe('the map', () => {
    const grey: Rgb = [128, 128, 128];

    it('brings a dark face and a bright one to medians close together', () => {
      const darkCrop = faceCrop([105, 105, 105], grey);
      const brightCrop = faceCrop([190, 190, 190], grey);
      const exposure = ALL_OFF_BUT('exposure');
      const dark = through(mapOf(darkCrop, exposure), darkCrop);
      const bright = through(mapOf(brightCrop, exposure), brightCrop);
      const gapBefore = skinMedian(brightCrop) - skinMedian(darkCrop);
      const gapAfter = Math.abs(skinMedian(bright) - skinMedian(dark));
      expect(gapBefore).toBeGreaterThan(0.3);
      expect(gapAfter).toBeLessThan(0.05);
      expect(skinMedian(bright)).toBeCloseTo(EXPOSURE_TARGET, 1);
    });

    it('leaves each face as taken with every box unticked', () => {
      expect(mapOf(faceCrop([105, 105, 105], grey), ALL_OFF)).toBeNull();
    });

    it('caps the exposure curve, so a near-black face is lifted only so far', () => {
      const crop = faceCrop([8, 8, 8], grey);
      const map = mapOf(crop, ALL_OFF_BUT('exposure'))!;
      expect(map.r.exponent).toBe(GAMMA_MIN);
      expect(skinMedian(through(map, crop))).toBeLessThan(EXPOSURE_TARGET - 0.1);
    });

    it('brings an unevenly lit face toward the target spread, and caps the stretch', () => {
      // Shaded 0.6× to 1.4× about 128: a spread well over the target.
      const crop = faceCrop([128, 128, 128], grey, [1, 1, 1], [0.6, 1.4]);
      const measure = measurePixels(crop, W, H, SKIN);
      const spreadBefore = measure.face!.raw.high - measure.face!.raw.low;
      const after = measurePixels(through(toneMap(measure, ALL_OFF_BUT('contrast')), crop), W, H, SKIN).face!.raw;
      expect(spreadBefore).toBeGreaterThan(CONTRAST_TARGET);
      expect(Math.abs(after.high - after.low - CONTRAST_TARGET)).toBeLessThan(Math.abs(spreadBefore - CONTRAST_TARGET));
      // The stretch keeps the median where it was.
      expect(after.median).toBeCloseTo(measure.face!.raw.median, 1);

      // A face lit dead flat is stretched only as far as the cap.
      const flat = mapOf(faceCrop([128, 128, 128], grey, [1, 1, 1], [1, 1]), ALL_OFF_BUT('contrast'))!;
      expect(flat.g.amplitude).toBeCloseTo(STRETCH_MAX, 6);
    });

    it('removes only its own part of the map when a box is unticked', () => {
      const crop = faceCrop([110, 70, 45], [140, 120, 100], [1.1, 1, 0.85], [0.6, 1.4]);
      const measure = measurePixels(crop, W, H, SKIN);
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

    it("measures a face with no landmarks where a crop's face sits", () => {
      const typical = skinBoxes(TYPICAL_LANDMARKS, W, H);
      const data = picture((x, y) => (typical.some((b) => x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1) ? [60, 45, 38] : [240, 240, 240]));
      const map = toneMap(measurePixels(data, W, H, skinBoxes(null, W, H)), ALL_ON)!;
      // A dark face on a white wall is lifted, as the face it is, not darkened for the wall.
      expect(map.g.exponent).toBeLessThan(1);
    });

    it('is the same curve on every pixel: nothing is applied to part of an image', () => {
      // The face is striped, half its pixels the wall's own value.
      const data = picture((x, y): Rgb => (inFace(x, y) && x % 2 === 0 ? [40, 35, 30] : [100, 90, 80]));
      const out = through(mapOf(data), data);
      const inside = ((FACE.y0 + 2) * W + (FACE.x0 % 2 === 0 ? FACE.x0 + 1 : FACE.x0)) * 4;
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
