import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { FaceEvenOutService, MeasureRequest } from './face-even-out.service';
import { SettingsStateService } from './settings-state.service';
import { EVEN_OUT_DEFAULTS, PixelMeasure } from '../components/center-panel/face-even-out/even-out';
import { provideHttpTesting } from '../testing/test-providers';
import { settleResource } from '../testing/settle-resource';

/**
 * The compare layout's evening out (#5674, #5675): the choices, held over the
 * round trip, and the reading of a picture's pixels, here through a stand-in
 * `Image` and canvas, since jsdom loads no pictures and draws nothing.
 */
describe('FaceEvenOutService', () => {
  let service: FaceEvenOutService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    service = TestBed.inject(FaceEvenOutService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function loadSettings(body: Record<string, unknown>): Promise<void> {
    TestBed.inject(SettingsStateService).load();
    TestBed.tick();
    await settleResource();
    httpMock.expectOne('/api/settings').flush(body);
    await settleResource();
  }

  describe('the choices', () => {
    it('are the defaults before settings land', () => {
      expect(service.choices()).toEqual(EVEN_OUT_DEFAULTS);
    });

    it('are the setting over the defaults', async () => {
      await loadSettings({ face_even_out: { contrast: false } });
      expect(service.choices()).toEqual({ ...EVEN_OUT_DEFAULTS, contrast: false });
    });

    it('take a click at once, write them all, and hold it until the server answers', async () => {
      await loadSettings({ face_even_out: { contrast: false } });

      service.set('greyscale_both', true);
      expect(service.choices().greyscale_both).toBe(true);
      expect(service.choices().contrast).toBe(false);

      const req = httpMock.expectOne('/api/settings');
      expect(req.request.method).toBe('PUT');
      expect(req.request.body).toEqual({ face_even_out: { ...EVEN_OUT_DEFAULTS, contrast: false, greyscale_both: true } });
      req.flush({ face_even_out: { ...EVEN_OUT_DEFAULTS, contrast: false, greyscale_both: true } });
      expect(service.choices().greyscale_both).toBe(true);
    });

    it('fall back to the saved ones when the write fails', async () => {
      await loadSettings({});
      service.set('exposure', false);
      expect(service.choices().exposure).toBe(false);
      httpMock.expectOne('/api/settings').flush('boom', { status: 500, statusText: 'Server Error' });
      expect(service.choices().exposure).toBe(true);
    });
  });

  describe('measuring a picture', () => {
    /** Every stand-in picture made, in order, to load or fail. */
    let images: FakeImage[];

    class FakeImage {
      naturalWidth = 40;
      naturalHeight = 50;
      decoding = '';
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      src = '';
      constructor() {
        images.push(this);
      }
    }

    beforeEach(() => {
      images = [];
      vi.stubGlobal('Image', FakeImage);
      // A canvas that "draws" a flat mid-grey picture of the size asked for.
      const realCreate = document.createElement.bind(document);
      vi.spyOn(document, 'createElement').mockImplementation((tag: string, options?: ElementCreationOptions) => {
        if (tag !== 'canvas') return realCreate(tag, options);
        const canvas = { width: 0, height: 0 } as { width: number; height: number; getContext: unknown };
        canvas.getContext = () => ({
          drawImage: vi.fn(),
          getImageData: (_x: number, _y: number, w: number, h: number) => ({
            data: new Uint8ClampedArray(w * h * 4).fill(128),
          }),
        });
        return canvas as unknown as HTMLElement;
      });
    });

    const REQUEST: MeasureRequest = { src: '/api/medias/4/image?dataset_id=faces', landmarks: null };

    function collect(request: MeasureRequest | undefined): (PixelMeasure | null)[] {
      const seen: (PixelMeasure | null)[] = [];
      service.measured(request).subscribe((m) => seen.push(m));
      return seen;
    }

    it('answers no request with nothing', () => {
      expect(collect(undefined)).toEqual([null]);
      expect(images.length).toBe(0);
    });

    it('answers nothing at once, then what the pixels say once the picture loads', async () => {
      const seen = collect(REQUEST);
      expect(seen).toEqual([null]);
      expect(images[0].src).toBe(REQUEST.src);

      images[0].onload!();
      await Promise.resolve();
      await Promise.resolve();
      expect(seen.length).toBe(2);
      expect(seen[1]!.grey).toBe(true);
      // No landmarks: read where a crop's face sits, here a flat mid-grey.
      expect(seen[1]!.face!.raw.median).toBeCloseTo(128 / 255, 2);
    });

    it('reads a picture once, and answers again at once', async () => {
      collect(REQUEST);
      images[0].onload!();
      await new Promise((resolve) => setTimeout(resolve));

      const again = collect(REQUEST);
      expect(again.length).toBe(1);
      expect(again[0]!.grey).toBe(true);
      expect(images.length).toBe(1);
    });

    it('reads it again for other landmarks, which place another face', () => {
      collect(REQUEST);
      collect({ ...REQUEST, landmarks: [[0.4, 0.4], [0.6, 0.4], [0.5, 0.5], [0.45, 0.6], [0.55, 0.6]] });
      expect(images.length).toBe(2);
    });

    it('keeps no failed read, so the next ask tries again', async () => {
      const seen = collect(REQUEST);
      images[0].onerror!();
      await new Promise((resolve) => setTimeout(resolve));
      expect(seen).toEqual([null, null]);

      collect(REQUEST);
      expect(images.length).toBe(2);
    });
  });
});
