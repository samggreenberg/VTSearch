import { ElementRef } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageCropOverlayComponent, ImageCropResult } from './image-crop-overlay.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';

/**
 * These specs exercise the crop-box geometry (hit-testing, resize/move drags,
 * min-size and bounds clamping) and the display→natural pixel conversion the
 * component emits. The image ref is faked with known natural (1000×800) and
 * displayed (500×400) dimensions so the coordinate math is deterministic; no
 * real image load or browser layout is involved.
 */
describe('ImageCropOverlayComponent', () => {
  let component: ImageCropOverlayComponent;
  let fixture: ComponentFixture<ImageCropOverlayComponent>;

  const NAT_W = 1000;
  const NAT_H = 800;
  const DISP_W = 500;
  const DISP_H = 400;

  /** Pointer event with a target that stubs pointer capture (jsdom lacks it). */
  function ptr(clientX: number, clientY: number): PointerEvent {
    return {
      clientX,
      clientY,
      pointerId: 1,
      target: { setPointerCapture: vi.fn(), releasePointerCapture: vi.fn() },
      preventDefault: () => {},
    } as unknown as PointerEvent;
  }

  function setImgRef() {
    const img = {
      naturalWidth: NAT_W,
      naturalHeight: NAT_H,
      clientWidth: DISP_W,
      clientHeight: DISP_H,
      complete: true,
      getBoundingClientRect: () =>
        ({ left: 0, top: 0, width: DISP_W, height: DISP_H }) as DOMRect,
    };
    component.imgRef = { nativeElement: img } as unknown as ElementRef<HTMLImageElement>;
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ImageCropOverlayComponent],
      providers: [...provideZoneless()],
    });
    fixture = TestBed.createComponent(ImageCropOverlayComponent);
    component = fixture.componentInstance;
    setImgRef();
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  describe('onImageLoaded', () => {
    it('seeds a centred 60% crop box in displayed coordinates', () => {
      component.onImageLoaded();
      expect(component.cropW).toBe(300); // 60% of 500
      expect(component.cropH).toBe(240); // 60% of 400
      expect(component.cropX).toBe(100); // (500 - 300) / 2
      expect(component.cropY).toBe(80); // (400 - 240) / 2
    });
  });

  describe('dragging', () => {
    beforeEach(() => component.onImageLoaded()); // box = {x:100, y:80, w:300, h:240}

    it('moves the whole box while staying in bounds', () => {
      component.onPointerDown(ptr(250, 200)); // inside box → move
      component.onPointerMove(ptr(280, 230)); // +30, +30
      expect(component.cropX).toBe(130);
      expect(component.cropY).toBe(110);
      expect(component.cropW).toBe(300);
      expect(component.cropH).toBe(240);
    });

    it('resizes from the bottom-right corner', () => {
      component.onPointerDown(ptr(400, 320)); // bottom-right corner (x2,y2)
      component.onPointerMove(ptr(450, 360)); // +50, +40
      expect(component.cropW).toBe(350);
      expect(component.cropH).toBe(280);
      expect(component.cropX).toBe(100);
      expect(component.cropY).toBe(80);
    });

    it('clamps a top-left drag to the image edge, growing width to compensate', () => {
      component.onPointerDown(ptr(100, 80)); // top-left corner
      component.onPointerMove(ptr(-50, 80)); // dx=-150, dy=0
      // Left edge clamps to 0; the extra width is absorbed rather than lost.
      expect(component.cropX).toBe(0);
      expect(component.cropW).toBe(400);
      expect(component.cropY).toBe(80);
    });

    it('enforces a 5px minimum size when a side handle is over-dragged', () => {
      component.onPointerDown(ptr(400, 200)); // right edge → 'r'
      component.onPointerMove(ptr(50, 200)); // collapse width past zero
      expect(component.cropW).toBe(5);
      expect(component.cropX).toBe(100); // left edge held fixed for a right-side drag
    });

    it('ignores pointer motion outside the box (no drag started)', () => {
      component.onPointerDown(ptr(5, 5)); // outside the centred box → 'none'
      component.onPointerMove(ptr(400, 400));
      expect(component.cropX).toBe(100);
      expect(component.cropY).toBe(80);
      expect(component.cropW).toBe(300);
      expect(component.cropH).toBe(240);
    });

    it('stops responding once the drag ends', () => {
      component.onPointerDown(ptr(250, 200)); // move
      component.onPointerMove(ptr(280, 230));
      component.onPointerUp(ptr(280, 230));
      component.onPointerMove(ptr(500, 500)); // ignored after release
      expect(component.cropX).toBe(130);
      expect(component.cropY).toBe(110);
    });
  });

  describe('apply / cancel', () => {
    it('converts the display-space box to original-image pixels on apply', () => {
      component.onImageLoaded(); // box {100,80,300,240}, scale 2× on both axes
      let result: ImageCropResult | undefined;
      component.applied.subscribe((r) => (result = r));
      component.apply();
      expect(result).toEqual({ box: [200, 160, 800, 640] });
    });

    it('clamps the emitted box to the natural image bounds', () => {
      component.onImageLoaded();
      // Push the box to the far bottom-right; conversion must not exceed 1000×800.
      component.cropX = 400;
      component.cropY = 300;
      component.cropW = 100;
      component.cropH = 100;
      let result: ImageCropResult | undefined;
      component.applied.subscribe((r) => (result = r));
      component.apply();
      expect(result).toEqual({ box: [800, 600, 1000, 800] });
    });

    it('does not emit before the image dimensions are known', () => {
      const spy = vi.fn();
      component.applied.subscribe(spy); // onImageLoaded not called → displayW is 0
      component.apply();
      expect(spy).not.toHaveBeenCalled();
    });

    it('emits cancelled on cancel', () => {
      const spy = vi.fn();
      component.cancelled.subscribe(spy);
      component.cancel();
      expect(spy).toHaveBeenCalledTimes(1);
    });
  });
});
