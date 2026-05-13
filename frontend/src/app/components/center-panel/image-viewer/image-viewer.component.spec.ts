import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ElementRef } from '@angular/core';
import { ImageViewerComponent, RegionBox } from './image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaItem } from '../../../models/api.models';

describe('ImageViewerComponent', () => {
  let component: ImageViewerComponent;
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: MediaItem = {
    id: 2,
    type: 'image',
    filename: 'test.png',
    md5: 'def456',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(ImageViewerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set imageSrc when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageSrc).toBe('/api/medias/2/image');
  });

  it('should hide image until loaded to prevent flash of old image', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageReady).toBeFalse();

    component.onImageLoad();
    expect(component.imageReady).toBeTrue();
  });

  it('should reset imageReady when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    component.onImageLoad();
    expect(component.imageReady).toBeTrue();

    const nextMedia = { ...mockMedia, id: 3, filename: 'next.png' };
    component.media = nextMedia;
    component.ngOnChanges({
      media: { currentValue: nextMedia, previousValue: mockMedia, firstChange: false, isFirstChange: () => false },
    });
    expect(component.imageReady).toBeFalse();
  });

  it('should show image on error to avoid stuck black screen', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageReady).toBeFalse();

    component.onImageError();
    expect(component.imageReady).toBeTrue();
  });

  it('should render image element', () => {
    component.media = mockMedia;
    component.imageSrc = '/api/medias/2/image';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('img')).toBeTruthy();
  });

  it('should expose control methods and properties', () => {
    expect(component.rotateLeft).toBeDefined();
    expect(component.rotateRight).toBeDefined();
    expect(component.resetView).toBeDefined();
    expect(component.onZoomInput).toBeDefined();
    expect(component.minZoom).toBe(1);
    expect(component.maxZoom).toBe(5);
  });

  it('should reset view', () => {
    component.zoom = 2;
    component.rotation = 90;
    component.resetView();
    expect(component.zoom).toBe(1);
    expect(component.rotation).toBe(0);
  });

  it('should rotate left', () => {
    component.rotateLeft();
    expect(component.rotation).toBe(-90);
  });

  it('should rotate right', () => {
    component.rotateRight();
    expect(component.rotation).toBe(90);
  });

  it('should generate transform string', () => {
    expect(component.imageTransform).toContain('scale(1)');
    expect(component.imageTransform).toContain('rotate(0deg)');
  });

  /**
   * Item 10 of the v2 patch-embedder plan (docs/plans/patch-embedder.md):
   * pure-function coverage for the screen↔image coordinate transform under
   * non-trivial pan / zoom / rotate. The transform is what lets the user draw
   * a box that stays anchored on the right image pixels regardless of how the
   * viewport is currently transformed; this is the math that backs that.
   */
  describe('screenToImageNormalized (coord transform)', () => {
    // Helper: wire up a 100×100 wrap centred at screen (50, 50) with a 100×100
    // rendered image. Lets each test focus on the transform math, not on DOM
    // plumbing.
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW = 100;
      component.renderedH = 100;
      component.wrapRef = {
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>;
    }

    function makeEvent(clientX: number, clientY: number): MouseEvent {
      return { clientX, clientY } as MouseEvent;
    }

    it('returns null when the image has not been laid out yet', () => {
      // renderedW/H still 0 → no transform possible.
      expect(component.screenToImageNormalized(makeEvent(50, 50))).toBeNull();
    });

    it('maps the wrap centre to the image centre at identity', () => {
      setupWrap(component);
      const local = component.screenToImageNormalized(makeEvent(50, 50));
      expect(local!.x).toBeCloseTo(0.5, 6);
      expect(local!.y).toBeCloseTo(0.5, 6);
    });

    it('maps corners correctly at identity', () => {
      setupWrap(component);
      const tl = component.screenToImageNormalized(makeEvent(0, 0))!;
      const br = component.screenToImageNormalized(makeEvent(100, 100))!;
      expect(tl.x).toBeCloseTo(0, 6);
      expect(tl.y).toBeCloseTo(0, 6);
      expect(br.x).toBeCloseTo(1, 6);
      expect(br.y).toBeCloseTo(1, 6);
    });

    it('compensates for zoom — screen offsets shrink in image coords as zoom grows', () => {
      setupWrap(component);
      component.zoom = 2;
      // 25px right of centre at 2× zoom should be 12.5px in image coords =
      // 0.125 normalised, so image x = 0.625.
      const local = component.screenToImageNormalized(makeEvent(75, 50))!;
      expect(local.x).toBeCloseTo(0.625, 6);
      expect(local.y).toBeCloseTo(0.5, 6);
    });

    it('compensates for pan — translating the image shifts the inferred image coords', () => {
      setupWrap(component);
      component.zoom = 2;
      // Image translated 20px right; a click at screen 70 used to map to image
      // 0.7 at zoom 2 (50px = 0.5 + 20/100/2 * 2... see math). Concretely:
      // dx = 70 - 50 - 20 = 0; sx = 0; image x = 0.5.
      component.panX = 20;
      const local = component.screenToImageNormalized(makeEvent(70, 50))!;
      expect(local.x).toBeCloseTo(0.5, 6);
      expect(local.y).toBeCloseTo(0.5, 6);
    });

    it('inverts rotation — a click on the screen-right edge maps to the image-top edge at 90° CW', () => {
      setupWrap(component);
      component.rotation = 90;
      // Positive rotation rotates the image clockwise, so screen-right is image-top.
      const local = component.screenToImageNormalized(makeEvent(100, 50))!;
      expect(local.x).toBeCloseTo(0.5, 5);
      expect(local.y).toBeCloseTo(0, 5);
    });

    it('combines pan + zoom + rotate self-consistently', () => {
      setupWrap(component);
      component.zoom = 2;
      component.panX = 10;
      component.panY = -5;
      component.rotation = 90;
      // Map a couple of points and verify the transform is invertible:
      // taking two points on screen and rotating by the matching angle, the
      // image-coord differences should respect rotation (a screen-x delta
      // becomes an image-y delta at +90°).
      const a = component.screenToImageNormalized(makeEvent(50, 50))!;
      const b = component.screenToImageNormalized(makeEvent(60, 50))!;
      // Screen Δx = +10 → image Δy = -10/zoom/renderedH = -0.05 (the inverse
      // rotation maps +x → -y); Δx in image coords ≈ 0.
      expect(b.x - a.x).toBeCloseTo(0, 5);
      expect(b.y - a.y).toBeCloseTo(-0.05, 5);
    });
  });

  /**
   * Item 11 of the v2 patch-embedder plan: a box drawn at one zoom level
   * stays anchored on the same image pixels when the user zooms in/out.
   * The box is stored in normalised image coordinates and the CSS overlay
   * lives inside `.region-stage` which is rotated/scaled with the image
   * transform, so the box should remain visually identical relative to the
   * image regardless of zoom.
   */
  describe('region box coord stability', () => {
    it('does not mutate the box coords when zoom changes', () => {
      component.regionBox = [0.1, 0.2, 0.5, 0.6];
      component.zoom = 2;
      expect(component.regionBox).toEqual([0.1, 0.2, 0.5, 0.6]);
      component.zoom = 4;
      expect(component.regionBox).toEqual([0.1, 0.2, 0.5, 0.6]);
      component.zoom = 1;
      expect(component.regionBox).toEqual([0.1, 0.2, 0.5, 0.6]);
    });

    it('keeps regionBoxStyle (percent-of-stage) stable across zoom changes', () => {
      component.regionBox = [0.1, 0.2, 0.5, 0.6];
      const before = component.regionBoxStyle;
      component.zoom = 3;
      expect(component.regionBoxStyle).toEqual(before);
      component.rotation = 45;
      expect(component.regionBoxStyle).toEqual(before);
    });

    it('returns null style when no box is drawn', () => {
      component.regionBox = null;
      expect(component.regionBoxStyle).toBeNull();
    });
  });

  /**
   * Item 13 of the v2 patch-embedder plan: a subsequent zero-area Shift-drag
   * click does not clear an already-drawn box. (Before the v2 sticky-armed-
   * state work this was a real bug: tooSmall release nuked the prior box.)
   */
  describe('region box preservation on zero-area Shift-drag', () => {
    function setupWrap(component: ImageViewerComponent) {
      component.renderedW = 100;
      component.renderedH = 100;
      component.wrapRef = {
        nativeElement: {
          getBoundingClientRect: () => ({
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            right: 100,
            bottom: 100,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          }),
        } as unknown as HTMLDivElement,
      } as ElementRef<HTMLDivElement>;
    }

    it('keeps the prior box when a Shift-click resolves to zero area', () => {
      setupWrap(component);
      const original: RegionBox = [0.2, 0.3, 0.6, 0.7];
      component.regionBox = original;
      component.shiftHeld = true;

      let lastEmitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (lastEmitted = v));

      // Click at (10, 10) with Shift held — mousedown then mouseup at same point.
      const ev: MouseEvent = {
        button: 0,
        clientX: 10,
        clientY: 10,
        preventDefault: () => {},
      } as unknown as MouseEvent;
      component.onMouseDown(ev);
      // During the click regionBox transiently becomes the zero-area anchor;
      // mouseup with no motion should restore the original box.
      (component as unknown as { onWindowMouseUp: () => void }).onWindowMouseUp();

      expect(component.regionBox).toEqual(original);
      // Restored to a state the parent already knew; no emit needed.
      expect(lastEmitted).toBeUndefined();
    });

    it('leaves the box null when the canvas was already empty and the click is zero-area', () => {
      setupWrap(component);
      component.regionBox = null;
      component.shiftHeld = true;

      const ev: MouseEvent = {
        button: 0,
        clientX: 25,
        clientY: 25,
        preventDefault: () => {},
      } as unknown as MouseEvent;
      component.onMouseDown(ev);
      (component as unknown as { onWindowMouseUp: () => void }).onWindowMouseUp();

      expect(component.regionBox).toBeNull();
    });
  });

  describe('armed-confirm cancel routing', () => {
    it('emits armedConfirmCanceled instead of clearing the box when Esc is pressed while armed', () => {
      component.regionBox = [0.1, 0.2, 0.5, 0.6];
      component.pendingBadConfirm = true;
      let canceled = false;
      let cleared = false;
      component.armedConfirmCanceled.subscribe(() => (canceled = true));
      component.regionBoxChange.subscribe((v) => {
        if (v === null) cleared = true;
      });
      const esc = new KeyboardEvent('keydown', { key: 'Escape' });
      (component as unknown as { onWindowKeyDown: (e: KeyboardEvent) => void }).onWindowKeyDown(esc);
      expect(canceled).toBeTrue();
      expect(cleared).toBeFalse();
      expect(component.regionBox).toEqual([0.1, 0.2, 0.5, 0.6]);
    });

    it('clears the box on Esc when no armed confirm is pending', () => {
      component.regionBox = [0.1, 0.2, 0.5, 0.6];
      component.pendingBadConfirm = false;
      let emitted: RegionBox | null | undefined = undefined;
      component.regionBoxChange.subscribe((v) => (emitted = v));
      const esc = new KeyboardEvent('keydown', { key: 'Escape' });
      (component as unknown as { onWindowKeyDown: (e: KeyboardEvent) => void }).onWindowKeyDown(esc);
      expect(component.regionBox).toBeNull();
      expect(emitted).toBeNull();
    });
  });
});
