import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MediaCropModalComponent, MediaCropResult } from './media-crop-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { blurKey, provideBlurStateStub } from '../../../testing/blur-state-stub';

/**
 * Drives the modal's confirm/crop view switching and the crop-region payloads it
 * forwards from the child overlays. The object-URL APIs are stubbed (jsdom lacks
 * them); the modal is instantiated but not rendered, so `ngOnInit` is invoked
 * directly to build `fileUrl` without pulling in the overlay children.
 */
describe('MediaCropModalComponent', () => {
  let component: MediaCropModalComponent;
  let fixture: ComponentFixture<MediaCropModalComponent>;
  const file = new File(['bytes'], 'clip.png', { type: 'image/png' });

  function build(mediaType: string): void {
    fixture = TestBed.createComponent(MediaCropModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('file', file);
    fixture.componentRef.setInput('mediaType', mediaType);
    component.ngOnInit();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [MediaCropModalComponent],
      providers: [...provideZoneless()],
    });
    URL.createObjectURL = vi.fn(() => 'blob:mock');
    URL.revokeObjectURL = vi.fn();
  });

  it('creates an object URL for the file on init', () => {
    build('image');
    expect(URL.createObjectURL).toHaveBeenCalledWith(file);
    expect(component.fileUrl).toBe('blob:mock');
  });

  it('revokes the object URL on destroy', () => {
    build('image');
    component.ngOnDestroy();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock');
  });

  describe('cropSupported', () => {
    it('is true for image and audio', () => {
      build('image');
      expect(component.cropSupported).toBe(true);
      build('audio');
      expect(component.cropSupported).toBe(true);
    });

    it('is false for other media types', () => {
      build('video');
      expect(component.cropSupported).toBe(false);
      build('');
      expect(component.cropSupported).toBe(false);
    });
  });

  describe('mediaItemWord (#5175)', () => {
    it("names one item of the example's media type for Toasty's hint", () => {
      build('image');
      expect(component.mediaItemWord).toBe('image');
      build('audio');
      expect(component.mediaItemWord).toBe('audio clip');
    });

    it('says "file" when the media type is unknown', () => {
      build('');
      expect(component.mediaItemWord).toBe('file');
    });
  });

  describe('view switching', () => {
    it('starts in the confirm view', () => {
      build('image');
      expect(component.view).toBe('confirm');
    });

    it('enters the cropping view when crop is supported', () => {
      build('audio');
      component.onOkButCrop();
      expect(component.view).toBe('cropping');
    });

    it('confirms directly (no crop view) when crop is unsupported', () => {
      build('video');
      let result: MediaCropResult | undefined;
      component.confirmed.subscribe((r) => (result = r));
      component.onOkButCrop();
      expect(component.view).toBe('confirm');
      expect(result).toEqual({ file });
    });

    it('returns to the confirm view when a crop overlay is cancelled', () => {
      build('image');
      component.onOkButCrop();
      component.onCropOverlayCancelled();
      expect(component.view).toBe('confirm');
    });
  });

  describe('emitted results', () => {
    it('confirms with the raw file on OK', () => {
      build('image');
      let result: MediaCropResult | undefined;
      component.confirmed.subscribe((r) => (result = r));
      component.onOk();
      expect(result).toEqual({ file });
    });

    it('forwards an image crop box as cropParams', () => {
      build('image');
      let result: MediaCropResult | undefined;
      component.confirmed.subscribe((r) => (result = r));
      component.onImageCropApplied({ box: [10, 20, 110, 220] });
      expect(result).toEqual({ file, cropParams: { box: [10, 20, 110, 220] } });
    });

    it('forwards an audio crop range as cropParams', () => {
      build('audio');
      let result: MediaCropResult | undefined;
      component.confirmed.subscribe((r) => (result = r));
      component.onAudioCropApplied({ start: 1.5, end: 4.25 });
      expect(result).toEqual({ file, cropParams: { start: 1.5, end: 4.25 } });
    });

    it('emits cancelled on cancel', () => {
      build('image');
      const spy = vi.fn();
      component.cancelled.subscribe(spy);
      component.onCancel();
      expect(spy).toHaveBeenCalledTimes(1);
    });
  });
});

/** Opened on a dataset image (Crop and Sort), the image shows its blur state (#5696). */
describe('MediaCropModalComponent (blur)', () => {
  const file = new File(['bytes'], 'clip.png', { type: 'image/png' });
  let fixture: ComponentFixture<MediaCropModalComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [MediaCropModalComponent],
      providers: [...provideZoneless(), provideBlurStateStub([blurKey('ds-1', 7)]).provider],
    });
    URL.createObjectURL = vi.fn(() => 'blob:mock');
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => fixture.destroy());

  async function render(mediaId: number | null): Promise<HTMLElement> {
    fixture = TestBed.createComponent(MediaCropModalComponent);
    fixture.componentRef.setInput('file', file);
    fixture.componentRef.setInput('mediaType', 'image');
    fixture.componentRef.setInput('mediaId', mediaId);
    fixture.componentRef.setInput('datasetId', 'ds-1');
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  it('blurs the preview and the crop target of a blurred dataset image', async () => {
    const el = await render(7);
    expect(el.querySelector('.preview img')?.classList.contains('vt-blurred')).toBe(true);

    const okButCrop = Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'OK but Crop')!;
    okButCrop.click();
    await settleZoneless(fixture);
    expect(el.querySelector('vt-image-crop-overlay img')?.classList.contains('vt-blurred')).toBe(true);
    // The image takes no pointer there; its drag surface is what peeks.
    expect(el.querySelector('.image-wrap')?.classList.contains('vt-blur-peek')).toBe(true);
  });

  it('leaves a sharp dataset image, and an uploaded file, sharp', async () => {
    expect((await render(8)).querySelector('.preview img')?.classList.contains('vt-blurred')).toBe(false);
    fixture.destroy();
    expect((await render(null)).querySelector('.preview img')?.classList.contains('vt-blurred')).toBe(false);
  });
});
