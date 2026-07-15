import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MediaCropModalComponent, MediaCropResult } from './media-crop-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';

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
