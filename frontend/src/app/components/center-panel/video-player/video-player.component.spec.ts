import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VideoPlayerComponent } from './video-player.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('VideoPlayerComponent', () => {
  let component: VideoPlayerComponent;
  let fixture: ComponentFixture<VideoPlayerComponent>;

  const mockMedia: Media = {
    id: 3,
    media_type: 'video',
    filename: 'test.mp4',
    md5: 'ghi789',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VideoPlayerComponent],
      providers: [...provideZoneless(), ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(VideoPlayerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set videoSrc when media changes', () => {
    fixture.componentRef.setInput('media', mockMedia);
    TestBed.tick();
    expect(component.videoSrc()).toBe('/api/medias/3/video');
  });

  it('should render video element', async () => {
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
    const video = fixture.nativeElement.querySelector('video');
    expect(video).toBeTruthy();
    expect(video.hasAttribute('controls')).toBe(true);
    expect(video.hasAttribute('loop')).toBe(true);
  });

  it('seeks into the clip window when clip extents arrive after the video loaded', async () => {
    // A clipped media first renders as a stub (no clip_start/clip_end), so the
    // (loadedmetadata) event fires before the extents are known. The batch
    // response then enriches the same media id with clip_start/clip_end on a
    // later change of the `media` input. The player must snap into the window
    // at that point.
    const stub: Media = { id: 7, media_type: 'video', filename: 'clip.mp4', md5: 'abc', custom_metadata: {} };
    fixture.componentRef.setInput('media', stub);
    await settleZoneless(fixture);

    // jsdom has no media pipeline; give the real element a working currentTime.
    const video: HTMLVideoElement = fixture.nativeElement.querySelector('video');
    let currentTime = 0;
    Object.defineProperty(video, 'currentTime', {
      configurable: true,
      get: () => currentTime,
      set: (v: number) => (currentTime = v),
    });

    // Video loads against the stub: no clip window yet, so no seek.
    component.onLoadedMetadata();
    expect(video.currentTime).toBe(0);

    // Batch hydration delivers the clip extents for the same id.
    const hydrated: Media = { ...stub, clip_start: 5, clip_end: 10 };
    fixture.componentRef.setInput('media', hydrated);
    await settleZoneless(fixture);
    expect(video.currentTime).toBe(5);
  });

  describe('letterbox crop (clip_box)', () => {
    // jsdom has no media pipeline, so videoWidth/videoHeight read 0; give the
    // element the natural size (loadedmetadata) would report. The CSS itself
    // can't be checked here (jsdom drops object-view-box), only the value the
    // template binds; utils/clip-box.spec.ts covers the box-to-CSS rule.
    async function loadAt(media: Media, width: number, height: number): Promise<HTMLVideoElement> {
      fixture.componentRef.setInput('media', media);
      await settleZoneless(fixture);
      const video: HTMLVideoElement = fixture.nativeElement.querySelector('video');
      Object.defineProperty(video, 'videoWidth', { configurable: true, get: () => width });
      Object.defineProperty(video, 'videoHeight', { configurable: true, get: () => height });
      component.onLoadedMetadata();
      return video;
    }

    it('crops the picture to clip_box once the frame size is known', async () => {
      const boxed: Media = { ...mockMedia, clip_box: [0, 60, 640, 420] };
      fixture.componentRef.setInput('media', boxed);
      await settleZoneless(fixture);
      expect(component.viewBox()).toBeNull();

      await loadAt(boxed, 640, 480);
      expect(component.viewBox()).toBe('inset(60px 0px 60px 0px)');
    });

    it('applies a clip_box that batch hydration delivers after the video loaded', async () => {
      await loadAt(mockMedia, 640, 480);
      expect(component.viewBox()).toBeNull();

      fixture.componentRef.setInput('media', { ...mockMedia, clip_box: [100, 120, 580, 420] });
      await settleZoneless(fixture);
      expect(component.viewBox()).toBe('inset(120px 60px 60px 100px)');
    });

    it('leaves a unit without a clip_box uncropped', async () => {
      await loadAt(mockMedia, 640, 480);
      expect(component.viewBox()).toBeNull();
    });

    it('drops the crop when the player moves to another item', async () => {
      await loadAt({ ...mockMedia, clip_box: [0, 60, 640, 420] }, 640, 480);
      expect(component.viewBox()).not.toBeNull();

      // The next item's box must not be clamped against the last item's frame.
      fixture.componentRef.setInput('media', { ...mockMedia, id: 4, clip_box: [0, 10, 320, 230] });
      await settleZoneless(fixture);
      expect(component.viewBox()).toBeNull();
    });
  });
});
