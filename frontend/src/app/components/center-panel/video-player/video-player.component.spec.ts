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
});
