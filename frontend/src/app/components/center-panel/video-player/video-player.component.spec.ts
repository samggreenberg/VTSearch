import { ElementRef } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VideoPlayerComponent } from './video-player.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';

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
      providers: [ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(VideoPlayerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set videoSrc when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.videoSrc).toBe('/api/medias/3/video');
  });

  it('should render video element', () => {
    component.media = mockMedia;
    component.videoSrc = '/api/medias/3/video';
    fixture.detectChanges();
    const video = fixture.nativeElement.querySelector('video');
    expect(video).toBeTruthy();
    expect(video.hasAttribute('controls')).toBeTrue();
    expect(video.hasAttribute('loop')).toBeTrue();
  });

  it('seeks into the clip window when clip extents arrive after the video loaded', () => {
    // A clipped media first renders as a stub (no clip_start/clip_end), so the
    // (loadedmetadata) event fires before the extents are known. The batch
    // response then enriches the same media id with clip_start/clip_end on a
    // later ngOnChanges. The player must snap into the window at that point.
    const fakeVideo = {
      volume: 0,
      currentTime: 0,
      paused: true,
      play: () => Promise.resolve(),
      pause: () => {},
    } as unknown as HTMLVideoElement;
    component.videoRef = { nativeElement: fakeVideo } as ElementRef<HTMLVideoElement>;

    const stub: Media = { id: 7, media_type: 'video', filename: 'clip.mp4', md5: 'abc', custom_metadata: {} };
    component.media = stub;
    component.ngOnChanges({
      media: { currentValue: stub, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    // Video loads against the stub: no clip window yet, so no seek.
    component.onLoadedMetadata();
    expect(fakeVideo.currentTime).toBe(0);

    // Batch hydration delivers the clip extents for the same id.
    const hydrated: Media = { ...stub, clip_start: 5, clip_end: 10 };
    component.media = hydrated;
    component.ngOnChanges({
      media: { currentValue: hydrated, previousValue: stub, firstChange: false, isFirstChange: () => false },
    });
    expect(fakeVideo.currentTime).toBe(5);

    component.ngOnDestroy();
  });
});
