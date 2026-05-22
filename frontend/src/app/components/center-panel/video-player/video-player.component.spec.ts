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
});
