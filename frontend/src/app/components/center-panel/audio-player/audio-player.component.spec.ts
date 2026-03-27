import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AudioPlayerComponent } from './audio-player.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaItem } from '../../../models/api.models';

describe('AudioPlayerComponent', () => {
  let component: AudioPlayerComponent;
  let fixture: ComponentFixture<AudioPlayerComponent>;

  const mockMedia: MediaItem = {
    id: 1,
    type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AudioPlayerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(AudioPlayerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set audioSrc when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.audioSrc).toBe('/api/medias/1/audio');
  });

  it('should render canvas and audio elements', () => {
    component.media = mockMedia;
    component.audioSrc = '/api/medias/1/audio';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('canvas')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('audio')).toBeTruthy();
  });

  it('should have controls on audio element', () => {
    component.media = mockMedia;
    component.audioSrc = '/api/medias/1/audio';
    fixture.detectChanges();
    const audio = fixture.nativeElement.querySelector('audio');
    expect(audio.hasAttribute('controls')).toBeTrue();
    expect(audio.hasAttribute('loop')).toBeTrue();
  });
});
