import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AudioPlayerComponent } from './audio-player.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('AudioPlayerComponent', () => {
  let component: AudioPlayerComponent;
  let fixture: ComponentFixture<AudioPlayerComponent>;

  const mockMedia: Media = {
    id: 1,
    media_type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AudioPlayerComponent],
      providers: [...provideZoneless(), ActiveContextService],
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

  it('should render canvas and audio elements', async () => {
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('canvas')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('audio')).toBeTruthy();
  });

  it('should have controls on audio element', async () => {
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
    const audio = fixture.nativeElement.querySelector('audio');
    expect(audio.hasAttribute('controls')).toBe(true);
    expect(audio.hasAttribute('loop')).toBe(true);
  });
});
