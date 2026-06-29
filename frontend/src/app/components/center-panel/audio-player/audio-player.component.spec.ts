import { ElementRef } from '@angular/core';
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

  it('seeks into the clip window when clip extents arrive after the audio loaded', () => {
    // Mirrors the video player: a windowed archive-member clip first renders as
    // a stub (no clip_start/clip_end), so (loadedmetadata) fires before the
    // extents are known. Batch hydration then enriches the same media id with
    // clip_start/clip_end on a later ngOnChanges; the player must snap into the
    // window (display-only seek; the whole member is served).
    const fakeAudio = {
      volume: 0,
      currentTime: 0,
      paused: true,
      play: () => Promise.resolve(),
      pause: () => {},
      removeAttribute: () => {},
      load: () => {},
    } as unknown as HTMLAudioElement;
    component.audioRef = { nativeElement: fakeAudio } as ElementRef<HTMLAudioElement>;

    const stub: Media = { id: 9, media_type: 'audio', filename: 'clip.aac', md5: 'abc', custom_metadata: {} };
    component.media = stub;
    component.ngOnChanges({
      media: { currentValue: stub, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    // Audio loads against the stub: no clip window yet, so no seek.
    component.onLoadedMetadata();
    expect(fakeAudio.currentTime).toBe(0);

    // Batch hydration delivers the clip extents for the same id.
    const hydrated: Media = { ...stub, clip_start: 12, clip_end: 22 };
    component.media = hydrated;
    component.ngOnChanges({
      media: { currentValue: hydrated, previousValue: stub, firstChange: false, isFirstChange: () => false },
    });
    expect(fakeAudio.currentTime).toBe(12);

    component.ngOnDestroy();
  });
});
