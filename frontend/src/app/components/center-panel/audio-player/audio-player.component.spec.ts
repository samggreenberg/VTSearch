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

  let originalFetch: typeof globalThis.fetch;
  let originalCreate: typeof URL.createObjectURL;
  let originalRevoke: typeof URL.revokeObjectURL;

  beforeEach(async () => {
    // Single-fetch path: the component downloads the clip once and feeds the
    // bytes to the <audio> element via an object URL. jsdom implements neither
    // fetch's media pipeline nor createObjectURL usefully, so stub them.
    originalFetch = globalThis.fetch;
    originalCreate = URL.createObjectURL;
    originalRevoke = URL.revokeObjectURL;
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob([new Uint8Array([0])], { type: 'audio/wav' })),
    }) as unknown as typeof globalThis.fetch;
    URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    URL.revokeObjectURL = vi.fn();
    // jsdom logs "Not implemented" for these; keep the audio path quiet.
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);

    await TestBed.configureTestingModule({
      imports: [AudioPlayerComponent],
      providers: [...provideZoneless(), ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(AudioPlayerComponent);
    component = fixture.componentInstance;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
    vi.restoreAllMocks();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('downloads the audio once and drives the <audio> element from an object URL', async () => {
    const fakeAudio = {
      src: '',
      volume: 0,
      paused: true,
      play: () => Promise.resolve(),
      pause: () => {},
      load: () => {},
    } as unknown as HTMLAudioElement;
    // A canvas whose 2D context is unavailable, so the waveform path (which
    // needs decodeAudioData, absent in jsdom) short-circuits after the src is set.
    const fakeCanvas = {
      getContext: () => null,
      getBoundingClientRect: () => ({ width: 0 }) as DOMRect,
      width: 600,
      height: 120,
    } as unknown as HTMLCanvasElement;
    component.media = mockMedia;
    component.audioRef = { nativeElement: fakeAudio } as ElementRef<HTMLAudioElement>;
    component.canvasRef = { nativeElement: fakeCanvas } as ElementRef<HTMLCanvasElement>;

    await (component as unknown as { loadAudio(): Promise<void> }).loadAudio();

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/medias/1/audio', expect.anything());
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(fakeAudio.src).toBe('blob:mock-url');
    expect(component.audioSrc).toBe('blob:mock-url');
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
