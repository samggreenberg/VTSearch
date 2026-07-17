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
    // The media-change effect fires loadAudio() once the view query resolves;
    // drive it through the real channel (setInput + render), then drain the
    // async fetch → blob → object-URL chain.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
    await settleZoneless(fixture);

    const audio: HTMLAudioElement = fixture.nativeElement.querySelector('audio');
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/medias/1/audio', expect.anything());
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(audio.src).toBe('blob:mock-url');
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

  it('seeks into the clip window when clip extents arrive after the audio loaded', async () => {
    // Mirrors the video player: a windowed archive-member clip first renders as
    // a stub (no clip_start/clip_end), so (loadedmetadata) fires before the
    // extents are known. Batch hydration then enriches the same media id with
    // clip_start/clip_end on a later change of the `media` input; the player
    // must snap into the window (display-only seek; the whole member is served).
    const stub: Media = { id: 9, media_type: 'audio', filename: 'clip.aac', md5: 'abc', custom_metadata: {} };
    fixture.componentRef.setInput('media', stub);
    await settleZoneless(fixture);

    // jsdom has no media pipeline; give the real element a working currentTime.
    const audio: HTMLAudioElement = fixture.nativeElement.querySelector('audio');
    let currentTime = 0;
    Object.defineProperty(audio, 'currentTime', {
      configurable: true,
      get: () => currentTime,
      set: (v: number) => (currentTime = v),
    });

    // Audio loads against the stub: no clip window yet, so no seek.
    component.onLoadedMetadata();
    expect(audio.currentTime).toBe(0);

    // Batch hydration delivers the clip extents for the same id.
    const hydrated: Media = { ...stub, clip_start: 12, clip_end: 22 };
    fixture.componentRef.setInput('media', hydrated);
    await settleZoneless(fixture);
    expect(audio.currentTime).toBe(12);
  });
});
