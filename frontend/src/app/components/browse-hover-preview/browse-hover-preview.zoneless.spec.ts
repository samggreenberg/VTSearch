import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { BrowseHoverPreviewComponent, NowPlaying } from './browse-hover-preview.component';
import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { makeActiveContextStub } from '../../testing/mocks';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless canary + behaviour spec for the browse hover preview.
 *
 * The text path (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.6)
 * signalized `textContent`, written from the async paragraph `fetch().then()`
 * continuation — an un-patched microtask that must still schedule CD.
 *
 * The audio path (docs/plans/browse-audio-player.md, Phase 4) auditions the
 * hovered bin's clip on a dwell with no on-canvas UI of its own; it emits
 * `nowPlaying` for the top-left indicator (`browse-view.component`) to render.
 * Those transitions run from `setTimeout` callbacks that emit the output, so
 * they are the same zoneless-staleness oracle: assert after `settleZoneless`,
 * never a forced `detectChanges`.
 */
describe('BrowseHoverPreviewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<BrowseHoverPreviewComponent>;
  let resolveFetch: (body: unknown) => void;
  let originalFetch: typeof globalThis.fetch;
  let playSpy: ReturnType<typeof vi.spyOn>;

  // Mirrors the component's dwell constant; the waits below clear it with margin.
  const DWELL_MS = 200;

  function hoverEvent(mediaId: number): HexHoverEvent {
    return {
      cell: { q: 0, r: 0, cx: 0, cy: 0, count: 1, rep_id: mediaId },
      screenX: 100,
      screenY: 100,
    };
  }

  const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  beforeEach(async () => {
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveFetch = (body: unknown) =>
            resolve({ json: () => Promise.resolve(body) } as Response);
        }),
    ) as typeof globalThis.fetch;

    // jsdom has no real media pipeline; keep play()/load() quiet so the audio
    // path's audition kick-off doesn't spam "Not implemented".
    playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    const activeContextStub = makeActiveContextStub();
    // The audio hover path reads clip extents through the metadata cache; stub it
    // so the component constructs without pulling the real root service (and its
    // HttpClient dependency) into the test injector.
    const metadataStub: Partial<MediaMetadataCacheService> = {
      version$: of(0),
      get: () => undefined,
      ensureLoaded: () => {},
    };

    await configureZoneless({
      imports: [BrowseHoverPreviewComponent],
      providers: [
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: MediaMetadataCacheService, useValue: metadataStub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BrowseHoverPreviewComponent);
    fixture.componentRef.setInput('mediaType', 'text');
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
    fixture.destroy();
  });

  it('repaints the popup text when the paragraph fetch resolves, no manual detectChanges', async () => {
    // Production channel: a hover on a text cell triggers the paragraph fetch.
    fixture.componentRef.setInput('hover', hoverEvent(42));
    await settleZoneless(fixture);

    // While loading, the popup shows the placeholder.
    expect(fixture.nativeElement.querySelector('.hover-text')!.textContent).toContain('Loading');

    // The async `fetch().then()` continuation writes the `textContent` signal —
    // an un-bound microtask that must still schedule CD under zoneless.
    resolveFetch({ content: 'a perching bird sings at dawn' });
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.hover-text')!.textContent).toContain(
      'a perching bird sings at dawn',
    );
  });

  it('auditions an audio bin only after the dwell elapses, with no floating panel', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));

    fixture.componentRef.setInput('hover', hoverEvent(7));
    await settleZoneless(fixture);

    // Before the dwell: nothing auditioning yet. No panel/controls appear in
    // the canvas either way — the bin's own hover-enlarge is the only visual
    // feedback that a bin is under the cursor.
    expect(emitted).toBeUndefined();
    expect(playSpy).not.toHaveBeenCalled();
    expect(fixture.nativeElement.querySelector('.hover-player')).toBeNull();

    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    // After the dwell: playback starts and `nowPlaying` carries the clip's
    // waveform for the top-left indicator — flagged `loading` until the clip is
    // actually sounding — still no panel in the canvas.
    expect(playSpy).toHaveBeenCalled();
    // `progress` is null until a finite duration is known (jsdom has no media
    // pipeline, so it never is); the sweeping playhead stays hidden meanwhile.
    expect(emitted).toEqual({ mediaId: 7, waveUrl: '/api/medias/7/thumbnail', loading: true, progress: null });
    expect(fixture.nativeElement.querySelector('.hover-player')).toBeNull();
  });

  it('clears the loading flag once the clip starts sounding, and re-sets it on rebuffer', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));

    fixture.componentRef.setInput('hover', hoverEvent(9));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    // Auditioning starts in the loading state (fetch/decode not done yet).
    expect(emitted).toEqual({ mediaId: 9, waveUrl: '/api/medias/9/thumbnail', loading: true, progress: null });

    // The reused (never-mounted) audio element drives the buffering listeners.
    const audioEl = (fixture.componentInstance as unknown as { audioEl: HTMLAudioElement }).audioEl;

    // The element reports it's now playing → spinner clears.
    audioEl.dispatchEvent(new Event('playing'));
    await settleZoneless(fixture);
    expect(emitted).toEqual({ mediaId: 9, waveUrl: '/api/medias/9/thumbnail', loading: false, progress: null });

    // A stall to rebuffer mid-play → spinner returns.
    audioEl.dispatchEvent(new Event('waiting'));
    await settleZoneless(fixture);
    expect(emitted).toEqual({ mediaId: 9, waveUrl: '/api/medias/9/thumbnail', loading: true, progress: null });
  });

  it('debounces the dwell so sweeping across bins plays only the settled one', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));

    // Sweep 1 → 2 → 3 faster than the dwell; each hover re-arms it.
    fixture.componentRef.setInput('hover', hoverEvent(1));
    await wait(DWELL_MS / 2);
    fixture.componentRef.setInput('hover', hoverEvent(2));
    await wait(DWELL_MS / 2);
    fixture.componentRef.setInput('hover', hoverEvent(3));
    await settleZoneless(fixture);

    // Mid-sweep the dwell never completed, so nothing started playing.
    expect(emitted).toBeUndefined();

    // Settling on bin 3 lets the dwell fire → only bin 3 plays.
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);
    expect(emitted?.mediaId).toBe(3);
  });

  it('stops the audition and emits nowPlaying(null) as soon as the hover clears', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));

    fixture.componentRef.setInput('hover', hoverEvent(5));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);
    expect(emitted?.mediaId).toBe(5);

    // No hover-bridge grace period: there's no panel to travel onto, so the
    // clip stops the instant the bin is no longer hovered.
    fixture.componentRef.setInput('hover', null);
    await settleZoneless(fixture);
    expect(emitted).toBeNull();
  });
});
