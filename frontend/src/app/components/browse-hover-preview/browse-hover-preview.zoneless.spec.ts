import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { BrowseHoverPreviewComponent } from './browse-hover-preview.component';
import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless canary + behaviour spec for the browse hover preview.
 *
 * The text path (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.6)
 * signalized `textContent`, written from the async paragraph `fetch().then()`
 * continuation — an un-patched microtask that must still schedule CD.
 *
 * The audio path (docs/plans/browse-audio-player.md, Phase 2) opens an anchored
 * player on a dwell and keeps it open while the cursor is on it. Those
 * transitions run from `setTimeout` callbacks that write the `player` signal, so
 * they are the same zoneless-staleness oracle: assert on rendered DOM after
 * `settleZoneless`, never a forced `detectChanges`.
 */
describe('BrowseHoverPreviewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<BrowseHoverPreviewComponent>;
  let resolveFetch: (body: unknown) => void;
  let originalFetch: typeof globalThis.fetch;

  // Dwell + hide-grace mirror the component's constants; the waits below clear
  // them with margin.
  const DWELL_MS = 200;
  const GRACE_MS = 140;

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
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    const activeContextStub: Partial<ActiveContextService> = {
      mediaUrl: (p: string) => p,
    };
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

  it('opens the anchored audio player only after the dwell elapses', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    fixture.componentRef.setInput('hover', hoverEvent(7));
    await settleZoneless(fixture);

    const root = fixture.nativeElement as HTMLElement;
    // Before the dwell: no player, no "sound from nowhere".
    expect(root.querySelector('.hover-player')).toBeNull();

    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    // After the dwell: the anchored player mounts with the bin's waveform and an
    // <audio> element wired to the clip.
    const panel = root.querySelector('.hover-player');
    expect(panel).not.toBeNull();
    expect(root.querySelector('.hover-player-wave')!.getAttribute('src')).toContain(
      '/api/medias/7/thumbnail',
    );
    expect(root.querySelector('audio')!.getAttribute('src')).toContain('/api/medias/7/audio');
  });

  it('debounces the dwell so sweeping across bins opens only the settled one', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    const root = fixture.nativeElement as HTMLElement;

    // Sweep 1 → 2 → 3 faster than the dwell; each hover re-arms it.
    fixture.componentRef.setInput('hover', hoverEvent(1));
    await wait(DWELL_MS / 2);
    fixture.componentRef.setInput('hover', hoverEvent(2));
    await wait(DWELL_MS / 2);
    fixture.componentRef.setInput('hover', hoverEvent(3));
    await settleZoneless(fixture);

    // Mid-sweep the dwell never completed, so no player opened.
    expect(root.querySelector('.hover-player')).toBeNull();

    // Settling on bin 3 lets the dwell fire → only bin 3's player opens.
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);
    expect(root.querySelector('audio')!.getAttribute('src')).toContain('/api/medias/3/audio');
  });

  it('bridges: keeps the player open when the cursor moves onto it, closes on leave', async () => {
    fixture.componentRef.setInput('mediaType', 'audio');
    fixture.componentRef.setInput('hover', hoverEvent(5));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    const root = fixture.nativeElement as HTMLElement;
    const panel = root.querySelector('.hover-player') as HTMLElement;
    expect(panel).not.toBeNull();

    // Cursor leaves the bin (hover → null) but bridges onto the panel within the
    // grace window: the player must stay open so its controls are reachable.
    fixture.componentRef.setInput('hover', null);
    panel.dispatchEvent(new MouseEvent('mouseenter'));
    await wait(GRACE_MS + 60);
    await settleZoneless(fixture);
    expect(root.querySelector('.hover-player')).not.toBeNull();

    // Leaving the panel closes it.
    panel.dispatchEvent(new MouseEvent('mouseleave'));
    await wait(GRACE_MS + 60);
    await settleZoneless(fixture);
    expect(root.querySelector('.hover-player')).toBeNull();
  });
});
