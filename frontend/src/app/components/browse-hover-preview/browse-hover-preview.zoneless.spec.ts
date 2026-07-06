import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { BrowseHoverPreviewComponent } from './browse-hover-preview.component';
import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the browse hover preview
 * (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.6). Phase 2.6 signalized
 * `textContent`, which is written from the async paragraph `fetch().then()`
 * continuation — an un-patched microtask that does NOT schedule CD for a plain
 * field under zoneless.
 *
 * The test runs under a zoneless `TestBed`, drives the text load through the
 * production channel (a hover input change + a resolving `fetch`) with NO manual
 * `detectChanges()`, then asserts the rendered popup text.
 */
describe('BrowseHoverPreviewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<BrowseHoverPreviewComponent>;
  let resolveFetch: (body: unknown) => void;
  let originalFetch: typeof globalThis.fetch;

  function hoverEvent(mediaId: number): HexHoverEvent {
    return {
      cell: { q: 0, r: 0, cx: 0, cy: 0, count: 1, rep_id: mediaId },
      screenX: 100,
      screenY: 100,
    };
  }

  beforeEach(async () => {
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveFetch = (body: unknown) =>
            resolve({ json: () => Promise.resolve(body) } as Response);
        }),
    ) as typeof globalThis.fetch;

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
});
