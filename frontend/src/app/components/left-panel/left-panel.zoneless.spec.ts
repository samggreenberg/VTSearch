import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SimpleChange } from '@angular/core';

import { LeftPanelComponent } from './left-panel.component';
import type { Media } from '../../models/api.models';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the left panel
 * (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.7). Phase 2.7 signalized
 * `mediaTypeName` (and `textSortAvailable`), which were plain template-bound
 * fields written from a constructor `effect()` that reacts to late-arriving
 * media-type metadata — an effect-into-plain-field write that does NOT repaint
 * the view under zoneless (Recipe F).
 *
 * The test runs under a zoneless `TestBed`, lets the media-type metadata arrive
 * late through the production channel (the resource's HTTP response), and asserts
 * the grid header upgrades from the capitalized fallback to the display name with
 * NO manual `detectChanges()`.
 */
describe('LeftPanelComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<LeftPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [LeftPanelComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LeftPanelComponent);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({});
    });
    fixture.destroy();
  });

  it('upgrades the grid header when media-type metadata arrives late, no manual detectChanges', async () => {
    // Issue the media-types resource GET (its loader runs in a root effect).
    TestBed.tick();
    await settleZoneless(fixture);

    // The grid populates before the metadata resolves, so the header first shows
    // the capitalized fallback derived in the sync `ngOnChanges` path.
    fixture.componentInstance.medias = [{ id: 1, media_type: 'audio' } as Media];
    fixture.componentInstance.ngOnChanges({
      medias: new SimpleChange(undefined, fixture.componentInstance.medias, false),
    });
    await settleZoneless(fixture);

    const header = fixture.nativeElement.querySelector('.images-header-title');
    expect(header!.textContent).toContain('Audios');

    // Production channel: the media-types metadata arrives late with a custom
    // display name. The deriving `effect()` writes the `mediaTypeName` signal —
    // an effect-into-signal write that must repaint the header under zoneless.
    httpMock
      .expectOne((r) => r.url.includes('/api/media-types'))
      .flush({ media_types: [{ type_id: 'audio', name: 'Sound Clips' }] });
    await settleResource();
    await settleZoneless(fixture);

    expect(header!.textContent).toContain('Sound Clips');
  });
});
