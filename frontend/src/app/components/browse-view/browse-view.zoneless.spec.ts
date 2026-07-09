import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, Router } from '@angular/router';
import { EMPTY, Subject, of } from 'rxjs';
import { signal } from '@angular/core';

import { BrowseViewComponent } from './browse-view.component';
import { ProjectionApiService } from '../../services/projection-api.service';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import { DetectorsRegistryApiService } from '../../services/detectors-registry-api.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { MediasApiService } from '../../services/medias-api.service';
import { VtDialogService } from '../../services/dialog.service';
import { ToastService } from '../../services/toast.service';
import type { ProjectionMeta } from '../../models/projection.models';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the VTSBrowse view
 * (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.6). Phase 2.6 signalized
 * the 13 template-bound fields browse-view writes from its async subscribes, the
 * projection build poller, and the settings effect (`status`, `errorMessage`,
 * `meta`, …). This drives the projection-load subscribe to an error from
 * outside any bound handler and asserts the `@switch (status())` view repaints
 * to the error state with the message — with NO manual `detectChanges()`.
 */
describe('BrowseViewComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<BrowseViewComponent>;
  let metaSubject: Subject<ProjectionMeta>;

  beforeEach(async () => {
    metaSubject = new Subject<ProjectionMeta>();

    const noop = () => {};
    const projectionStub: Partial<ProjectionApiService> = {
      getMeta: () => metaSubject.asObservable(),
    };
    const tileCacheStub: Partial<TileCacheService> = {
      setSubset: noop,
      setProjectionId: noop,
      setContentVersion: noop,
      clear: noop,
    };
    const activeContextStub: Partial<ActiveContextService> = {
      pair$: EMPTY,
      datasetId: '',
      modelId: '',
      setActive: noop,
      mediaUrl: (p: string) => p,
    };
    const datasetsStub: Partial<DatasetsRegistryApiService> = {
      getStatus: () => of({ display_name: 'Canary DS', media_type: 'audio' }) as ReturnType<
        DatasetsRegistryApiService['getStatus']
      >,
    };
    const detectorsStub: Partial<DetectorsRegistryApiService> = {
      releasePositivesBrowse: () =>
        of(undefined) as unknown as ReturnType<
          DetectorsRegistryApiService['releasePositivesBrowse']
        >,
    };
    const settingsStub: Partial<SettingsStateService> = {
      // Settings resolve (as they do in production): the browse view holds its
      // first projection load until settings + media type are in, so it applies
      // the saved per-media display prefs before the first canvas fit.
      settingsSignal: signal({}) as unknown as SettingsStateService['settingsSignal'],
      error: signal(null) as unknown as SettingsStateService['error'],
      load: noop,
      update: () => of({}) as ReturnType<SettingsStateService['update']>,
    };
    const subsetStub: Partial<BrowseSubsetService> = {
      take: () => null,
      markReturningToFind: noop,
    };
    const mediasStub: Partial<MediasApiService> = {};
    const routeStub = {
      snapshot: {
        queryParamMap: { get: () => null },
        paramMap: { get: () => '' },
      },
    } as unknown as ActivatedRoute;
    const routerStub = { navigate: () => Promise.resolve(true) } as unknown as Router;

    await configureZoneless({
      imports: [BrowseViewComponent],
      providers: [
        { provide: ProjectionApiService, useValue: projectionStub },
        { provide: TileCacheService, useValue: tileCacheStub },
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: DatasetsRegistryApiService, useValue: datasetsStub },
        { provide: DetectorsRegistryApiService, useValue: detectorsStub },
        { provide: SettingsStateService, useValue: settingsStub },
        { provide: BrowseSubsetService, useValue: subsetStub },
        { provide: MediasApiService, useValue: mediasStub },
        { provide: VtDialogService, useValue: {} },
        { provide: ToastService, useValue: { error: () => {}, success: () => {} } },
        { provide: ActivatedRoute, useValue: routeStub },
        { provide: Router, useValue: routerStub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BrowseViewComponent);
  });

  afterEach(() => fixture.destroy());

  it('repaints to the error state when the projection load subscribe errors, no manual detectChanges', async () => {
    await settleZoneless(fixture);

    // ngOnInit's loadProjection has issued getMeta but it hasn't emitted: the
    // view is in the `loading` state.
    expect(fixture.nativeElement.querySelector('.browse-status-message')!.textContent).toContain(
      'Loading map',
    );
    expect(fixture.nativeElement.querySelector('.browse-status-error')).toBeNull();

    // Production channel: the getMeta subscribe errors (non-404), writing the
    // `status` + `errorMessage` signals from an async callback.
    metaSubject.error({ status: 500, error: { message: 'projection exploded' } });
    await settleZoneless(fixture);

    const errEl = fixture.nativeElement.querySelector('.browse-status-error');
    expect(errEl).not.toBeNull();
    expect(errEl!.textContent).toContain('projection exploded');
  });

  it('covers the canvas until it reports its opening view is painted', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;

    // On entry the canvas is covered so the user never sees a half-loaded grid.
    expect(component.revealed()).toBe(false);
    // The cover names what it's loading. The canary DS is audio, which now
    // tiles as waveform-thumbnail squares, so the cover reads "Loading
    // thumbnails…"; a pure-density type (text) would read "Loading map…".
    expect(component.preloadMessage).toBe('Loading thumbnails…');

    // The canvas signals its opening view is fully painted → uncover it.
    component.onCanvasFirstView();
    expect(component.revealed()).toBe(true);
  });

  it('engages region-draw while Shift is held, releasing on keyup and blur', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;

    // Idle: neither the GUI toggle nor Shift is active.
    expect(component.shiftHeld()).toBe(false);
    expect(component.regionDrawActive).toBe(false);

    // Holding Shift previews the region-select gesture (button + cursor).
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    expect(component.shiftHeld()).toBe(true);
    expect(component.regionDrawActive).toBe(true);

    // Releasing Shift disengages it again.
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Shift' }));
    expect(component.shiftHeld()).toBe(false);
    expect(component.regionDrawActive).toBe(false);

    // A window blur (alt-tab) drops a stuck Shift so the view never strands.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    expect(component.shiftHeld()).toBe(true);
    window.dispatchEvent(new Event('blur'));
    expect(component.shiftHeld()).toBe(false);

    // The GUI toggle keeps engaging region-draw independently of Shift.
    component.toggleMarqueeMode();
    expect(component.regionDrawActive).toBe(true);
  });

  it('drives preview-audio volume from the toolbar: slider lowers the level and mute round-trips', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;

    // Settings resolved to {} → volume defaults to full.
    expect(component.volume()).toBe(1);

    // Dragging the slider lowers the level (this signal feeds the `[volume]`
    // inputs on the hover-preview + bin-popup, quieting a playing clip live).
    component.onVolumeInput({ target: { value: '0.3' } } as unknown as Event);
    expect(component.volume()).toBeCloseTo(0.3);

    // Muting drops to 0; unmuting restores the pre-mute level, not full volume.
    component.toggleMute();
    expect(component.volume()).toBe(0);
    component.toggleMute();
    expect(component.volume()).toBeCloseTo(0.3);

    // A committed (released) value clamps into [0, 1].
    component.onVolumeCommit({ target: { value: '5' } } as unknown as Event);
    expect(component.volume()).toBe(1);
  });
});
