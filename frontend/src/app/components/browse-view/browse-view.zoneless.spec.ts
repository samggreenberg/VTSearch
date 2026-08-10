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
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediasApiService } from '../../services/medias-api.service';
import { DatasetsListingsApiService } from '../../services/datasets-listings-api.service';
import { VtDialogService } from '../../services/dialog.service';
import { ToastService } from '../../services/toast.service';
import type { ProjectionMeta } from '../../models/projection.models';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { makeActiveContextStub } from '../../testing/mocks';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the VTSBrowse view.
 * Phase 2.6 signalized
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
      // The post-verify cull's in-place re-bin. Left un-emitting so the verify
      // tests below assert what the panel does *before* the server answers —
      // which is the whole point: the details must repaint on the click, not a
      // round-trip later.
      subsetRemove: () => EMPTY as unknown as ReturnType<ProjectionApiService['subsetRemove']>,
    };
    const tileCacheStub: Partial<TileCacheService> = {
      setSubset: noop,
      setProjectionId: noop,
      setContentVersion: noop,
      clear: noop,
    };
    const activeContextStub = makeActiveContextStub({
      pair$: EMPTY,
      datasetId: '',
      modelId: '',
      setActive: noop,
    });
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
    const mediasStub: Partial<MediasApiService> = {
      voteBulk: () =>
        of({ ok: true, changed: 0, missing: [] }) as ReturnType<MediasApiService['voteBulk']>,
    };
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
        // ngOnInit calls MediaTypeCapabilityService.ensureLoaded(), which lazily
        // resolves this service to fetch the thumbnail-type registry. The canary
        // DS is audio, so the served set must mark audio as thumbnail-backed.
        {
          provide: DatasetsListingsApiService,
          useValue: {
            getMediaTypes: () =>
              of({
                media_types: [
                  { type_id: 'image', name: 'Image', has_thumbnail: true },
                  { type_id: 'video', name: 'Video', has_thumbnail: true },
                  { type_id: 'document', name: 'Document', has_thumbnail: true },
                  { type_id: 'audio', name: 'Audio', has_thumbnail: true },
                  { type_id: 'text', name: 'Text', has_thumbnail: false },
                ],
              }),
          },
        },
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

  it('routes a right-clicked bin into the docked panel (not a floating popup) when docked', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    // Force the docked presentation for the active media type.
    component.detailsDocked.set(true);

    component.onCanvasContextMenu({
      members: [3, 4, 5],
      repId: 4,
      clientX: 120,
      clientY: 140,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);

    // The docked panel shows the bin (members carried), and NO floating popup opens.
    expect(component.contextMembers()).toEqual([3, 4, 5]);
    expect(component.contextRepId()).toBe(4);
    expect(component.contextMenuOpen).toBe(false);

    // The panel's X clears the shown bin but leaves the panel itself docked.
    component.onDetailsDismiss();
    expect(component.contextMembers()).toEqual([]);
    expect(component.detailsDocked()).toBe(true);
    expect(component.detailsPanelShown()).toBe(true);
  });

  it("hides the empty details panel on a second X, and a right-clicked bin brings it back docked", async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    component.detailsDocked.set(true);
    expect(component.detailsPanelShown()).toBe(true);

    // X with nothing showing: there is no bin left to clear, so the panel itself
    // goes away and the canvas takes back its width.
    component.onDetailsDismiss();
    expect(component.detailsPanelHidden()).toBe(true);
    expect(component.detailsPanelShown()).toBe(false);
    // Dismissing is not a presentation change: the docked choice is untouched.
    expect(component.detailsDocked()).toBe(true);

    // Right-clicking a bin brings the panel back — docked, showing that bin,
    // with no floating window.
    component.onCanvasContextMenu({
      members: [11, 12],
      repId: 11,
      clientX: 60,
      clientY: 70,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);
    expect(component.detailsPanelHidden()).toBe(false);
    expect(component.detailsPanelShown()).toBe(true);
    expect(component.contextMenuOpen).toBe(false);
    expect(component.contextMembers()).toEqual([11, 12]);
  });

  it('un-hides the details panel when a floating window is docked into it', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;

    // Hide the docked panel, then work in the floating presentation instead.
    component.onDetailsDismiss();
    expect(component.detailsPanelHidden()).toBe(true);
    component.detailsDocked.set(false);
    component.onCanvasContextMenu({
      members: [21],
      repId: 21,
      clientX: 10,
      clientY: 10,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);
    expect(component.contextMenuOpen).toBe(true);

    // Docking that window must not drop it into a hidden panel.
    component.onDockRequested();
    expect(component.detailsPanelHidden()).toBe(false);
    expect(component.detailsPanelShown()).toBe(true);
  });

  it('carries the open bin across dock / pop-out toggles', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;

    // Force the floating presentation (docked is the default) before opening a bin.
    component.detailsDocked.set(false);
    component.onCanvasContextMenu({
      members: [7, 8],
      repId: 7,
      clientX: 100,
      clientY: 100,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);
    expect(component.contextMenuOpen).toBe(true);

    // Dock: the floating window closes, the panel takes over the same bin.
    component.onDockRequested();
    expect(component.detailsDocked()).toBe(true);
    expect(component.contextMenuOpen).toBe(false);
    expect(component.contextMembers()).toEqual([7, 8]);

    // Pop out: the panel's bin re-opens as a floating window at a default anchor.
    component.onPopOutRequested();
    expect(component.detailsDocked()).toBe(false);
    expect(component.contextMenuOpen).toBe(true);
    expect(component.contextMembers()).toEqual([7, 8]);
  });

  it('prunes verified items out of the open bin so the details panel repaints', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    const selection = fixture.debugElement.injector.get(BrowseSelectionService);

    // A Find-positives browse with a bin open in the docked left panel.
    component.subset = true;
    component.subsetIds = [1, 2, 3, 4];
    component.detailsDocked.set(true);
    component.onCanvasContextMenu({
      members: [1, 2, 3],
      repId: 2,
      clientX: 0,
      clientY: 0,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);

    // Verify two of that bin's items away — the representative among them.
    selection.addAll([1, 2]);
    component.onVerify('bad');
    await settleZoneless(fixture);

    // The panel drops the removed items and keeps the survivor, moving the
    // representative off the culled one instead of showing a stale bin.
    expect(component.contextMembers()).toEqual([3]);
    expect(component.contextRepId()).toBe(3);
    // The map lost them too, and the selection they came from is empty again.
    expect(component.subsetIds).toEqual([3, 4]);
    expect(selection.size).toBe(0);
  });

  it('resets the details panel to empty when the whole open bin is verified away', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    const selection = fixture.debugElement.injector.get(BrowseSelectionService);

    component.subset = true;
    component.subsetIds = [1, 2, 3];
    component.detailsDocked.set(true);
    component.onCanvasContextMenu({
      members: [1, 2],
      repId: 1,
      clientX: 0,
      clientY: 0,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);

    // Every member of the open bin goes.
    selection.addAll([1, 2]);
    component.onVerify('good');
    await settleZoneless(fixture);

    // Empty details, not a stale bin of removed items. The panel itself stays
    // docked (showing its empty hint) — only its contents were cleared.
    expect(component.contextMembers()).toEqual([]);
    expect(component.contextRepId()).toBeNull();
    expect(component.detailsPanelShown()).toBe(true);
  });

  it('leaves the open bin alone when the verified items came from elsewhere', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    const selection = fixture.debugElement.injector.get(BrowseSelectionService);

    component.subset = true;
    component.subsetIds = [1, 2, 3, 4];
    component.detailsDocked.set(true);
    component.onCanvasContextMenu({
      members: [1, 2],
      repId: 1,
      clientX: 0,
      clientY: 0,
      bounds: null,
    } as unknown as Parameters<typeof component.onCanvasContextMenu>[0]);

    // The user selected items from other bins (marquee, select-all-in-view).
    selection.addAll([3, 4]);
    component.onVerify('bad');
    await settleZoneless(fixture);

    expect(component.contextMembers()).toEqual([1, 2]);
    expect(component.contextRepId()).toBe(1);
  });

  it('persists the details panel width from a divider drag, clamped', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    const updates: Record<string, unknown>[] = [];
    const settings = TestBed.inject(SettingsStateService);
    (settings.update as unknown) = (u: Record<string, unknown>) => {
      updates.push(u);
      return of({});
    };

    component.onDetailsDividerMouseDown({ preventDefault: () => {} } as MouseEvent);
    // Release commits the settled width to settings.
    (component as unknown as { onDetailsMouseUp(): void }).onDetailsMouseUp();

    expect(updates.some((u) => 'browse_details_panel_width' in u)).toBe(true);
  });

  it('resizes the details panel from the in-panel row divider: a vertical drag maps 1:1 to panel width', async () => {
    await settleZoneless(fixture);
    const component = fixture.componentInstance;
    const updates: Record<string, unknown>[] = [];
    const settings = TestBed.inject(SettingsStateService);
    (settings.update as unknown) = (u: Record<string, unknown>) => {
      updates.push(u);
      return of({});
    };

    // A wide layout so neither the floor nor the max clamps this drag.
    (component as unknown as { content: () => unknown }).content = () => ({
      nativeElement: { getBoundingClientRect: () => ({ width: 2000, left: 0, right: 2000 }) },
    });

    const startWidth = component.detailsPanelWidth();
    component.onDetailsRowDividerMouseDown({
      preventDefault: () => {},
      clientY: 100,
    } as unknown as MouseEvent);
    expect(component.draggingDetailsRow()).toBe(true);

    // Dragging the divider DOWN 60px grows the panel by 60 — the focused item is
    // square, so a taller item is a wider panel (the whole point of this divider).
    (component as unknown as { onDetailsRowMouseMove(e: MouseEvent): void }).onDetailsRowMouseMove({
      clientY: 160,
    } as unknown as MouseEvent);
    expect(component.detailsPanelWidth()).toBeCloseTo(startWidth + 60);

    // Dragging back UP past the start shrinks it below the start width.
    (component as unknown as { onDetailsRowMouseMove(e: MouseEvent): void }).onDetailsRowMouseMove({
      clientY: 60,
    } as unknown as MouseEvent);
    expect(component.detailsPanelWidth()).toBeLessThan(startWidth);

    // Release ends the drag and commits the settled width to settings.
    (component as unknown as { onDetailsRowMouseUp(): void }).onDetailsRowMouseUp();
    expect(component.draggingDetailsRow()).toBe(false);
    expect(updates.some((u) => 'browse_details_panel_width' in u)).toBe(true);
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
