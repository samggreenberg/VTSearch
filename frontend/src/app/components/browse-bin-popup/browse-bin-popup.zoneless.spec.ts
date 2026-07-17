import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of, Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BrowseBinPopupComponent } from './browse-bin-popup.component';
import type { NowPlaying } from '../browse-hover-preview/browse-hover-preview.component';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { makeActiveContextStub } from '../../testing/mocks';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Drain several zoneless settle passes. The popup places itself across a chain
 * of `setTimeout(place)` → `afterNextRender(nudge)` → `markForCheck`, and the
 * `setTimeout` is itself scheduled by the async CD tick that runs *after* the
 * first settle's own macrotask — so a single `settleZoneless` can return before
 * the reveal chain has run. A few passes flush it deterministically.
 */
async function settlePasses(fixture: ComponentFixture<unknown>, passes = 3): Promise<void> {
  for (let i = 0; i < passes; i++) await settleZoneless(fixture);
}

/**
 * Zoneless positioning guard for the VTSBrowse bin detail popup.
 *
 * The popup opens hidden and reveals itself only after its position has been
 * clamped on-screen and the rendered panel measured (see `place()` /
 * `nudgeOnScreen()`). Under zoneless change detection those steps run in a bare
 * `setTimeout` + `afterNextRender`, with no change detection attached, so
 * the reveal must schedule its own `markForCheck()` or the `visibility` binding
 * silently goes stale — the popup would stay invisible, or (before the reveal
 * was moved past the measurement) flash at the un-corrected spot. This spec
 * drives the real component through the production channel and asserts on the
 * rendered DOM, with NO manual `detectChanges()`, so a missing notification
 * surfaces as a stale style rather than being papered over.
 */
describe('BrowseBinPopupComponent (zoneless positioning)', () => {
  let settings: ReturnType<typeof signal<Record<string, unknown> | null>>;
  let rafSpy: ReturnType<typeof vi.spyOn>;

  function makeFixture(): ComponentFixture<BrowseBinPopupComponent> {
    const selectionStub: Partial<BrowseSelectionService> = {
      version: signal(0) as unknown as BrowseSelectionService['version'],
      has: () => false,
      selectedCountIn: () => 0,
      addAll: () => {},
      removeAll: () => {},
      remove: () => {},
    };
    const metadataStub: Partial<MediaMetadataCacheService> = {
      version$: of(0),
      get: () => undefined,
      ensureLoaded: () => {},
    };
    const activeContextStub = makeActiveContextStub();
    const settingsStub: Partial<SettingsStateService> = {
      settingsSignal: settings as unknown as SettingsStateService['settingsSignal'],
      load: () => {},
      update: () => of({}) as ReturnType<SettingsStateService['update']>,
    };

    TestBed.resetTestingModule();
    configureZoneless({
      imports: [BrowseBinPopupComponent],
      providers: [
        { provide: BrowseSelectionService, useValue: selectionStub },
        { provide: MediaMetadataCacheService, useValue: metadataStub },
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: SettingsStateService, useValue: settingsStub },
      ],
    });

    const fixture = TestBed.createComponent(BrowseBinPopupComponent);
    // A singleton bin summoned far off the right/bottom edge of a small canvas:
    // the raw anchor (5000, 5000) is well outside the 400×400 region, so a
    // correct placement must pull the popup back inside it.
    fixture.componentRef.setInput('memberIds', [1]);
    fixture.componentRef.setInput('repId', 1);
    fixture.componentRef.setInput('mediaType', 'image');
    fixture.componentRef.setInput('bounds', new DOMRect(0, 0, 400, 400));
    fixture.componentRef.setInput('x', 5000);
    fixture.componentRef.setInput('y', 5000);
    return fixture;
  }

  beforeEach(() => {
    settings = signal<Record<string, unknown> | null>({});
    // jsdom doesn't implement Element.scrollTo; the CDK virtual viewport calls it
    // when a grid-bearing popup (any multi-item bin, or a non-preview media type
    // like audio, which always renders the member grid) scrolls to the
    // representative. Stub it so those popups don't throw during placement.
    if (!Element.prototype.scrollTo) {
      (Element.prototype as unknown as { scrollTo: () => void }).scrollTo = () => {};
    }
    // Run requestAnimationFrame callbacks on a microtask so `settleZoneless`'s
    // macrotask + `whenStable` drain them deterministically (jsdom's real rAF
    // fires on a ~16ms timer that would race the settle). Deferring rather than
    // running synchronously avoids re-entrancy if a callback re-schedules rAF.
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      void Promise.resolve().then(() => cb(0));
      return 0;
    });
  });

  afterEach(() => {
    rafSpy.mockRestore();
  });

  it('reveals the popup at its clamped spot once measured, no manual detectChanges', async () => {
    const fixture = makeFixture();
    await settlePasses(fixture);

    const panel = fixture.nativeElement.querySelector('.bin-popup') as HTMLElement;
    expect(panel).not.toBeNull();
    // The reveal must have reached the DOM through scheduled change detection
    // alone: this is the zoneless staleness guard for the reveal's markForCheck.
    expect(panel.style.visibility).toBe('visible');
    // …and it must sit inside the visible region, not at the raw off-screen
    // anchor it was summoned from. A revealed-but-unclamped popup would still
    // read 5000px here.
    expect(parseFloat(panel.style.left)).toBeLessThan(400);
    expect(parseFloat(panel.style.top)).toBeLessThan(400);

    fixture.destroy();
  });

  it('reveals once settings resolve after mount, with no re-summon (the two-right-click bug)', async () => {
    // Regression for "the first right-click shows nothing; the second opens it".
    //
    // In the app the popup mounts while its settings-driven prefs are already known
    // (settings loaded app-wide), but its own ngAfterViewInit `settingsState.load()`
    // momentarily resets `settingsSignal()` to null while the resource refetches. So
    // its first placement pass runs with `settingsReady` false and the reveal gate in
    // nudgeOnScreen holds `placed` false. When settings land *again* the prefs are
    // unchanged (same values), so the settings effect's size-change branch is a no-op
    // — and before the fix nothing else re-ran `place()`, stranding the popup hidden
    // until the next summon (the second right-click). The `!this.placed` clause makes
    // the effect re-place on settings-arrival while still unrevealed.
    //
    // Reproduce that state precisely: place with settings null (→ hidden, `placed`
    // false), then pre-seed the effect's "last seen prefs" to the values the arriving
    // settings will produce, so the size-change branch stays a no-op and only the
    // `!this.placed` clause can reveal — exactly the code path the fix adds.
    settings.set(null);
    const fixture = makeFixture();
    // Summon fully on-screen so there is no size/clamp reason to re-place — the reveal
    // must come purely from settings landing, not from a corrective measurement.
    fixture.componentRef.setInput('x', 50);
    fixture.componentRef.setInput('y', 50);
    await settlePasses(fixture);

    const panel = fixture.nativeElement.querySelector('.bin-popup') as HTMLElement;
    expect(panel).not.toBeNull();
    // Held hidden while settings are still null (the stranded state the user saw).
    expect(panel.style.visibility).toBe('hidden');

    const comp = fixture.componentInstance as unknown as {
      lastPreviewOverride: number | null;
      lastMetadataShown: boolean | null;
      previewOverride: number | null;
      showMetadataColumn: boolean;
    };
    comp.lastPreviewOverride = comp.previewOverride;
    comp.lastMetadataShown = comp.showMetadataColumn;

    // Settings resolve (resource refetch completes). No new summon, and — with last*
    // pre-seeded — no pref change, so only the unrevealed-re-place path is left.
    settings.set({});
    await settlePasses(fixture);

    // The popup must now reveal itself — without a second right-click.
    expect(panel.style.visibility).toBe('visible');

    fixture.destroy();
  });

  it('pulls a panel taller than the region up so its floor stays on-screen', async () => {
    // The corrective measurement pass (nudgeOnScreen) keeps the *rendered* panel
    // inside the region even when the computed clamp under-modelled its height —
    // the "detail window pops up slightly out of frame" regression. jsdom reports
    // 0-size rects, so stub the panel to report a height that overflows the 400px
    // region and assert the popup is pulled up until its floor lands at
    // regionBottom - EDGE_MARGIN (8), i.e. a top of 400 - 380 - 8 = 12.
    const fixture = makeFixture();
    fixture.componentRef.setInput('x', 50);
    fixture.componentRef.setInput('y', 50);
    await settlePasses(fixture);

    const panel = fixture.nativeElement.querySelector('.bin-popup') as HTMLElement;
    panel.getBoundingClientRect = () =>
      ({
        width: 100,
        height: 380,
        top: 0,
        left: 0,
        right: 100,
        bottom: 380,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }) as DOMRect;

    // Re-place: a size-input change re-clamps and re-measures without
    // re-anchoring to the cursor, so the stubbed oversize drives the correction.
    fixture.componentRef.setInput('hoverThumbRadius', 40);
    await settlePasses(fixture, 5);

    const top = parseFloat(panel.style.top);
    expect(top).toBe(12);
    // The measured floor sits exactly at the region's bottom edge less the margin.
    expect(top + 380).toBeLessThanOrEqual(400 - 8);

    fixture.destroy();
  });

  it('offers the metadata toggle and column for text (no preview pane)', async () => {
    // Text is a non-thumbnailed type: no magnified preview pane on the canvas or
    // in the popup. (Audio now tiles as waveform thumbnails, so it is no longer
    // this branch.) The metadata panel is media-agnostic, though, so the Info
    // button and the (default-shown) metadata column must still be offered, so
    // hovering a bin member surfaces its metadata just like it does for images.
    const fixture = makeFixture();
    fixture.componentRef.setInput('mediaType', 'text');
    fixture.componentRef.setInput('memberIds', [1, 2]);
    fixture.componentRef.setInput('repId', 1);
    await settlePasses(fixture);

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('.bin-popup-preview')).toBeNull();
    expect(root.querySelector('.bin-popup-meta-toggle')).not.toBeNull();
    expect(root.querySelector('.bin-popup-meta-col')).not.toBeNull();

    fixture.destroy();
  });

  it('reserves the body padding so a small text bin grid does not scroll', async () => {
    // The body is ``box-sizing: border-box`` with 12px of vertical padding, so its
    // bound height must exceed the flex content it holds by that padding — else the
    // grid column's ``height: 100%`` content box comes up 12px short and the member
    // grid gets a stray scrollbar even on a single row (the bug this guards). Text
    // has no preview pane, so the grid is the exact-fit element that reveals it.
    const fixture = makeFixture();
    fixture.componentRef.setInput('mediaType', 'text');
    fixture.componentRef.setInput('memberIds', [1, 2]);
    fixture.componentRef.setInput('repId', 1);
    await settlePasses(fixture);

    const cmp = fixture.componentInstance;
    const body = fixture.nativeElement.querySelector('.bin-popup-body') as HTMLElement;
    // The bound (border-box) height carries the grid column's full content height
    // (count label + rows) plus the body's own 12px vertical padding on top.
    expect(parseFloat(body.style.height)).toBe(cmp.gridColHeight + 12);

    fixture.destroy();
  });

  it('auto-plays the representative clip when an audio bin opens', async () => {
    // Opening the detail popup is an "intense hover": for audio it should start
    // auditioning the bin's representative straight away, with no grid-row hover
    // — the behavior a singleton audio bin (no member grid) relies on entirely.
    // jsdom doesn't implement the media element's play/load, so stub them; the
    // now-playing emit happens before the deferred play() regardless.
    const playStub = vi
      .spyOn(HTMLMediaElement.prototype, 'play')
      .mockResolvedValue(undefined);
    const loadStub = vi
      .spyOn(HTMLMediaElement.prototype, 'load')
      .mockImplementation(() => {});

    const fixture = makeFixture();
    fixture.componentRef.setInput('mediaType', 'audio');
    fixture.componentRef.setInput('memberIds', [7, 8, 9]);
    fixture.componentRef.setInput('repId', 8);
    const played: (NowPlaying | null)[] = [];
    fixture.componentInstance.nowPlaying.subscribe((v) => played.push(v));
    await settlePasses(fixture);

    // The window opened auditioning the representative (id 8), not the list's
    // first member (id 7), and surfaced it on the shared now-playing indicator —
    // flagged `loading` until the clip is actually sounding.
    // `progress` is null until a finite duration is known (jsdom has no media
    // pipeline, so it never is); the sweeping playhead stays hidden meanwhile.
    expect(played.at(-1)).toEqual({ mediaId: 8, waveUrl: '/api/medias/8/thumbnail', loading: true, progress: null });

    // The clip's audio element drives the buffering spinner: `playing` clears
    // the loading flag, a `waiting` stall re-sets it.
    const audioEl = fixture.nativeElement.querySelector('audio') as HTMLAudioElement;
    audioEl.dispatchEvent(new Event('playing'));
    await settlePasses(fixture);
    expect(played.at(-1)).toEqual({ mediaId: 8, waveUrl: '/api/medias/8/thumbnail', loading: false, progress: null });

    audioEl.dispatchEvent(new Event('waiting'));
    await settlePasses(fixture);
    expect(played.at(-1)).toEqual({ mediaId: 8, waveUrl: '/api/medias/8/thumbnail', loading: true, progress: null });

    playStub.mockRestore();
    loadStub.mockRestore();
    fixture.destroy();
  });

  it('stays hidden until settings load, then reveals', async () => {
    // Cold open: settings not yet resolved. The popup's size (thumbnail size,
    // metadata column, preview pane) all come from settings, so it must not
    // reveal at default sizes and then re-clamp when they arrive.
    settings.set(null);
    const fixture = makeFixture();
    await settlePasses(fixture);

    const panel = fixture.nativeElement.querySelector('.bin-popup') as HTMLElement;
    expect(panel.style.visibility).toBe('hidden');

    // Settings resolve through the same signal the app writes; the popup's
    // settings effect re-places and reveals.
    settings.set({});
    await settlePasses(fixture);
    expect(panel.style.visibility).toBe('visible');

    fixture.destroy();
  });
});

/**
 * Docked presentation: the same component rendered as the Browse view's
 * persistent left panel instead of a floating window. It must drop every piece
 * of floating machinery (positioning, clamping, dragging, outside-click /
 * Escape dismissal, document-level shortcuts) and instead render inline, show
 * an empty hint before a bin is opened, keep the member-grid area allocated but
 * empty for a singleton, and — for audio — track whatever clip is auditioning
 * anywhere in the Browse view.
 */
describe('BrowseBinPopupComponent (docked presentation)', () => {
  let settings: ReturnType<typeof signal<Record<string, unknown> | null>>;
  let rafSpy: ReturnType<typeof vi.spyOn>;

  function makeDockedFixture(
    memberIds: number[],
    mediaType = 'image',
  ): ComponentFixture<BrowseBinPopupComponent> {
    const selectionStub: Partial<BrowseSelectionService> = {
      version: signal(0) as unknown as BrowseSelectionService['version'],
      has: () => false,
      selectedCountIn: () => 0,
      addAll: () => {},
      removeAll: () => {},
      remove: () => {},
    };
    const metadataStub: Partial<MediaMetadataCacheService> = {
      version$: of(0),
      get: () => undefined,
      ensureLoaded: () => {},
    };
    const activeContextStub = makeActiveContextStub();
    const settingsStub: Partial<SettingsStateService> = {
      settingsSignal: settings as unknown as SettingsStateService['settingsSignal'],
      load: () => {},
      update: () => of({}) as ReturnType<SettingsStateService['update']>,
    };

    TestBed.resetTestingModule();
    configureZoneless({
      imports: [BrowseBinPopupComponent],
      providers: [
        { provide: BrowseSelectionService, useValue: selectionStub },
        { provide: MediaMetadataCacheService, useValue: metadataStub },
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: SettingsStateService, useValue: settingsStub },
      ],
    });

    const fixture = TestBed.createComponent(BrowseBinPopupComponent);
    fixture.componentRef.setInput('docked', true);
    fixture.componentRef.setInput('availableWidth', 340);
    fixture.componentRef.setInput('memberIds', memberIds);
    fixture.componentRef.setInput('repId', memberIds[0] ?? null);
    fixture.componentRef.setInput('mediaType', mediaType);
    return fixture;
  }

  beforeEach(() => {
    settings = signal<Record<string, unknown> | null>({});
    if (!Element.prototype.scrollTo) {
      (Element.prototype as unknown as { scrollTo: () => void }).scrollTo = () => {};
    }
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      void Promise.resolve().then(() => cb(0));
      return 0;
    });
  });

  afterEach(() => {
    rafSpy.mockRestore();
  });

  it('renders inline (no floating positioning) and is immediately visible', async () => {
    const fixture = makeDockedFixture([1, 2, 3]);
    await settlePasses(fixture);

    const panel = fixture.nativeElement.querySelector('.bin-popup') as HTMLElement;
    expect(panel).not.toBeNull();
    expect(panel.classList.contains('bin-popup--docked')).toBe(true);
    // Docked never sets the fixed-position left/top/visibility bindings.
    expect(panel.style.left).toBe('');
    expect(panel.style.top).toBe('');
    expect(panel.style.visibility).toBe('');

    fixture.destroy();
  });

  it('shows an empty hint when no bin has been opened', async () => {
    const fixture = makeDockedFixture([]);
    await settlePasses(fixture);

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('.bin-popup-empty-hint')).not.toBeNull();
    // No grid rows and no count when empty.
    expect(root.querySelector('.bin-popup-grid-header')).toBeNull();

    fixture.destroy();
  });

  it('offers a pop-out button (not the dock button) in the header', async () => {
    const fixture = makeDockedFixture([1, 2]);
    await settlePasses(fixture);

    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('[aria-label="Pop out into a floating window"]')).not.toBeNull();
    expect(root.querySelector('[aria-label="Dock as a side panel"]')).toBeNull();

    fixture.destroy();
  });

  it('keeps the member grid empty for a docked singleton (area still allocated)', async () => {
    const fixture = makeDockedFixture([5]);
    await settlePasses(fixture);

    const cmp = fixture.componentInstance;
    // The grid column is present (not dropped like the floating previewOnly path)…
    const root = fixture.nativeElement as HTMLElement;
    expect(root.querySelector('.bin-popup-grid-col')).not.toBeNull();
    // …but renders no rows: the lone item lives in the detail pane above.
    expect(cmp.displayRows.length).toBe(0);

    fixture.destroy();
  });

  it('tracks the externally auditioning clip for docked audio (now-playing pane)', async () => {
    const fixture = makeDockedFixture([7, 8, 9], 'audio');
    await settlePasses(fixture);

    const cmp = fixture.componentInstance;
    // With nothing auditioning yet the pane follows the representative.
    expect(cmp.displayedId).toBe(7);

    // A canvas hover starts a clip elsewhere in the Browse view; the docked
    // pane + metadata must follow it (the now-playing replacement).
    fixture.componentRef.setInput('nowPlayingExt', {
      mediaId: 42,
      waveUrl: '/api/medias/42/thumbnail',
      loading: true,
      progress: 0.5,
    } as NowPlaying);
    await settlePasses(fixture);

    expect(cmp.displayedId).toBe(42);
    expect(cmp.paneWaveUrl()).toBe('/api/medias/42/thumbnail');
    expect(cmp.paneProgress).toBe(0.5);
    expect(cmp.paneLoading).toBe(true);

    fixture.destroy();
  });

  it('derives the docked grid column count from the panel width', async () => {
    const fixture = makeDockedFixture([1, 2, 3, 4, 5, 6]);
    await settlePasses(fixture);

    const cmp = fixture.componentInstance;
    const wide = cmp.columns;
    // Narrowing the panel (divider drag) re-chunks to fewer columns.
    fixture.componentRef.setInput('availableWidth', 220);
    await settlePasses(fixture);
    expect(cmp.columns).toBeLessThanOrEqual(wide);

    fixture.destroy();
  });

  it('does not dismiss on outside click or Escape when docked', async () => {
    const fixture = makeDockedFixture([1, 2]);
    await settlePasses(fixture);

    const dismissed = vi.fn();
    fixture.componentInstance.dismissed.subscribe(dismissed);

    document.body.click();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await settlePasses(fixture);

    expect(dismissed).not.toHaveBeenCalled();

    fixture.destroy();
  });
});

describe('BrowseBinPopupComponent (scroll-prefetch re-wiring)', () => {
  it('re-subscribes prefetch when the member-grid viewport is recreated', () => {
    // Regression: the popup subscribed scrolledIndexChange only once, but the
    // viewport lives behind @if (!previewOnly) and the popup is reused across
    // summons (right-clicking another bin only swaps inputs). A singleton→multi
    // transition created a fresh viewport whose stream was never subscribed, so
    // scrolling the member grid never hydrated thumbnails beyond the
    // initially-prefetched window. In production a constructor effect tracking
    // the `viewport` view-query signal re-runs ensureScrollSubscription
    // whenever the instance changes; here the query is stubbed as a plain
    // function and the method driven directly, since the virtualized grid
    // doesn't render reliably under jsdom.
    const component = Object.create(
      BrowseBinPopupComponent.prototype,
    ) as BrowseBinPopupComponent;
    const state = component as unknown as {
      viewport: () => unknown;
      scrollSub: unknown;
      scrollSubscribedViewport: unknown;
      prefetchVisible(): void;
      ensureScrollSubscription(): void;
    };
    state.scrollSub = null;
    state.scrollSubscribedViewport = null;
    const prefetchSpy = vi.fn();
    state.prefetchVisible = prefetchSpy;

    const vp1 = { scrolledIndexChange: new Subject<number>() };
    state.viewport = () => vp1;
    state.ensureScrollSubscription();
    vp1.scrolledIndexChange.next(0);
    const callsAfterFirstScroll = prefetchSpy.mock.calls.length;
    expect(callsAfterFirstScroll).toBeGreaterThan(0);

    // Viewport destroyed (previewOnly summon) …
    vp1.scrolledIndexChange.complete();
    state.viewport = () => undefined;
    state.ensureScrollSubscription();

    // … then a new multi-member summon creates a fresh instance.
    const vp2 = { scrolledIndexChange: new Subject<number>() };
    state.viewport = () => vp2;
    state.ensureScrollSubscription();

    const callsBeforeSecondScroll = prefetchSpy.mock.calls.length;
    vp2.scrolledIndexChange.next(2);
    expect(prefetchSpy.mock.calls.length).toBe(callsBeforeSecondScroll + 1);
  });
});

/**
 * Keyboard focus sync for the member grid.
 *
 * Arrow keys walk the highlighted item (``previewId``), but before this they
 * left DOM focus behind on whatever entry the user last tabbed/clicked. Enter is
 * only caught by the focused entry's own handler, so it toggled the stale
 * DOM-focused entry rather than the arrow-walked one; Space fired both the
 * document fallback and the focused entry, double-toggling. The fix keeps DOM
 * focus glued to the walked entry (and ``previewId`` glued to DOM focus), and
 * lets the focused entry own its activation without the fallback double-firing.
 *
 * The member grid is virtualized and doesn't render individual entries reliably
 * under jsdom, so these drive the sync contract directly on the component rather
 * than through a rendered ArrowRight → ``document.activeElement`` round-trip.
 */
describe('BrowseBinPopupComponent (keyboard focus sync)', () => {
  interface GridState {
    ids: number[];
    columns: number;
    previewId: number | null;
    cdr: { markForCheck: () => void };
    // The `panelRef` view query is a signal; stub it as a plain function.
    panelRef?: () => { nativeElement: { querySelector: (sel: string) => HTMLElement | null } };
    mediaType: () => string;
    mediaTypeCaps: { usesThumbnails: (t: string) => boolean };
    scrollRowIntoView: (index: number) => void;
    selection: { has: (id: number) => boolean; addAll: (ids: number[]) => void; remove: (id: number) => void };
    moveFocus(dCol: number, dRow: number): void;
  }

  function makeGridComponent(): { component: BrowseBinPopupComponent; state: GridState } {
    const component = Object.create(BrowseBinPopupComponent.prototype) as BrowseBinPopupComponent;
    const state = component as unknown as GridState;
    state.ids = [10, 20, 30, 40];
    state.columns = 2;
    state.previewId = 10;
    state.cdr = { markForCheck: vi.fn() };
    // Non-thumbnail media so `previewOnly` is false and the grid path is live.
    state.mediaType = () => 'text';
    state.mediaTypeCaps = { usesThumbnails: (t: string) => t !== 'text' };
    state.scrollRowIntoView = vi.fn();
    state.selection = { has: () => false, addAll: vi.fn(), remove: vi.fn() };
    return { component, state };
  }

  let rafSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    // Run rAF synchronously so `focusEntry`'s deferred (and retried) focus lands
    // within the test tick, deterministically.
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      cb(0);
      return 0;
    });
  });
  afterEach(() => rafSpy.mockRestore());

  it('moves DOM focus to the arrow-walked entry', () => {
    const { component, state } = makeGridComponent();
    const walked = { focus: vi.fn() } as unknown as HTMLElement;
    state.panelRef = () => ({
      nativeElement: {
        // Only the entry for id 20 (the ArrowRight target from index 0) exists.
        querySelector: (sel: string) => (sel.includes('"20"') ? walked : null),
      },
    });

    state.moveFocus(1, 0);

    // The highlight advanced to id 20 …
    expect(state.previewId).toBe(20);
    // … and DOM focus followed it, so Enter/Space now act on the highlighted item.
    expect(walked.focus).toHaveBeenCalledWith({ preventScroll: true });
  });

  it('retries the focus until the virtualized entry renders, then focuses it', () => {
    const { state } = makeGridComponent();
    const walked = { focus: vi.fn() } as unknown as HTMLElement;
    let calls = 0;
    state.panelRef = () => ({
      nativeElement: {
        // Absent for the first two frames (row still virtualizing in), then in.
        querySelector: (sel: string) => {
          if (!sel.includes('"20"')) return null;
          return ++calls >= 3 ? walked : null;
        },
      },
    });

    state.moveFocus(1, 0);

    expect(walked.focus).toHaveBeenCalledTimes(1);
  });

  it('syncs the highlight to DOM focus that arrives by Tab or click', () => {
    const { component, state } = makeGridComponent();
    component.onEntryFocus(30);
    expect(state.previewId).toBe(30);
  });

  it('lets the focused entry own its activation, stopping the fallback double-toggle', () => {
    const { component, state } = makeGridComponent();
    const event = {
      key: 'Enter',
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as KeyboardEvent;

    component.onEntryKeydown(event, 42);

    expect(event.preventDefault).toHaveBeenCalled();
    // The bubble is stopped so the document-level Space fallback (which acts on
    // previewId) can't also fire and cancel this toggle.
    expect(event.stopPropagation).toHaveBeenCalled();
    expect(state.selection.addAll).toHaveBeenCalledWith([42]);
  });
});
