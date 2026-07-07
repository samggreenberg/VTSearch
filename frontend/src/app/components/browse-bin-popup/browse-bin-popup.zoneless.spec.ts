import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BrowseBinPopupComponent } from './browse-bin-popup.component';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Drain several zoneless settle passes. The popup places itself across a chain
 * of `setTimeout(place)` → `requestAnimationFrame(nudge)` → `markForCheck`, and
 * the `setTimeout` is itself scheduled by the async CD tick that runs *after*
 * the first settle's own macrotask — so a single `settleZoneless` can return
 * before the reveal chain has run. A few passes flush it deterministically.
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
 * `setTimeout` + `requestAnimationFrame`, with no change detection attached, so
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
    const activeContextStub: Partial<ActiveContextService> = {
      mediaUrl: (p: string) => p,
    };
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
