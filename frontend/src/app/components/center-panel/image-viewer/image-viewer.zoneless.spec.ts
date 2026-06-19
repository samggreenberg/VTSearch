import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageViewerComponent } from './image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/**
 * Zoneless staleness canary for the image viewer (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 2.3). Phase 2.3 signalized the viewer state that is written from
 * *un-patched* callbacks — the window-level `keydown`/`keyup`/`blur` (Shift) and
 * `mousemove`/`mouseup` drag handlers, the `ResizeObserver` rendered-size writes,
 * and the shake `setTimeout`. None of those callbacks are bound template/host
 * listeners, so under zoneless they schedule no change detection on their own; only
 * because the destinations are signals read in the template does a write notify the
 * scheduler.
 *
 * This spec runs under a zoneless `TestBed` and drives the component through the
 * *production channel* — a real `keydown`/`keyup` dispatched on `window` (handled by
 * the live constructor-registered listener, NOT a bound template event) — then
 * asserts on the rendered DOM after `settleZoneless()` with NO manual
 * `detectChanges()`. If `shiftHeld` were still a plain field, the `Shift` press
 * would never repaint the `.region-mode` crosshair affordance and these assertions
 * would fail.
 */
describe('ImageViewerComponent (zoneless Shift-drag canary)', () => {
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: Media = {
    id: 2,
    media_type: 'image',
    filename: 'test.png',
    md5: 'def456',
    custom_metadata: {},
  };

  beforeEach(async () => {
    configureZoneless({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    });
    fixture = TestBed.createComponent(ImageViewerComponent);
    // `media` is a decorator @Input; set it through `setInput` (the same channel
    // the parent's binding uses) so the first render has a media to read.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
  });

  function wrap(): HTMLElement {
    return fixture.nativeElement.querySelector('.image-wrap');
  }

  it('toggles the region-mode crosshair when Shift is pressed/released via the window listener, with no manual detectChanges', async () => {
    expect(wrap().classList.contains('region-mode')).toBe(false);

    // Production channel: a real Shift keydown, handled by the live constructor
    // listener (an un-bound window callback). It writes the `shiftHeld` signal,
    // which `regionDrawActive` reads in the template — the only thing that can
    // schedule CD for this un-bound chain under zoneless.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(true);

    // Releasing Shift (also an un-bound window callback) must repaint it away.
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(false);
  });

  it('clears region-mode on window blur via the un-bound blur listener', async () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Shift' }));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(true);

    // alt-tab / focus loss fires a window 'blur' that drops the Shift state.
    window.dispatchEvent(new Event('blur'));
    await settleZoneless(fixture);
    expect(wrap().classList.contains('region-mode')).toBe(false);
  });
});
