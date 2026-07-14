
import { HttpTestingController } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CenterPanelComponent } from './center-panel.component';
import { Media } from '../../models/api.models';
import { ANIMATIONS_OFF_CLASS } from '../../utils/reduced-motion';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the center panel (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 1.2). Phase 1.2 dropped the `zone.run(...)` re-entries from
 * `KeyboardService`, moving the change-detection trigger to this consumer, whose
 * shortcut-driven state (`isVoting`/`spinningVote`/`swipeClass`/`undoToastText`/
 * `volume`/`pendingBadConfirm` + the settings mirror) is now signalized.
 *
 * This spec runs under a zoneless `TestBed` and drives the component through the
 * *production channel* — a real `keydown` dispatched on `document` (handled by the
 * live `KeyboardService` listener, NOT a bound template event) — then asserts on
 * the rendered DOM after `settleZoneless()` with NO manual `detectChanges()`. The
 * keyboard callback and the vote/undo HTTP `.subscribe()` callbacks are all
 * un-bound, so if any of those state writes were a plain field instead of a signal
 * the scheduler would never be notified and the DOM would stay stale — failing
 * these assertions.
 */
describe('CenterPanelComponent (zoneless keyboard canary)', () => {
  let fixture: ComponentFixture<CenterPanelComponent>;
  let component: CenterPanelComponent;
  let httpMock: HttpTestingController;

  // Document media keeps the viewer fully inert in jsdom: it only sets a
  // sanitized iframe URL — no HTTP, no Loading placeholder (text viewer's
  // NG0100), no native `fetch()` of a relative URL (audio/video) and no
  // ViewChild settling quirk (image). The viewer is incidental here — the
  // assertions are on the always-rendered voting overlay / undo toast.
  const mockMedia: Media = {
    id: 1,
    media_type: 'document',
    filename: 'test.pdf',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    configureZoneless({
      imports: [CenterPanelComponent],
      providers: [...provideHttpTesting()],
    });
    fixture = TestBed.createComponent(CenterPanelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);

    // `media` is a decorator @Input; set it through `setInput` (the same channel
    // the parent's binding uses) so the write schedules CD.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);

    // init() wires the keyboard subscription + starts the document listener and
    // kicks the settings/embedders loads. The settings GET is driven by an
    // `rxResource`; while it is loading the zoneless app stays unstable, so we
    // must NOT `whenStable()` until it is flushed. Drain a macrotask + tick to
    // let the GETs be issued, flush them, THEN settle. show_animations:'hide'
    // takes the non-animated vote branch (no dangling timers) and is the
    // production path for "reduce motion".
    component.init();
    await new Promise<void>((resolve) => setTimeout(resolve));
    TestBed.tick();
    // label_hint_dismissed:true so the first vote's hint-dismissal does not fire
    // a settings PUT we'd have to flush.
    for (const req of httpMock.match((r) => r.url.includes('settings'))) {
      req.flush({ show_animations: 'hide', label_hint_dismissed: true });
    }
    for (const req of httpMock.match((r) => r.url.includes('embedders'))) {
      req.flush({ embedders: [] });
    }
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
    // The settings effect mirrors show_animations:'hide' onto <html>; undo it so
    // the class does not leak into other specs in the same jsdom document.
    document.documentElement.classList.remove(ANIMATIONS_OFF_CLASS);
  });

  function votingOverlay(): HTMLElement {
    return fixture.nativeElement.querySelector('vt-voting-overlay');
  }

  it('renders the vote as cast after a keyboard ArrowRight, with no manual detectChanges', async () => {
    expect(votingOverlay().querySelector('.btn-good.voted')).toBeNull();

    // Production channel: a real → keypress, handled by the live KeyboardService
    // listener (un-bound document callback), dispatches a 'vote' action that the
    // component's subscription turns into castVote('good').
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    await settleZoneless(fixture);

    // The optimistic vote POST; its response callback flips `isVoting` (a signal)
    // back, which is the only thing that can schedule CD for the un-bound chain.
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'good', click_time: 1 });
    await settleZoneless(fixture);

    expect(component.voteState.goodVotes.has(1)).toBe(true);
    // If `isVoting`/the vote state weren't signalized, this would still be null.
    expect(votingOverlay().querySelector('.btn-good.voted')).not.toBeNull();
  });

  it('shows the undo toast after a keyboard Cmd/Ctrl-Z, with no manual detectChanges', async () => {
    // Land a vote first so the undo stack has an entry to pop.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    await settleZoneless(fixture);
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'good', click_time: 1 });
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.undo-toast')).toBeNull();

    // Ctrl-Z → KeyboardService emits {type:'undo'} → voteState.undo() emits on
    // toast$, whose un-bound subscription writes the `undoToastText` signal.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true }));
    await settleZoneless(fixture);
    // undo() re-POSTs the reverted target ('none'); flush it.
    httpMock.expectOne('/api/medias/1/vote').flush({ state: 'none', click_time: 2 });
    await settleZoneless(fixture);

    const toast = fixture.nativeElement.querySelector('.undo-toast');
    expect(toast).not.toBeNull();
    expect(toast!.textContent).toContain('Undid vote on test.pdf');
  });
});
