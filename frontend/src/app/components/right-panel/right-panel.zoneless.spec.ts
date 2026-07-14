import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';

import { RightPanelComponent } from './right-panel.component';
import { VoteStateService } from '../../services/vote-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the right panel (docs/plans/zoneless-migration.md,
 * Phases 0.3/0.4 + 2.5/2.8). The panel used to mirror six `VoteStateService`
 * observables into plain fields via `subscribe`; Phase 2.5 signalized the service
 * and replaced those mirrors with `computed`s over its getters. This drives the
 * vote state through the production channel (`applyOptimisticState`, the same
 * signal write a vote click makes) and asserts the rendered DOM repaints with NO
 * manual `detectChanges()` — the Find-mode Browse action enables only when
 * `goodIds().length > 0`, so a stale mirror would leave it disabled.
 */
describe('RightPanelComponent (zoneless votes canary)', () => {
  let fixture: ComponentFixture<RightPanelComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [RightPanelComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(RightPanelComponent);
    httpMock = TestBed.inject(HttpTestingController);
    // Find mode renders the good bucket with the Browse/To-Dataset/Export
    // actions whose disabled state reads the `goodIds()` computed.
    fixture.componentRef.setInput('mode', 'find');
    fixture.componentRef.setInput('medias', [
      { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'x', custom_metadata: {} },
    ]);
  });

  afterEach(() => {
    fixture.componentInstance.ngOnDestroy();
    TestBed.inject(VoteStateService).stopPolling();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
    });
    fixture.destroy();
  });

  function browseButton(): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('button[aria-label="Browse"]');
  }

  // Drain init: the settings rxResource loader and the first votes poll. Hold
  // nothing back; the state change under test is driven without HTTP below.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    for (let i = 0; i < 3; i++) {
      await settleResource();
      httpMock.match('/api/settings').forEach((req) => req.flush({ volume: 1 }));
      httpMock.match('/api/votes').forEach((req) =>
        req.flush({ good: [], bad: [], click_times: {}, learned_scores: {} }),
      );
    }
  }

  it('enables the Find Browse action when a good vote lands, no manual detectChanges', async () => {
    await flushInit();
    await settleZoneless(fixture);

    // No goods yet → the Browse action is disabled.
    const btn = browseButton();
    expect(btn).not.toBeNull();
    expect(btn!.disabled).toBe(true);

    // Production channel: an optimistic good vote writes the goodVotes signal
    // (the same path a vote click takes), with no HTTP.
    TestBed.inject(VoteStateService).applyOptimisticState(1, 'good');
    await settleZoneless(fixture);

    // The `goodIds()` computed updated and the getter-bound `[disabled]`
    // repainted with no manual pump. A stale mirror would keep it disabled.
    expect(browseButton()!.disabled).toBe(false);
  });
});
