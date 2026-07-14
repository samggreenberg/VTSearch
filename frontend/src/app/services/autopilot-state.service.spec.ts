import { TestBed } from '@angular/core/testing';
import { AutopilotStateService } from './autopilot-state.service';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';
import type { StatusIndicator } from '../generated/api-client/models/status-indicator';

function makeStatus(
  smart: StatusIndicator,
  stable: StatusIndicator,
  span: StatusIndicator,
): LabelingStatusResponse {
  return { good_count: 0, bad_count: 0, total_count: 0, smart, stable, span };
}

describe('AutopilotStateService', () => {
  let service: AutopilotStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AutopilotStateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start idle', () => {
    expect(service.state.phase).toBe('idle');
    expect(service.running).toBe(false);
  });

  it('activate should move to good phase', () => {
    service.activate();
    expect(service.state.phase).toBe('good');
    expect(service.running).toBe(true);
  });

  it('activate when already running should be a no-op', () => {
    service.activate();
    service.activate();
    expect(service.state.phase).toBe('good');
  });

  it('deactivate should move to idle', () => {
    service.activate();
    service.deactivate();
    expect(service.state.phase).toBe('idle');
    expect(service.running).toBe(false);
  });

  it('should transition from good to bad when enough good votes', () => {
    service.activate();
    service.checkPhaseTransition(3, 0); // goodToStart default is 3
    expect(service.state.phase).toBe('bad');
  });

  it('should transition from bad to hard when enough bad votes', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4); // badToStart default is 4
    expect(service.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart and stable are green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    const status: LabelingStatusResponse = makeStatus(
      { status: 'green' },
      { status: 'green' },
      { status: 'yellow' },
    );
    service.updateFromLabelingStatus(status);
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');
  });

  it('should transition from new to done when span is green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(10, 10);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'green' }),
    );
    service.checkPhaseTransition(15, 15);
    expect(service.state.phase).toBe('done');
  });

  it('should bounce from new back to hard when smart goes non-green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');

    // Smart drops to yellow (surprise destabilized the model)
    service.updateFromLabelingStatus(
      makeStatus({ status: 'yellow' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(12, 12);
    expect(service.state.phase).toBe('hard');
  });

  it('should bounce from new back to hard when stable goes non-green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');

    // Stable drops to yellow (surprise caused prediction flips)
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'yellow' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(12, 12);
    expect(service.state.phase).toBe('hard');
  });

  it('should return to new after bouncing back to hard once indicators recover', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');

    // Bounce back to hard
    service.updateFromLabelingStatus(
      makeStatus({ status: 'yellow' }, { status: 'yellow' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(12, 12);
    expect(service.state.phase).toBe('hard');

    // Indicators recover; should go back to new
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(15, 15);
    expect(service.state.phase).toBe('new');
  });

  it('should not bounce from new if both smart and stable remain green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');

    // Both still green; should stay in new
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
    );
    service.checkPhaseTransition(12, 12);
    expect(service.state.phase).toBe('new');
  });

  it('should cascade good→bad→hard in a single checkPhaseTransition call', () => {
    service.activate();
    // Both thresholds met at once (user labeled in Manual before switching to Autopilot)
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('hard');
  });

  it('should cascade good→bad in one call when only good threshold met', () => {
    service.activate();
    service.checkPhaseTransition(5, 2); // enough goods, not enough bads
    expect(service.state.phase).toBe('bad');
  });

  it('updateFromLabelingStatus should update status fields', () => {
    service.activate();
    const status: LabelingStatusResponse = makeStatus(
      { status: 'yellow' },
      { status: 'green' },
      { status: 'red', diversity_level: 0.75 },
    );
    service.updateFromLabelingStatus(status);

    expect(service.state.smartStatus).toBe('yellow');
    expect(service.state.stableStatus).toBe('green');
    expect(service.state.spanStatus).toBe('red');
    expect(service.state.fracDiversity).toBe(0.75);
  });

  it('clear should reset to initial state', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.clear();

    expect(service.state.phase).toBe('idle');
    expect(service.state.smartStatus).toBe('');
  });

  it('activate without retrainMode should default to false', () => {
    service.activate();
    expect(service.state.retrainMode).toBe(false);
  });

  it('activate with retrainMode=true should set the flag on state', () => {
    service.activate(true);
    expect(service.state.retrainMode).toBe(true);
    expect(service.state.phase).toBe('good');
  });

  it('retrainMode should persist through phase transitions until cleared', () => {
    service.activate(true);
    service.checkPhaseTransition(3, 4);
    expect(service.state.phase).toBe('hard');
    expect(service.state.retrainMode).toBe(true);

    service.deactivate();
    expect(service.state.retrainMode).toBe(true); // deactivate only flips phase
    service.clear();
    expect(service.state.retrainMode).toBe(false);
  });

  it('caps the good target so a 1-item dataset can advance past the good phase', () => {
    service.activate();
    // 1-item dataset: default good target is 3, but only 1 item exists.
    // Voting that single item good should satisfy the (capped) good target.
    service.checkPhaseTransition(1, 0, 1);
    expect(service.state.phase).not.toBe('good');
  });

  it('reaches the exhausted terminal state when a tiny dataset is fully labeled', () => {
    service.activate();
    // 1-item dataset, single item voted good: nothing left to label and the
    // indicators can never go green, so autopilot lands in "exhausted".
    service.checkPhaseTransition(1, 0, 1);
    expect(service.state.phase).toBe('exhausted');
  });

  it('reaches exhausted regardless of whether the lone item was voted good or bad', () => {
    service.activate();
    service.checkPhaseTransition(0, 1, 1); // single item voted bad
    expect(service.state.phase).toBe('exhausted');
  });

  it('reaches exhausted on a small dataset that cannot meet the good+bad quorum', () => {
    service.activate();
    // 5 items: cannot reach 3 good AND 4 bad (needs 7). Fully labeled → exhausted.
    service.checkPhaseTransition(3, 2, 5);
    expect(service.state.phase).toBe('exhausted');
  });

  it('does not go exhausted while unlabeled items remain', () => {
    service.activate();
    // 10-item dataset, only 2 labeled: still in an early phase, not exhausted.
    service.checkPhaseTransition(1, 1, 10);
    expect(service.state.phase).not.toBe('exhausted');
  });

  it('prefers the all-green done state over exhausted when indicators are green', () => {
    service.activate();
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'green' }),
    );
    // Fully labeled AND all green: the happy "done" path wins over "exhausted".
    service.checkPhaseTransition(3, 2, 5);
    expect(service.state.phase).toBe('done');
  });

  it('regresses out of exhausted when an unlabeled item reappears (vote cleared)', () => {
    service.activate();
    service.checkPhaseTransition(1, 0, 1);
    expect(service.state.phase).toBe('exhausted');
    // User clears the vote: now nothing is labeled again and the item is
    // available, so the phase regresses to the good phase.
    service.checkPhaseTransition(0, 0, 1);
    expect(service.state.phase).toBe('good');
  });

  it('leaves targets uncapped when totalCount is unknown (default 0)', () => {
    service.activate();
    // No size passed: 1 good is not enough for the default target of 3.
    service.checkPhaseTransition(1, 0);
    expect(service.state.phase).toBe('good');
  });

  it('state$ should emit on changes', () => new Promise<void>((done) => {
    const phases: string[] = [];
    service.state$.subscribe((s) => phases.push(s.phase));

    service.activate();

    setTimeout(() => {
      expect(phases).toContain('idle');
      expect(phases).toContain('good');
      done();
    });
  }));
});
