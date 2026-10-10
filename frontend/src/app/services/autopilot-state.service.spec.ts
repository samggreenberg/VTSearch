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

  describe('completion eligibility', () => {
    it('withholds the hand-off until the labelset has been read', () => {
      service.activate();
      expect(service.shouldAnnounceCompletion).toBe(false);

      service.noteInitialLabelset(0, 0);
      expect(service.shouldAnnounceCompletion).toBe(true);
    });

    it('withholds it for a detector that was already trained at run start', () => {
      service.activate();
      service.noteInitialLabelset(3, 2);
      expect(service.shouldAnnounceCompletion).toBe(false);
    });

    it('ignores every reading after the first, so the user\'s own votes cannot flip it', () => {
      service.activate();
      service.noteInitialLabelset(0, 0);
      // The user labels; the labelset is no longer evidence of anything.
      service.noteInitialLabelset(3, 2);
      expect(service.shouldAnnounceCompletion).toBe(true);
    });

    it('offers the hand-off once per run', () => {
      service.activate();
      service.noteInitialLabelset(0, 0);
      service.markCompletionAnnounced();
      expect(service.shouldAnnounceCompletion).toBe(false);
    });

    it('re-arms on a new run, but only for a detector that is still untrained', () => {
      service.activate();
      service.noteInitialLabelset(0, 0);
      service.markCompletionAnnounced();

      service.deactivate();
      service.activate();
      expect(service.shouldAnnounceCompletion).toBe(false);
      // Whatever the run just trained is in the labelset now, so the next run
      // reads as already-trained and stays quiet.
      service.noteInitialLabelset(3, 2);
      expect(service.shouldAnnounceCompletion).toBe(false);
    });
  });

  describe('retrain mode', () => {
    // #3535: the panel's `activate()` guess is taken before `/api/votes` has
    // answered, so on entry to the Train window it is always `false`. The run's
    // first real reading of the labelset is what decides.
    it('engages when the run\'s first labelset reading has both classes', () => {
      service.activate(false);
      expect(service.state.retrainMode).toBe(false);

      service.noteInitialLabelset(3, 2);
      expect(service.state.retrainMode).toBe(true);
    });

    it('stays off for a labelset with only one class', () => {
      service.activate(false);
      service.noteInitialLabelset(3, 0);
      expect(service.state.retrainMode).toBe(false);
    });

    it('turns a stale reading off again', () => {
      service.activate(true);
      service.noteInitialLabelset(0, 0);
      expect(service.state.retrainMode).toBe(false);
    });

    it('ignores every reading after the first, so the user\'s own votes cannot flip it', () => {
      service.activate(false);
      service.noteInitialLabelset(0, 0);
      // The user labels a good and a bad; that is training, not a detector that
      // arrived trained.
      service.noteInitialLabelset(1, 1);
      expect(service.state.retrainMode).toBe(false);
    });

    it('publishes the correction, so a subscriber can act on it', () => {
      service.activate(false);
      const seen: boolean[] = [];
      const sub = service.state$.subscribe((s) => seen.push(s.retrainMode));
      service.noteInitialLabelset(3, 2);
      sub.unsubscribe();
      expect(seen).toEqual([false, true]);
    });
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

  it('should walk for more goods once the good and bad quorum is met', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4); // badToStart default is 4
    expect(service.state.phase).toBe('more');
  });

  it('should transition from more to hard when the walk reaches its good target', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);
    service.checkPhaseTransition(20, 4); // moreToStart default is 20
    expect(service.state.phase).toBe('hard');
    expect(service.state.moreDone).toBe(true);
  });

  describe('the more walk (#4282)', () => {
    function enterWalk(): void {
      service.activate();
      service.checkPhaseTransition(3, 0);
      service.checkPhaseTransition(3, 4);
      expect(service.state.phase).toBe('more');
    }

    it('runs dry after moreDryRun misses in a row', () => {
      enterWalk();
      let bad = 4;
      for (let i = 0; i < 15; i++) {
        service.checkPhaseTransition(3, ++bad);
        expect(service.state.phase).toBe('more');
      }
      service.checkPhaseTransition(3, ++bad); // the 16th miss in a row
      expect(service.state.phase).toBe('hard');
      expect(service.state.moreDone).toBe(true);
    });

    it('restarts the run of misses on a hit', () => {
      enterWalk();
      let bad = 4;
      for (let i = 0; i < 15; i++) service.checkPhaseTransition(3, ++bad);
      service.checkPhaseTransition(4, bad); // a good
      expect(service.state.moreMisses).toBe(0);
      for (let i = 0; i < 15; i++) service.checkPhaseTransition(4, ++bad);
      expect(service.state.phase).toBe('more');
    });

    it('does not count the bad phase\'s own votes as walk misses', () => {
      enterWalk();
      expect(service.state.moreMisses).toBe(0);
    });

    it('ignores repeated checks with unchanged counts', () => {
      enterWalk();
      service.checkPhaseTransition(3, 5);
      service.checkPhaseTransition(3, 5);
      service.checkPhaseTransition(3, 5);
      expect(service.state.moreMisses).toBe(1);
    });

    it('never resumes once it has ended', () => {
      enterWalk();
      service.checkPhaseTransition(20, 4);
      service.checkPhaseTransition(19, 4); // a positive un-voted
      expect(service.state.phase).toBe('hard');
    });

    it('is skipped in retrain mode, where there is no seed sort to walk', () => {
      service.activate(true);
      service.checkPhaseTransition(3, 4);
      expect(service.state.phase).toBe('hard');
    });

    it('is skipped when the first labelset reading turns retrain mode on', () => {
      service.activate();
      service.noteInitialLabelset(3, 4);
      service.checkPhaseTransition(3, 4);
      expect(service.state.phase).toBe('hard');
    });

    it('caps its target at what a small dataset can still supply', () => {
      service.activate();
      service.checkPhaseTransition(3, 4, 9); // two unlabeled items left
      expect(service.state.phase).toBe('more');
      service.checkPhaseTransition(5, 4, 9); // both were goods: nothing left
      expect(service.state.phase).not.toBe('more');
    });

    it('starts afresh on a new run', () => {
      enterWalk();
      service.checkPhaseTransition(20, 4);
      service.clear();
      enterWalk();
      expect(service.state.moreDone).toBe(false);
    });
  });

  describe('the good walk runs dry (#4731)', () => {
    /** One Good (an example, or the first find), then `misses` Bads one vote at a time. */
    function walk(misses: number, good = 1): void {
      service.activate();
      service.checkPhaseTransition(0, 0);
      service.checkPhaseTransition(good, 0);
      for (let bad = 1; bad <= misses; bad++) service.checkPhaseTransition(good, bad);
    }

    it('ends the good phase after moreDryRun misses in a row with a good in hand', () => {
      walk(15);
      expect(service.state.phase).toBe('good');
      expect(service.state.goodMisses).toBe(15);
      service.checkPhaseTransition(1, 16); // the 16th miss in a row
      expect(service.state.goodRanDry).toBe(true);
      // The Bads it cast meet the bad quorum, and the more walk down the same sort is spent.
      expect(service.state.moreDone).toBe(true);
      expect(service.state.phase).toBe('hard');
    });

    it('does not count misses before the first good', () => {
      service.activate();
      service.checkPhaseTransition(0, 0);
      for (let bad = 1; bad <= 30; bad++) service.checkPhaseTransition(0, bad);
      expect(service.state.phase).toBe('good');
      expect(service.state.goodMisses).toBe(0);
      service.checkPhaseTransition(1, 30);
      for (let bad = 31; bad <= 45; bad++) service.checkPhaseTransition(1, bad);
      expect(service.state.phase).toBe('good');
      service.checkPhaseTransition(1, 46);
      expect(service.state.phase).toBe('hard');
    });

    it('restarts the run on a good', () => {
      walk(15);
      service.checkPhaseTransition(2, 15);
      expect(service.state.goodMisses).toBe(0);
      for (let bad = 16; bad <= 30; bad++) service.checkPhaseTransition(2, bad);
      expect(service.state.phase).toBe('good');
    });

    it('leaves the app as it was when the third good comes first', () => {
      service.activate();
      service.checkPhaseTransition(0, 0);
      service.checkPhaseTransition(3, 0);
      expect(service.state.phase).toBe('bad');
      expect(service.state.goodRanDry).toBe(false);
    });

    it('stays ended if the good count later falls', () => {
      walk(16, 2);
      expect(service.state.phase).toBe('hard');
      service.checkPhaseTransition(1, 16); // a good un-voted
      expect(service.state.phase).toBe('hard');
    });

    it('starts afresh on a new run', () => {
      walk(16);
      service.clear();
      service.activate();
      expect(service.state.goodRanDry).toBe(false);
      expect(service.state.goodMisses).toBe(0);
    });
  });

  describe('a document dataset stops on the dry run (#4488)', () => {
    const off: StatusIndicator = { status: 'off' };
    const documentStatus: LabelingStatusResponse = {
      ...makeStatus(off, off, off),
      stop_rule: 'dry_run',
      dry_run: { status: 'red', run: 0, target: 16 },
    };

    function enterDocumentWalk(retrainMode = false): void {
      service.activate(retrainMode);
      service.updateFromLabelingStatus(documentStatus);
      service.checkPhaseTransition(3, 0);
      service.checkPhaseTransition(3, 4);
      expect(service.state.phase).toBe('more');
    }

    it('reads the stop rule off the labeling status', () => {
      service.activate();
      service.updateFromLabelingStatus(documentStatus);
      expect(service.state.dryRunStop).toBe(true);
      service.updateFromLabelingStatus(makeStatus(off, off, off));
      expect(service.state.dryRunStop).toBe(false);
    });

    it('is done after moreDryRun misses in a row, with no Boundary or Diversity step', () => {
      enterDocumentWalk();
      let bad = 4;
      for (let i = 0; i < 15; i++) {
        service.checkPhaseTransition(3, ++bad);
        expect(service.state.phase).toBe('more');
      }
      service.checkPhaseTransition(3, ++bad);
      expect(service.state.phase).toBe('done');
    });

    it('walks past the twenty-Good target', () => {
      enterDocumentWalk();
      service.checkPhaseTransition(25, 4);
      expect(service.state.phase).toBe('more');
    });

    it('ignores the lights', () => {
      enterDocumentWalk();
      const green: StatusIndicator = { status: 'green' };
      service.updateFromLabelingStatus({ ...documentStatus, smart: green, stable: green, span: green });
      service.checkPhaseTransition(3, 5);
      expect(service.state.phase).toBe('more');
    });

    it('walks in retrain mode too, and still stops', () => {
      enterDocumentWalk(true);
      let bad = 4;
      for (let i = 0; i < 16; i++) service.checkPhaseTransition(3, ++bad);
      expect(service.state.phase).toBe('done');
    });

    it('a Good restarts the run', () => {
      enterDocumentWalk();
      let bad = 4;
      for (let i = 0; i < 15; i++) service.checkPhaseTransition(3, ++bad);
      service.checkPhaseTransition(4, bad);
      for (let i = 0; i < 15; i++) service.checkPhaseTransition(4, ++bad);
      expect(service.state.phase).toBe('more');
      expect(service.state.moreMisses).toBe(15);
    });

    it('is exhausted when the dataset runs out before the run does', () => {
      service.activate();
      service.updateFromLabelingStatus(documentStatus);
      service.checkPhaseTransition(5, 4, 9);
      expect(service.state.phase).toBe('exhausted');
    });
  });

  it('should transition from hard to new when smart and stable are green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);
    service.checkPhaseTransition(20, 4); // the walk meets its target

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
    service.checkPhaseTransition(20, 4); // the walk meets its target

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
    service.checkPhaseTransition(20, 4); // the walk meets its target

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
    service.checkPhaseTransition(20, 4); // the walk meets its target

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
    service.checkPhaseTransition(20, 4); // the walk meets its target

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
    service.checkPhaseTransition(20, 4); // the walk meets its target

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

  describe('doneReached (#4621)', () => {
    const green = makeStatus({ status: 'green' }, { status: 'green' }, { status: 'green' });

    /** Run to Done: the quorum, the walk's target, and every indicator green. */
    function reachDone(): void {
      service.activate();
      service.updateFromLabelingStatus(green);
      service.checkPhaseTransition(20, 4);
      expect(service.state.phase).toBe('done');
    }

    it('starts false', () => {
      service.activate();
      service.checkPhaseTransition(20, 4);
      expect(service.state.phase).toBe('hard');
      expect(service.state.doneReached).toBe(false);
    });

    it('is set on reaching Done', () => {
      reachDone();
      expect(service.state.doneReached).toBe(true);
    });

    it('holds while the phase follows the indicators back to hard and new', () => {
      reachDone();
      service.updateFromLabelingStatus(
        makeStatus({ status: 'green' }, { status: 'yellow' }, { status: 'green' }),
      );
      service.checkPhaseTransition(21, 4);
      // The phase itself is unchanged by the latch: it still drives the picks.
      expect(service.state.phase).toBe('hard');
      expect(service.state.doneReached).toBe(true);

      service.updateFromLabelingStatus(
        makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }),
      );
      service.checkPhaseTransition(21, 5);
      expect(service.state.phase).toBe('new');
      expect(service.state.doneReached).toBe(true);
    });

    it('clears when votes fall back into the opening', () => {
      reachDone();
      service.checkPhaseTransition(1, 4);
      expect(service.state.phase).toBe('good');
      expect(service.state.doneReached).toBe(false);
    });

    it('resets on a new run and on clear', () => {
      reachDone();
      service.deactivate();
      service.activate();
      expect(service.state.doneReached).toBe(false);

      reachDone();
      service.clear();
      expect(service.state.doneReached).toBe(false);
    });
  });

  it('should cascade good→bad→more in a single checkPhaseTransition call', () => {
    service.activate();
    // Both thresholds met at once (user labeled in Manual before switching to Autopilot)
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('more');
  });

  it('should cascade all the way to hard when the walk\'s target is already met', () => {
    service.activate();
    service.checkPhaseTransition(25, 10);
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
    expect(service.state.stablePlateau).toBe(false);
    expect(service.state.spanStatus).toBe('red');
    expect(service.state.fracDiversity).toBe(0.75);
  });

  it('updateFromLabelingStatus should carry the Stable plateau flag', () => {
    service.activate();
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green', plateau: true }, { status: 'green' }),
    );
    expect(service.state.stablePlateau).toBe(true);

    service.updateFromLabelingStatus(makeStatus({ status: 'green' }, { status: 'green' }, { status: 'green' }));
    expect(service.state.stablePlateau).toBe(false);
  });

  it('updateFromLabelingStatus should carry the Span green target, keeping it when unreported', () => {
    service.activate();
    expect(service.state.spanTarget).toBe(0);
    service.updateFromLabelingStatus(
      makeStatus({ status: 'green' }, { status: 'green' }, { status: 'red', diversity_level: 5, target: 40 }),
    );
    expect(service.state.spanTarget).toBe(40);

    service.updateFromLabelingStatus(makeStatus({ status: 'green' }, { status: 'green' }, { status: 'yellow' }));
    expect(service.state.spanTarget).toBe(40);
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
