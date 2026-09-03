import { TestBed } from '@angular/core/testing';

import { AutopilotStateService } from './autopilot-state.service';
import { SortStateService } from './sort-state.service';
import { VoteProvenanceService } from './vote-provenance.service';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The provenance assembler is the one place in the client that decides what a
 * vote's surfacing context *was*, so its job is to read the state already on
 * screen rather than to be told. These tests pin the two derivations that are
 * easy to get subtly wrong: which flow autopilot actually owns, and that the
 * rank recorded is the item's real position in the ranking.
 */
describe('VoteProvenanceService', () => {
  let service: VoteProvenanceService;
  let sortState: SortStateService;
  let autopilot: AutopilotStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    service = TestBed.inject(VoteProvenanceService);
    sortState = TestBed.inject(SortStateService);
    autopilot = TestBed.inject(AutopilotStateService);
  });

  afterEach(() => autopilot.clear());

  it('records list_review when autopilot is idle', () => {
    expect(service.forVote(1).flow).toBe('list_review');
  });

  it('records the autopilot phase while autopilot is surfacing items', () => {
    autopilot.activate();
    autopilot.checkPhaseTransition(3, 4, 100);
    expect(autopilot.state.phase).toBe('hard');

    const p = service.forVote(1);
    expect(p.flow).toBe('autopilot');
    expect(p.phase).toBe('hard');
  });

  it('records the initial good phase, which is a top-of-list draw', () => {
    // The other case a fused enum gets wrong: autopilot's `good` phase reads
    // as "autopilot, therefore safe" while being mechanically the same
    // top-of-list draw as manual review.
    autopilot.activate();
    autopilot.checkPhaseTransition(0, 0, 100);

    const p = service.forVote(1);
    expect(p.flow).toBe('autopilot');
    expect(p.phase).toBe('good');
  });

  it('does not claim autopilot in its terminal phases', () => {
    // `exhausted` (and `done`) leave the user labelling off the last sort by
    // hand, which is list review whatever the panel still says.
    autopilot.activate();
    autopilot.checkPhaseTransition(3, 4, 7); // nothing left unlabeled
    expect(autopilot.state.phase).toBe('exhausted');

    const p = service.forVote(1);
    expect(p.flow).toBe('list_review');
    expect(p.phase).toBeUndefined();
  });

  it('records the sort and select modes as independent axes', () => {
    // The case a fused `autopilot:hard` enum could not express: a user picking
    // the margin-sampled draw by hand, outside autopilot.
    sortState.setSortMode('learned');
    sortState.setSelectMode('hard');
    const p = service.forVote(1);
    expect(p.flow).toBe('list_review');
    expect(p.select_mode).toBe('hard');
    expect(p.sort_kind).toBe('learned');
  });

  it('records the item rank and score from the ranking on screen', () => {
    sortState.setSortResults(
      [
        { id: 7, score: 0.9 },
        { id: 3, score: 0.5 },
        { id: 9, score: 0.1 },
      ],
      0.4,
    );
    const p = service.forVote(3);
    expect(p.rank_at_vote).toBe(1);
    expect(p.score_at_vote).toBe(0.5);
  });

  it('omits rank and score for an item absent from the ranking', () => {
    sortState.setSortResults([{ id: 7, score: 0.9 }], 0.4);
    const p = service.forVote(999);
    expect(p.rank_at_vote).toBeUndefined();
    expect(p.score_at_vote).toBeUndefined();
  });

  it('omits rank and score when no sort has run', () => {
    const p = service.forVote(1);
    expect(p.rank_at_vote).toBeUndefined();
    expect(p.score_at_vote).toBeUndefined();
  });

  it('lets a caller override the flow it knows better than the state does', () => {
    autopilot.activate();
    autopilot.checkPhaseTransition(3, 4, 100);
    const p = service.forVote(1, 'find_verify');
    expect(p.flow).toBe('find_verify');
    expect(p.phase).toBeUndefined();
  });
});
