import { TestBed } from '@angular/core/testing';
import { HttpTestingController, type TestRequest } from '@angular/common/http/testing';
import { SpotCheckRunnerService } from './spot-check-runner.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';
import { wireBalance } from '../../testing/line-balance';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { MediaStateService } from '../../services/media-state.service';
import { PairScopeService } from '../../services/pair-scope.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { VoteStateService } from '../../services/vote-state.service';

/** A running walk's wire state (#4388, #4413), dealing *picks*. */
function running(picks: number[], round = 1) {
  return {
    status: 'running',
    beta: 1,
    round,
    rounds: 6,
    picks_per_round: picks.length,
    candidate: 32,
    start_candidate: 32,
    bands: 3,
    band: { index: round - 1, lo: 1, hi: 8 },
    direction: 'start',
    estimate: null,
    fbeta: null,
    recall: null,
    picks,
    labelled: 0,
    right: 0,
    range: null,
  };
}

/** A finished check's wire state. */
function finished() {
  const range = { lo: 0.55, hi: 1, labelled: 2, right: 2 };
  return { ...running([]), status: 'checked', band: null, labelled: 2, right: 2, range, recall: range, fbeta: 0.7 };
}

const response = (check: unknown) => ({ balance: wireBalance('unchecked'), check });

describe('SpotCheckRunnerService (#5136)', () => {
  let runner: SpotCheckRunnerService;
  let httpMock: HttpTestingController;
  let mediaState: MediaStateService;
  let voteState: VoteStateService;
  let ended: { finished: boolean; owed: boolean }[];

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideZoneless(), ...provideHttpTesting(), PairScopeService, SpotCheckRunnerService],
    });
    httpMock = TestBed.inject(HttpTestingController);
    mediaState = TestBed.inject(MediaStateService);
    voteState = TestBed.inject(VoteStateService);
    vi.spyOn(TestBed.inject(MediaMetadataCacheService), 'ensureLoaded').mockImplementation(() => undefined);
    runner = TestBed.inject(SpotCheckRunnerService);
    ended = [];
    runner.ended = (f, owed) => ended.push({ finished: f, owed });
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({});
    });
  });

  const startReq = (): TestRequest => httpMock.expectOne((req) => req.url === '/api/precision-check/start');
  const votesReq = (): TestRequest => httpMock.expectOne((req) => req.url === '/api/precision-check/votes');
  const cancelReqs = () => httpMock.match((req) => req.url === '/api/precision-check/cancel');
  /** The labels the server holds, as the vote poll would land them. */
  const labels = (good: number[], bad: number[]) => {
    voteState.loadVotes();
    httpMock.expectOne('/api/votes').flush({ good, bad, click_times: {}, learned_scores: {} });
  };
  const started = (picks: number[]) => {
    runner.start();
    startReq().flush(response(running(picks)));
  };

  it('draws the first round without moving the item on screen', () => {
    mediaState.selectMedia(99);
    started([3, 5]);
    expect(mediaState.selectedId()).toBe(99);
    expect(runner.isPick(3)).toBe(true);
    expect(runner.isPick(99)).toBe(false);
    expect(runner.advancePending()).toBe(false);
    expect(runner.fresh()).toBe(true);
  });

  it('is fresh until the check has its first vote', () => {
    started([3, 5]);
    labels([3], []);
    expect(runner.fresh()).toBe(false);
    labels([3, 5], []);
    runner.advance(5);
    votesReq().flush(response(running([8, 9], 2)));
    expect(runner.fresh()).toBe(false);
  });

  it('shows the round in draw order, one unlabeled pick per advance', () => {
    started([5, 3, 7]);
    runner.advance(99);
    expect(mediaState.selectedId()).toBe(5);
    labels([5], []);
    runner.advance(5);
    expect(mediaState.selectedId()).toBe(3);
    expect(runner.upcoming(3, 4)).toEqual([7]);
    httpMock.expectNone((req) => req.url === '/api/precision-check/votes');
  });

  it('makes an advance that came before the picks once they land', () => {
    runner.start();
    runner.advance(99);
    expect(runner.advancePending()).toBe(true);
    startReq().flush(response(running([4, 6])));
    expect(mediaState.selectedId()).toBe(4);
    expect(runner.advancePending()).toBe(false);
  });

  it('sends a round whole, with the labels its picks carry then, and deals the next', () => {
    started([3, 5]);
    // 3 was voted Good, then put right to Bad, before the round was whole.
    labels([5], [3]);
    runner.advance(5);
    const req = votesReq();
    expect(req.request.body).toEqual({
      votes: [
        { id: 3, label: 'bad' },
        { id: 5, label: 'good' },
      ],
    });
    expect(runner.advancePending()).toBe(true);
    req.flush(response(running([8, 9], 2)));
    expect(mediaState.selectedId()).toBe(8);
    expect(runner.advancePending()).toBe(false);
    expect(runner.isPick(3)).toBe(false);
  });

  it('shows an undone pick again rather than sending the round', () => {
    started([3, 5]);
    labels([5], []);
    runner.advance(5);
    expect(mediaState.selectedId()).toBe(3);
    httpMock.expectNone((req) => req.url === '/api/precision-check/votes');
  });

  it('comes back to the voted pick when a vote toggled its label off and it is the last one owed', () => {
    started([3, 5]);
    labels([5], []);
    runner.advance(3);
    expect(mediaState.selectedId()).toBe(3);
  });

  it('reports a finished check to the host, with the advance it owes', () => {
    started([3]);
    labels([3], []);
    runner.advance(3);
    votesReq().flush(response(finished()));
    expect(ended).toEqual([{ finished: true, owed: true }]);
    expect(runner.isPick(3)).toBe(false);
    expect(runner.advancePending()).toBe(false);
  });

  it('ends quietly when the server refuses to start, owing no advance', () => {
    mediaState.selectMedia(99);
    runner.start();
    startReq().flush({ message: 'Nothing is left unvoted to check.' }, { status: 409, statusText: 'Conflict' });
    expect(ended).toEqual([{ finished: false, owed: false }]);
    expect(mediaState.selectedId()).toBe(99);
  });

  it('cancels and ends quietly when a round fails to send', () => {
    started([3]);
    labels([], [3]);
    runner.advance(3);
    votesReq().flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(cancelReqs().length).toBe(1);
    expect(ended).toEqual([{ finished: false, owed: true }]);
  });

  it('hands the check the picks voted this round before cancelling it', () => {
    started([3, 5]);
    labels([3], []);
    runner.cancel();
    const req = votesReq();
    expect(req.request.body).toEqual({ votes: [{ id: 3, label: 'good' }] });
    expect(cancelReqs()).toEqual([]);
    req.flush(response(running([5])));
    expect(cancelReqs().length).toBe(1);
    expect(ended).toEqual([]);
    expect(runner.isPick(5)).toBe(false);
  });

  it('cancels outright when nothing was voted this round, and is a no-op with no check', () => {
    runner.cancel();
    expect(cancelReqs()).toEqual([]);
    started([3, 5]);
    runner.cancel();
    expect(cancelReqs().length).toBe(1);
    httpMock.expectNone((req) => req.url === '/api/precision-check/votes');
  });

  it('cancels a running check when it goes away', () => {
    started([3]);
    const cancel = vi.spyOn(TestBed.inject(SortingApiService), 'cancelPrecisionCheck');
    TestBed.resetTestingModule();
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it('drops a late answer to a check it cancelled', () => {
    mediaState.selectMedia(99);
    runner.start();
    const req = startReq();
    runner.advance(99);
    runner.cancel();
    expect(req.cancelled).toBe(true);
    expect(cancelReqs().length).toBe(1);
    expect(runner.advancePending()).toBe(false);
    expect(mediaState.selectedId()).toBe(99);
  });

  it('forgets the check on a pair switch, and draws a fresh one at the next advance', () => {
    started([3]);
    runner.reset();
    expect(runner.isPick(3)).toBe(false);
    httpMock.expectNone(() => true);
    runner.advance();
    expect(runner.advancePending()).toBe(true);
    startReq().flush(response(running([11])));
    expect(mediaState.selectedId()).toBe(11);
  });
});
