import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LineTestSessionService } from './line-test-session.service';
import { PairScopeService } from './pair-scope.service';
import { VoteStateService } from './vote-state.service';
import { provideHttpTesting } from '../testing/test-providers';
import { provideZoneless } from '../testing/zoneless-testbed';
import { wireDone, wireLineTest, wireTest, wireVotes } from '../testing/line-test';

describe('LineTestSessionService (#4524)', () => {
  let service: LineTestSessionService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideZoneless(), ...provideHttpTesting(), PairScopeService, LineTestSessionService],
    });
    service = TestBed.inject(LineTestSessionService);
    httpMock = TestBed.inject(HttpTestingController);
    vi.spyOn(TestBed.inject(VoteStateService), 'loadVotes').mockImplementation(() => undefined);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function start(picks = [51, 36, 41, 44, 48]): void {
    service.start();
    httpMock.expectOne('/api/line-test/start').flush(wireLineTest(wireTest({ picks })));
  }

  /** Vote the pick on screen and flush the server's answer. */
  function vote(direction: 'good' | 'bad', reply: Parameters<typeof wireTest>[0]): void {
    service.vote(direction);
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest(reply)));
  }

  it('starts in the score phase with nothing on the table', () => {
    expect(service.phase()).toBe('score');
    expect(service.running()).toBe(false);
    expect(service.locked()).toBe(false);
    expect(service.currentId()).toBeNull();
  });

  it('start deals the round as dealt: every pick with no vote, the first on screen', () => {
    start();
    expect(service.phase()).toBe('matches');
    expect(service.running()).toBe(true);
    expect(service.locked()).toBe(true);
    expect(service.roundPicks()).toEqual([51, 36, 41, 44, 48]);
    expect(service.currentId()).toBe(51);
    expect(service.currentVote()).toBeNull();
    expect(service.votes().size).toBe(0);
  });

  it('a vote goes out as it is cast, moves to the next unvoted pick, and the test keeps the vote', () => {
    start();
    service.vote('good');
    expect(service.currentId()).toBe(36);
    expect(service.votes().get(51)).toBe('good');
    const req = httpMock.expectOne('/api/line-test/votes');
    expect(req.request.body).toEqual({ votes: [{ id: 51, label: 'good' }] });
    req.flush(wireLineTest(wireTest({ picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) })));
    // The dealt round stays: the server's pending picks only say the vote landed.
    expect(service.roundPicks()).toEqual([51, 36, 41, 44, 48]);
    expect(service.votes().get(51)).toBe('good');
    expect(service.trail()).toEqual([51]);
  });

  it('votes on the pick the centre panel names, not whatever is on screen when the swipe ends', () => {
    start();
    service.vote('bad', 36);
    expect(httpMock.expectOne('/api/line-test/votes').request.body).toEqual({ votes: [{ id: 36, label: 'bad' }] });
    expect(service.votes().get(36)).toBe('bad');
    expect(service.currentId()).toBe(41);
    // Not a pick of the test: nothing goes out.
    service.vote('good', 999);
    httpMock.expectNone('/api/line-test/votes');
  });

  it('a vote or a correction never re-reads the vote piles: Review reads them when it opens (#5099)', () => {
    const loadVotes = TestBed.inject(VoteStateService).loadVotes;
    start();
    vote('good', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) });
    service.navigate('back');
    vote('bad', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'bad']) });
    expect(loadVotes).not.toHaveBeenCalled();
  });

  it('names the next picks a vote brings on screen, in order, skipping the voted and wrapping round (#5099)', () => {
    expect(service.upcoming(2)).toEqual([]);
    start();
    expect(service.upcoming(2)).toEqual([36, 41]);
    vote('good', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) });
    expect(service.currentId()).toBe(36);
    expect(service.upcoming(2)).toEqual([41, 44]);
    // From the last pick the walk wraps to the first unvoted one, past 51.
    service.index.set(4);
    expect(service.upcoming(2)).toEqual([36, 41]);
    // Never more than the round has left, and never the pick on screen.
    expect(service.upcoming(10)).toEqual([36, 41, 44]);
  });

  it('with one pick left the next round is upcoming both ways, in turns, then the way the vote went (#5116)', () => {
    start([1, 2]);
    const ahead = { id: 2, good: [7, 8, 9], bad: [20, 21, 22] };
    vote('good', { picks: [2], labelled: 1, ahead, votes: wireVotes([1, 'good']) });
    expect(service.currentId()).toBe(2);
    expect(service.upcoming(2)).toEqual([7, 20]);
    expect(service.upcoming(4)).toEqual([7, 20, 8, 21]);
    service.vote('bad');
    // The vote is out and its reply not back: the next round is the one it names.
    expect(service.awaitingRound()).toBe(true);
    expect(service.upcoming(2)).toEqual([20, 21]);
    httpMock
      .expectOne('/api/line-test/votes')
      .flush(wireLineTest(wireTest({ picks: [20, 21, 22], round: 2, labelled: 2, votes: wireVotes([1, 'good'], [2, 'bad']) })));
    expect(service.awaitingRound()).toBe(false);
    expect(service.currentId()).toBe(20);
    expect(service.upcoming(2)).toEqual([21, 22]);
  });

  it('both ways dealing the same round names its picks once each (#5116)', () => {
    start([1, 2]);
    const ahead = { id: 2, good: [7, 8, 9], bad: [7, 8, 9] };
    vote('good', { picks: [2], labelled: 1, ahead, votes: wireVotes([1, 'good']) });
    expect(service.upcoming(2)).toEqual([7, 8]);
  });

  it('the round\'s last vote brings the next round, dealt afresh, and the test keeps every earlier vote', () => {
    start([1, 2]);
    vote('good', { picks: [2], labelled: 1, votes: wireVotes([1, 'good']) });
    vote('bad', { picks: [7, 8, 9, 10, 11], round: 2, labelled: 2, votes: wireVotes([1, 'good'], [2, 'bad']) });
    expect(service.roundPicks()).toEqual([7, 8, 9, 10, 11]);
    expect(service.index()).toBe(0);
    expect(service.currentId()).toBe(7);
    expect([...service.votes()]).toEqual([[1, 'good'], [2, 'bad']]);
  });

  it('↓ walks back along the picks voted, with their votes lit, and ↑ returns to the pick waiting (#5502)', () => {
    start();
    vote('good', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) });
    vote('bad', { picks: [41, 44, 48], labelled: 2, votes: wireVotes([51, 'good'], [36, 'bad']) });
    service.navigate('back');
    expect(service.currentId()).toBe(36);
    expect(service.currentVote()).toBe('bad');
    service.navigate('back');
    expect(service.currentId()).toBe(51);
    expect(service.currentVote()).toBe('good');
    // Nothing older: it stays put.
    service.navigate('back');
    expect(service.currentId()).toBe(51);
    service.navigate('forward');
    expect(service.currentId()).toBe(41);
    expect(service.currentVote()).toBeNull();
    // A walk sends nothing.
    httpMock.expectNone('/api/line-test/votes');
  });

  it('the last pick of a round is walked back to and corrected after the next round is dealt (#5502)', () => {
    start([1, 2]);
    vote('good', { picks: [2], labelled: 1, votes: wireVotes([1, 'good']) });
    vote('bad', { picks: [7, 8, 9], round: 2, labelled: 2, votes: wireVotes([1, 'good'], [2, 'bad']) });
    expect(service.currentId()).toBe(7);

    service.navigate('back');
    expect(service.currentId()).toBe(2);
    expect(service.currentVote()).toBe('bad');
    service.vote('good');
    const req = httpMock.expectOne('/api/line-test/votes');
    expect(req.request.body).toEqual({ votes: [{ id: 2, label: 'good' }] });
    // Lit at once, and back at the pick waiting for a vote, as a vote anywhere moves on.
    expect(service.votes().get(2)).toBe('good');
    expect(service.currentId()).toBe(7);
    req.flush(wireLineTest(wireTest({ picks: [7, 8, 9], round: 2, labelled: 2, votes: wireVotes([1, 'good'], [2, 'good']) })));
    expect(service.roundPicks()).toEqual([7, 8, 9]);
    expect(service.votes().get(2)).toBe('good');
  });

  it('the same vote again on a pick walked back to sends nothing and returns to the pick waiting', () => {
    start();
    vote('good', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) });
    service.navigate('back');
    service.vote('good');
    httpMock.expectNone('/api/line-test/votes');
    expect(service.currentId()).toBe(36);
  });

  it('a refused correction puts the vote back as it was', () => {
    start();
    vote('good', { picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) });
    service.navigate('back');
    service.vote('bad');
    httpMock.expectOne('/api/line-test/votes').flush({ message: 'Not a pick of this test.' }, { status: 400, statusText: 'Bad Request' });
    expect(service.votes().get(51)).toBe('good');
    expect(service.error()).toBe('Not a pick of this test.');
  });

  it('from Done ↓ brings the last pick voted back on screen, and a correction may reopen the test (#5502)', () => {
    start([1]);
    service.vote('good');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireDone({ votes: wireVotes([1, 'good']) })));
    expect(service.phase()).toBe('done');
    expect(service.currentId()).toBeNull();

    service.navigate('back');
    expect(service.currentId()).toBe(1);
    expect(service.currentVote()).toBe('good');
    service.navigate('forward');
    expect(service.currentId()).toBeNull();

    service.navigate('back');
    service.vote('bad');
    expect(service.currentId()).toBeNull();
    httpMock
      .expectOne('/api/line-test/votes')
      .flush(wireLineTest(wireTest({ phase: 'misses', picks: [70, 71], round: 2, labelled: 1, votes: wireVotes([1, 'bad']) })));
    expect(service.running()).toBe(true);
    expect(service.currentId()).toBe(70);
  });

  it('a start that resumed from kept picks leaves the piles to the Review tab, and can be walked back (#4526, #5099)', () => {
    service.start();
    httpMock
      .expectOne('/api/line-test/start')
      .flush(wireLineTest(wireDone({ kept_at: 1790000000, votes: wireVotes([3, 'good'], [9, 'bad']) })));
    expect(service.phase()).toBe('done');
    expect(TestBed.inject(VoteStateService).loadVotes).not.toHaveBeenCalled();
    service.navigate('back');
    expect(service.currentId()).toBe(9);
  });

  it('done clears the round and frees the balance', () => {
    start([1]);
    service.vote('good');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireDone({ votes: wireVotes([1, 'good']) })));
    expect(service.phase()).toBe('done');
    expect(service.running()).toBe(false);
    expect(service.locked()).toBe(false);
    expect(service.roundPicks()).toEqual([]);
  });

  it('an older response never overwrites a newer one', () => {
    start([1, 2, 3]);
    service.vote('good');
    const first = httpMock.expectOne('/api/line-test/votes');
    service.vote('bad');
    const second = httpMock.expectOne('/api/line-test/votes');
    second.flush(wireLineTest(wireTest({ picks: [3], labelled: 2, votes: wireVotes([1, 'good'], [2, 'bad']) })));
    first.flush(wireLineTest(wireTest({ picks: [2, 3], labelled: 1, votes: wireVotes([1, 'good']) })));
    expect(service.test()!.labelled).toBe(2);
    expect(service.votes().get(1)).toBe('good');
    expect(service.votes().get(2)).toBe('bad');
  });

  it('a reply that lands while the next vote is out does not erase it', () => {
    start([1, 2, 3]);
    service.vote('good');
    const first = httpMock.expectOne('/api/line-test/votes');
    service.vote('bad');
    const second = httpMock.expectOne('/api/line-test/votes');
    first.flush(wireLineTest(wireTest({ picks: [2, 3], labelled: 1, votes: wireVotes([1, 'good']) })));
    expect(service.votes().get(2)).toBe('bad');
    expect(service.trail()).toEqual([1, 2]);
    second.flush(wireLineTest(wireTest({ picks: [3], labelled: 2, votes: wireVotes([1, 'good'], [2, 'bad']) })));
    expect(service.trail()).toEqual([1, 2]);
  });

  it('a refused vote is rolled back, the pick back on screen, and the refusal said', () => {
    start();
    service.vote('good');
    expect(service.currentId()).toBe(36);
    httpMock.expectOne('/api/line-test/votes').flush({ message: 'No test is running.' }, { status: 409, statusText: 'Conflict' });
    expect(service.error()).toBe('No test is running.');
    expect(service.busy()).toBe(false);
    expect(service.votes().has(51)).toBe(false);
    expect(service.trail()).toEqual([]);
    expect(service.currentId()).toBe(51);
  });

  it('a fast voter\'s next vote goes out while the last is still in flight', () => {
    start();
    service.vote('good');
    service.vote('bad');
    const sent = httpMock.match('/api/line-test/votes');
    expect(sent.map((r) => r.request.body)).toEqual([{ votes: [{ id: 51, label: 'good' }] }, { votes: [{ id: 36, label: 'bad' }] }]);
    expect(service.busy()).toBe(true);
    sent[0].flush(wireLineTest(wireTest({ picks: [36, 41, 44, 48], labelled: 1, votes: wireVotes([51, 'good']) })));
    expect(service.busy()).toBe(true);
    sent[1].flush(wireLineTest(wireTest({ picks: [41, 44, 48], labelled: 2, votes: wireVotes([51, 'good'], [36, 'bad']) })));
    expect(service.busy()).toBe(false);
    expect(service.votes().size).toBe(2);
    expect(service.currentId()).toBe(41);
  });

  it('load reads a running test off the server, its votes included; cancel drops it; clear forgets it', () => {
    service.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest({ labelled: 2, votes: wireVotes([3, 'good'], [9, 'bad']) })));
    expect(service.running()).toBe(true);
    expect(service.trail()).toEqual([3, 9]);
    service.cancel();
    httpMock.expectOne('/api/line-test/cancel').flush(wireLineTest(null));
    expect(service.phase()).toBe('score');
    expect(service.roundPicks()).toEqual([]);
    expect(service.trail()).toEqual([]);
    service.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest()));
    service.clear();
    expect(service.response()).toBeNull();
  });
});
