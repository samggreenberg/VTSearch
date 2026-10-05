import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LineTestSessionService } from './line-test-session.service';
import { PairScopeService } from './pair-scope.service';
import { VoteStateService } from './vote-state.service';
import { provideHttpTesting } from '../testing/test-providers';
import { provideZoneless } from '../testing/zoneless-testbed';
import { wireDone, wireLineTest, wireTest } from '../testing/line-test';

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

  it('starts in the score phase with nothing on the table', () => {
    expect(service.phase()).toBe('score');
    expect(service.running()).toBe(false);
    expect(service.locked()).toBe(false);
  });

  it('start deals the round as dealt: every pick with no vote, the first on screen', () => {
    start();
    expect(service.phase()).toBe('matches');
    expect(service.running()).toBe(true);
    expect(service.locked()).toBe(true);
    expect(service.roundPicks()).toEqual([51, 36, 41, 44, 48]);
    expect(service.currentId()).toBe(51);
    expect(service.roundVotes().size).toBe(0);
  });

  it('a vote goes out as it is cast, moves to the next unvoted pick, and the round keeps the vote', () => {
    start();
    service.vote('good');
    expect(service.currentId()).toBe(36);
    expect(service.roundVotes().get(51)).toBe('good');
    const req = httpMock.expectOne('/api/line-test/votes');
    expect(req.request.body).toEqual({ votes: [{ id: 51, label: 'good' }] });
    req.flush(wireLineTest(wireTest({ picks: [36, 41, 44, 48], labelled: 1 })));
    // The dealt round stays: the server's pending picks only say the vote landed.
    expect(service.roundPicks()).toEqual([51, 36, 41, 44, 48]);
    expect(service.roundVotes().get(51)).toBe('good');
    expect(TestBed.inject(VoteStateService).loadVotes).toHaveBeenCalled();
  });

  it('the round\'s last vote brings the next round, dealt afresh', () => {
    start([1, 2]);
    service.vote('good');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest({ picks: [2], labelled: 1 })));
    service.vote('bad');
    httpMock
      .expectOne('/api/line-test/votes')
      .flush(wireLineTest(wireTest({ picks: [7, 8, 9, 10, 11], round: 2, labelled: 2 })));
    expect(service.roundPicks()).toEqual([7, 8, 9, 10, 11]);
    expect(service.roundVotes().size).toBe(0);
    expect(service.index()).toBe(0);
  });

  it('↓ steps back a pick and takes its vote back; ↑ goes on to the next unvoted', () => {
    start();
    service.vote('good');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest({ picks: [36, 41, 44, 48], labelled: 1 })));
    service.navigate('back');
    expect(service.index()).toBe(0);
    expect(service.roundVotes().has(51)).toBe(false);
    const req = httpMock.expectOne('/api/line-test/unvote');
    expect(req.request.body).toEqual({ id: 51 });
    req.flush(wireLineTest(wireTest()));
    service.navigate('forward');
    expect(service.index()).toBe(1);
  });

  it('↓ on the first pick with no vote sends nothing', () => {
    start();
    service.navigate('back');
    expect(service.index()).toBe(0);
    httpMock.expectNone('/api/line-test/unvote');
  });

  it('done clears the table and frees the balance', () => {
    start([1]);
    service.vote('good');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireDone()));
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
    second.flush(wireLineTest(wireTest({ picks: [3], labelled: 2 })));
    first.flush(wireLineTest(wireTest({ picks: [2, 3], labelled: 1 })));
    expect(service.test()!.labelled).toBe(2);
    expect(service.roundVotes().get(1)).toBe('good');
    expect(service.roundVotes().get(2)).toBe('bad');
  });

  it('a refused vote is rolled back, the pick back on screen, and the refusal said', () => {
    start();
    service.vote('good');
    expect(service.currentId()).toBe(36);
    httpMock.expectOne('/api/line-test/votes').flush({ message: 'No test is running.' }, { status: 409, statusText: 'Conflict' });
    expect(service.error()).toBe('No test is running.');
    expect(service.busy()).toBe(false);
    expect(service.roundVotes().has(51)).toBe(false);
    expect(service.currentId()).toBe(51);
  });

  it('a fast voter\'s next vote goes out while the last is still in flight', () => {
    start();
    service.vote('good');
    service.vote('bad');
    const sent = httpMock.match('/api/line-test/votes');
    expect(sent.map((r) => r.request.body)).toEqual([{ votes: [{ id: 51, label: 'good' }] }, { votes: [{ id: 36, label: 'bad' }] }]);
    expect(service.busy()).toBe(true);
    sent[0].flush(wireLineTest(wireTest({ picks: [36, 41, 44, 48], labelled: 1 })));
    expect(service.busy()).toBe(true);
    sent[1].flush(wireLineTest(wireTest({ picks: [41, 44, 48], labelled: 2 })));
    expect(service.busy()).toBe(false);
    expect(service.roundVotes().size).toBe(2);
    expect(service.currentId()).toBe(41);
  });

  it('load reads a running test off the server; cancel drops it; clear forgets it', () => {
    service.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest()));
    expect(service.running()).toBe(true);
    service.cancel();
    httpMock.expectOne('/api/line-test/cancel').flush(wireLineTest(null));
    expect(service.phase()).toBe('score');
    expect(service.roundPicks()).toEqual([]);
    service.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest()));
    service.clear();
    expect(service.response()).toBeNull();
  });
});
