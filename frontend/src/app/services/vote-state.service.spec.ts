import { TestBed, fakeAsync, tick, discardPeriodicTasks } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { VoteStateService } from './vote-state.service';

describe('VoteStateService', () => {
  let service: VoteStateService;
  let httpMock: HttpTestingController;

  const mockVotes = {
    good: [1, 2],
    bad: [3],
    click_times: { '1': 100, '2': 200 },
    learned_scores: { '1': 0.9 },
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(VoteStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    service.stopPolling();
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with empty votes', () => {
    expect(service.goodVotes.size).toBe(0);
    expect(service.badVotes.size).toBe(0);
    expect(service.clickTimes).toEqual({});
    expect(service.learnedScores).toEqual({});
  });

  it('loadVotes should fetch and store votes', () => {
    service.loadVotes();
    const req = httpMock.expectOne('/api/votes');
    req.flush(mockVotes);

    expect(service.goodVotes.has(1)).toBeTrue();
    expect(service.goodVotes.has(2)).toBeTrue();
    expect(service.badVotes.has(3)).toBeTrue();
    expect(service.clickTimes).toEqual({ '1': 100, '2': 200 });
    expect(service.learnedScores).toEqual({ '1': 0.9 });
  });

  it('startPolling should periodically fetch votes', fakeAsync(() => {
    service.startPolling(1000);

    // First poll at t=0
    httpMock.expectOne('/api/votes').flush(mockVotes);
    expect(service.goodVotes.size).toBe(2);

    // Second poll at t=1000
    tick(1000);
    httpMock.expectOne('/api/votes').flush({ good: [1], bad: [], click_times: {}, learned_scores: {} });
    expect(service.goodVotes.size).toBe(1);

    service.stopPolling();
    discardPeriodicTasks();
  }));

  it('stopPolling should stop periodic fetches', fakeAsync(() => {
    service.startPolling(1000);
    httpMock.expectOne('/api/votes').flush(mockVotes);

    service.stopPolling();
    tick(1000);
    httpMock.expectNone('/api/votes');

    discardPeriodicTasks();
  }));

  it('clear should reset all state', () => {
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush(mockVotes);

    service.clear();
    expect(service.goodVotes.size).toBe(0);
    expect(service.badVotes.size).toBe(0);
    expect(service.clickTimes).toEqual({});
    expect(service.learnedScores).toEqual({});
  });

  it('applyOptimisticVote should add to good and set click time', () => {
    service.applyOptimisticVote(5, 'good');
    expect(service.goodVotes.has(5)).toBeTrue();
    expect(service.badVotes.has(5)).toBeFalse();
    expect(service.clickTimes['5']).toBe(1);
  });

  it('applyOptimisticVote should add to bad and set click time', () => {
    service.applyOptimisticVote(5, 'bad');
    expect(service.badVotes.has(5)).toBeTrue();
    expect(service.goodVotes.has(5)).toBeFalse();
    expect(service.clickTimes['5']).toBe(1);
  });

  it('applyOptimisticVote click time should exceed existing max', () => {
    // Load votes with existing click times
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush(mockVotes);
    // mockVotes has click_times: { '1': 100, '2': 200 }

    service.applyOptimisticVote(7, 'good');
    expect(service.clickTimes['7']).toBe(201);
  });

  it('applyOptimisticVote toggle-off should not set click time', () => {
    service.applyOptimisticVote(5, 'good');
    const timeAfterAdd = service.clickTimes['5'];
    expect(timeAfterAdd).toBe(1);

    // Toggle off
    service.applyOptimisticVote(5, 'good');
    expect(service.goodVotes.has(5)).toBeFalse();
    // Click time should remain unchanged (no new time set on removal)
    expect(service.clickTimes['5']).toBe(1);
  });

  it('applyOptimisticVote should move from bad to good with new click time', () => {
    service.applyOptimisticVote(5, 'bad');
    expect(service.clickTimes['5']).toBe(1);

    service.applyOptimisticVote(5, 'good');
    expect(service.goodVotes.has(5)).toBeTrue();
    expect(service.badVotes.has(5)).toBeFalse();
    expect(service.clickTimes['5']).toBe(2);
  });

  it('applyVotes should preserve optimistic vote when server has not caught up', () => {
    // Simulate optimistic vote
    service.applyOptimisticVote(10, 'good');
    expect(service.goodVotes.has(10)).toBeTrue();
    expect(service.clickTimes['10']).toBe(1);

    // Server response arrives WITHOUT the new vote (stale data)
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush({
      good: [1, 2],
      bad: [3],
      click_times: { '1': 100, '2': 200 },
      learned_scores: {},
    });

    // Optimistic vote should be preserved
    expect(service.goodVotes.has(10)).toBeTrue();
    expect(service.clickTimes['10']).toBe(1);
    // Server data should also be present
    expect(service.goodVotes.has(1)).toBeTrue();
    expect(service.goodVotes.has(2)).toBeTrue();
  });

  it('applyVotes should clear optimistic tracking once server confirms', () => {
    // Simulate optimistic vote
    service.applyOptimisticVote(10, 'good');

    // Server response now includes the voted item
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush({
      good: [1, 2, 10],
      bad: [3],
      click_times: { '1': 100, '2': 200, '10': 300 },
      learned_scores: {},
    });

    expect(service.goodVotes.has(10)).toBeTrue();
    // Server's click time should be used now (not the optimistic one)
    expect(service.clickTimes['10']).toBe(300);
  });

  it('stale polling should not remove optimistic bad vote', fakeAsync(() => {
    service.startPolling(1000);
    // Initial poll
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });

    // User votes bad optimistically
    service.applyOptimisticVote(5, 'bad');
    expect(service.badVotes.has(5)).toBeTrue();

    // Next poll arrives with stale data (no vote for 5)
    tick(1000);
    httpMock.expectOne('/api/votes').flush({ good: [], bad: [], click_times: {}, learned_scores: {} });

    // Optimistic bad vote should still be preserved
    expect(service.badVotes.has(5)).toBeTrue();
    expect(service.clickTimes['5']).toBe(1);

    service.stopPolling();
    discardPeriodicTasks();
  }));

  it('goodVotes$ should emit on load', (done: DoneFn) => {
    const emissions: Set<number>[] = [];
    service.goodVotes$.subscribe((v) => emissions.push(v));

    service.loadVotes();
    httpMock.expectOne('/api/votes').flush(mockVotes);

    setTimeout(() => {
      const last = emissions[emissions.length - 1];
      expect(last.has(1)).toBeTrue();
      expect(last.has(2)).toBeTrue();
      done();
    });
  });
});
