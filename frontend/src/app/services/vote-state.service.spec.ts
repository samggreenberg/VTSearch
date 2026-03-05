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

  it('goodVotes$ should emit on load', (done) => {
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
