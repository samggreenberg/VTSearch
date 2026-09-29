import { TestBed } from '@angular/core/testing';
import { VoteHistoryService } from './vote-history.service';
import { configureZoneless } from '../testing/zoneless-testbed';

describe('VoteHistoryService', () => {
  let service: VoteHistoryService;

  beforeEach(() => {
    configureZoneless({});
    service = TestBed.inject(VoteHistoryService);
  });

  it('returns null when nothing has been voted on', () => {
    expect(service.stepBack(null)).toBeNull();
  });

  it('steps back to the most recently voted item', () => {
    service.record(1);
    service.record(2);
    // Standing on the item the advance moved to (3, never voted on).
    expect(service.stepBack(3)).toBe(2);
  });

  it('walks further back on repeated presses', () => {
    service.record(1);
    service.record(2);
    service.record(3);
    expect(service.stepBack(99)).toBe(3);
    expect(service.stepBack(3)).toBe(2);
    expect(service.stepBack(2)).toBe(1);
    // Nothing older; the walk stops rather than wrapping.
    expect(service.stepBack(1)).toBeNull();
  });

  it('restarts the walk when the user moved somewhere else in between', () => {
    service.record(1);
    service.record(2);
    service.record(3);
    expect(service.stepBack(99)).toBe(3);
    // The user clicked item 50 in a pile; the old walk is over.
    expect(service.stepBack(50)).toBe(3);
  });

  it('restarts the walk after a fresh vote', () => {
    service.record(1);
    service.record(2);
    expect(service.stepBack(99)).toBe(2);
    expect(service.stepBack(2)).toBe(1);
    service.record(7);
    expect(service.stepBack(99)).toBe(7);
  });

  it('moves a re-voted item to the newest end instead of duplicating it', () => {
    service.record(1);
    service.record(2);
    service.record(1);
    expect([...service.recentlyVoted]).toEqual([2, 1]);
    expect(service.stepBack(99)).toBe(1);
    expect(service.stepBack(1)).toBe(2);
  });

  it('caps the trail, dropping the oldest entries', () => {
    for (let i = 0; i < 60; i++) service.record(i);
    expect(service.recentlyVoted.length).toBe(50);
    expect(service.recentlyVoted[0]).toBe(10);
    expect(service.recentlyVoted[49]).toBe(59);
  });

  it('clears the trail', () => {
    service.record(1);
    service.clear();
    expect([...service.recentlyVoted]).toEqual([]);
    expect(service.stepBack(null)).toBeNull();
  });

  describe('stepForward (#4306)', () => {
    it('returns to where the walk started', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(99)).toBe(2);
      expect(service.stepForward(2)).toBe(99);
    });

    it('returns to the start however far back the walk went', () => {
      service.record(1);
      service.record(2);
      service.record(3);
      expect(service.stepBack(99)).toBe(3);
      expect(service.stepBack(3)).toBe(2);
      expect(service.stepBack(2)).toBe(1);
      expect(service.stepForward(1)).toBe(99);
    });

    it('still returns to the start after running off the oldest end', () => {
      service.record(1);
      expect(service.stepBack(99)).toBe(1);
      expect(service.stepBack(1)).toBeNull();
      expect(service.stepForward(1)).toBe(99);
    });

    it('ends the walk, so a second press takes the host advance', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(99)).toBe(2);
      expect(service.stepForward(2)).toBe(99);
      expect(service.stepForward(99)).toBeNull();
      // And the next ↓ starts a new walk from the newest end.
      expect(service.stepBack(99)).toBe(2);
    });

    it('returns null with no walk in progress', () => {
      expect(service.stepForward(null)).toBeNull();
      service.record(1);
      expect(service.stepForward(99)).toBeNull();
    });

    it('returns null when the user moved somewhere else in between', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(99)).toBe(2);
      // The user clicked item 50 in a pile; the walk is over.
      expect(service.stepForward(50)).toBeNull();
    });

    it('returns null after a vote, which ends the walk', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(99)).toBe(2);
      // Re-voting the item the walk landed on; the vote's own advance moved on.
      service.record(2);
      expect(service.stepForward(2)).toBeNull();
    });

    it('returns null after a clear', () => {
      service.record(1);
      expect(service.stepBack(99)).toBe(1);
      service.clear();
      expect(service.stepForward(1)).toBeNull();
    });

    it('returns null when the walk started with nothing on screen', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(null)).toBe(2);
      expect(service.stepForward(2)).toBeNull();
    });

    it('remembers the start of the newest walk, not an abandoned one', () => {
      service.record(1);
      service.record(2);
      expect(service.stepBack(99)).toBe(2);
      // Clicked item 50, then ↓ again: a new walk, starting from 50.
      expect(service.stepBack(50)).toBe(2);
      expect(service.stepForward(2)).toBe(50);
    });
  });
});
