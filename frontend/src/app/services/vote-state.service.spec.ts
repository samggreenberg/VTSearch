import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { VoteStateService } from './vote-state.service';
import { provideHttpTesting } from '../testing/test-providers';

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
      providers: [...provideHttpTesting()],
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

    expect(service.goodVotes.has(1)).toBe(true);
    expect(service.goodVotes.has(2)).toBe(true);
    expect(service.badVotes.has(3)).toBe(true);
    expect(service.clickTimes).toEqual({ '1': 100, '2': 200 });
    expect(service.learnedScores).toEqual({ '1': 0.9 });
  });

  it('startPolling should periodically fetch votes', async () => {
    vi.useFakeTimers();
    try {
      service.startPolling(1000);

      // First poll at t=0 (timer(0, …) emits on a macrotask; flush it).
      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/votes').flush(mockVotes);
      expect(service.goodVotes.size).toBe(2);

      // Second poll at t=1000
      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectOne('/api/votes').flush({ good: [1], bad: [], click_times: {}, learned_scores: {} });
      expect(service.goodVotes.size).toBe(1);

      service.stopPolling();
    } finally {
      vi.useRealTimers();
    }
  });

  it('stopPolling should stop periodic fetches', async () => {
    vi.useFakeTimers();
    try {
      service.startPolling(1000);
      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/votes').flush(mockVotes);

      service.stopPolling();
      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectNone('/api/votes');
    } finally {
      vi.useRealTimers();
    }
  });

  // Regression for logical-bug-audit H24: a single /api/votes failure
  // (502 from a proxy, stale X-Dataset-Id after a context switch, etc.)
  // used to terminate the entire polling chain and leave `polling` stuck
  // at true, freezing vote state for the rest of the session.
  it('polling survives a transient /api/votes error and keeps polling', async () => {
    vi.useFakeTimers();
    try {
      service.startPolling(1000);

      // First tick: server is unreachable. Pre-fix this would tear the
      // whole observable down.
      await vi.advanceTimersByTimeAsync(0);
      httpMock.expectOne('/api/votes').flush(null, { status: 502, statusText: 'Bad Gateway' });

      // Local state must NOT be clobbered to empty on a failed tick.
      // (Without the EMPTY shortcut, an empty stub VotesResponse here
      // would silently erase optimistic votes.)
      service.applyOptimisticState(99, 'good');
      expect(service.goodVotes.has(99)).toBe(true);

      // Second tick: server is back. The chain must still be alive.
      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectOne('/api/votes').flush(mockVotes);
      expect(service.goodVotes.has(1)).toBe(true);
      expect(service.goodVotes.has(2)).toBe(true);

      service.stopPolling();
    } finally {
      vi.useRealTimers();
    }
  });

  it('clear should reset all state', () => {
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush(mockVotes);

    service.clear();
    expect(service.goodVotes.size).toBe(0);
    expect(service.badVotes.size).toBe(0);
    expect(service.clickTimes).toEqual({});
    expect(service.learnedScores).toEqual({});
  });

  describe('toggleTargetFor (local toggle rule → absolute target)', () => {
    it('returns clicked direction when media is unvoted', () => {
      expect(service.toggleTargetFor(5, 'good')).toBe('good');
      expect(service.toggleTargetFor(5, 'bad')).toBe('bad');
    });

    it('returns none when clicked direction matches current polarity', () => {
      service.applyOptimisticState(5, 'good');
      expect(service.toggleTargetFor(5, 'good')).toBe('none');
      service.applyOptimisticState(5, 'bad');
      expect(service.toggleTargetFor(5, 'bad')).toBe('none');
    });

    it('returns clicked direction when polarities differ (flip)', () => {
      service.applyOptimisticState(5, 'good');
      expect(service.toggleTargetFor(5, 'bad')).toBe('bad');
      service.applyOptimisticState(5, 'bad');
      expect(service.toggleTargetFor(5, 'good')).toBe('good');
    });
  });

  describe('Find-mode verification gating', () => {
    /** Push a /api/votes payload (with a verified array) through applyVotes. */
    const flushVotes = (good: number[], bad: number[], verified: number[]) => {
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good, bad, verified, click_times: {}, learned_scores: {} });
    };

    it('treats a flood-filled but unverified item as having no vote', () => {
      // Detector presumption: item 5 is good, but the human has not verified.
      flushVotes([5], [], []);
      service.setFindMode(true);

      // Buttons read neutral...
      expect(service.effectiveGood(5)).toBe(false);
      expect(service.effectiveBad(5)).toBe(false);
      // ...and clicking Good verifies (sets absolute good) rather than toggling off.
      expect(service.toggleTargetFor(5, 'good')).toBe('good');
      // Clicking Bad flips the presumption to a verified bad (cull).
      expect(service.toggleTargetFor(5, 'bad')).toBe('bad');
    });

    it('honours the real vote once an item is verified', () => {
      flushVotes([5], [], [5]);
      service.setFindMode(true);

      expect(service.effectiveGood(5)).toBe(true);
      expect(service.effectiveBad(5)).toBe(false);
      // A verified-good item toggles off when Good is clicked again.
      expect(service.toggleTargetFor(5, 'good')).toBe('none');
      // ...and flips to bad when Bad is clicked.
      expect(service.toggleTargetFor(5, 'bad')).toBe('bad');
    });

    it('outside Find mode, membership alone decides state', () => {
      flushVotes([5], [], []);
      // findMode left false (default).
      expect(service.effectiveGood(5)).toBe(true);
      expect(service.toggleTargetFor(5, 'good')).toBe('none');
    });
  });

  describe('applyOptimisticState (absolute target)', () => {
    it('sets good and assigns a click time', () => {
      service.applyOptimisticState(5, 'good');
      expect(service.goodVotes.has(5)).toBe(true);
      expect(service.badVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBe(1);
    });

    it('sets bad and assigns a click time', () => {
      service.applyOptimisticState(5, 'bad');
      expect(service.badVotes.has(5)).toBe(true);
      expect(service.goodVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBe(1);
    });

    it('target=none clears the vote and click time', () => {
      service.applyOptimisticState(5, 'good');
      service.applyOptimisticState(5, 'none');
      expect(service.goodVotes.has(5)).toBe(false);
      expect(service.badVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBeUndefined();
    });

    it('moves bad → good with a new click time', () => {
      service.applyOptimisticState(5, 'bad');
      expect(service.clickTimes['5']).toBe(1);

      service.applyOptimisticState(5, 'good');
      expect(service.goodVotes.has(5)).toBe(true);
      expect(service.badVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBe(2);
    });

    it('click time exceeds existing max', () => {
      service.loadVotes();
      httpMock.expectOne('/api/votes').flush(mockVotes);
      // mockVotes has click_times: { '1': 100, '2': 200 }

      service.applyOptimisticState(7, 'good');
      expect(service.clickTimes['7']).toBe(201);
    });
  });

  describe('submitToggleVote (the H1 fix surface)', () => {
    it('posts absolute target=good when clicking good on an unvoted media', () => {
      service.submitToggleVote(5, 'good').subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'good' });
      req.flush({ ok: true, state: 'good', click_time: 1 });
      expect(service.goodVotes.has(5)).toBe(true);
    });

    it('posts absolute target=none when clicking good on an already-good media (toggle off)', () => {
      service.applyOptimisticState(5, 'good');
      service.submitToggleVote(5, 'good').subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'none' });
      req.flush({ ok: true, state: 'none', click_time: null });
      expect(service.goodVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBeUndefined();
    });

    it('posts absolute target=bad when flipping from good (polarity switch)', () => {
      service.applyOptimisticState(5, 'good');
      service.submitToggleVote(5, 'bad').subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      req.flush({ ok: true, state: 'bad', click_time: 2 });
      expect(service.badVotes.has(5)).toBe(true);
      expect(service.goodVotes.has(5)).toBe(false);
    });

    it('honours region_box only when the computed target is good', () => {
      service.submitToggleVote(5, 'good', [0.1, 0.2, 0.3, 0.4]).subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.3, 0.4] });
      req.flush({ ok: true, state: 'good', click_time: 1 });

      // Click good again → target=none, region_box must be omitted.
      service.submitToggleVote(5, 'good', [0.1, 0.2, 0.3, 0.4]).subscribe();
      const req2 = httpMock.expectOne('/api/medias/5/vote');
      expect(req2.request.body).toEqual({ target: 'none' });
      req2.flush({ ok: true, state: 'none', click_time: null });
    });

    it('reconciles local state from the server response, even when the prediction was wrong', () => {
      service.submitToggleVote(5, 'good').subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      // Server says the state is actually "none" (e.g. another tab raced
      // ahead of us); the optimistic 'good' must be overridden.
      req.flush({ ok: true, state: 'none', click_time: null });

      expect(service.goodVotes.has(5)).toBe(false);
      expect(service.badVotes.has(5)).toBe(false);
      expect(service.clickTimes['5']).toBeUndefined();
    });
  });

  describe('polling interaction with optimistic state', () => {
    it('preserves the optimistic state when a poll lands before the POST returns', () => {
      service.submitToggleVote(10, 'good').subscribe();
      // POST is in-flight; a poll arrives WITHOUT the new vote (stale data).
      service.loadVotes();
      httpMock.expectOne('/api/votes').flush({
        good: [1, 2],
        bad: [3],
        click_times: { '1': 100, '2': 200 },
        learned_scores: {},
      });

      // Optimistic vote should still be visible.
      expect(service.goodVotes.has(10)).toBe(true);
      expect(service.goodVotes.has(1)).toBe(true);
      expect(service.goodVotes.has(2)).toBe(true);

      // Drain the POST so the test doesn't leave a pending request.
      httpMock.expectOne('/api/medias/10/vote').flush({ ok: true, state: 'good', click_time: 1 });
    });

    it('clears pendingOptimistic deterministically when the POST resolves', () => {
      service.submitToggleVote(10, 'good').subscribe();
      httpMock
        .expectOne('/api/medias/10/vote')
        .flush({ ok: true, state: 'good', click_time: 300 });

      // A subsequent stale poll (server reflects pre-vote state) must NOT
      // re-introduce the optimistic ghost; pendingOptimistic was cleared
      // by reconcileVoteResponse, so the poll's absence of the vote wins.
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {} });

      expect(service.goodVotes.has(10)).toBe(false);
    });

    it('does not get stuck in a predict-vs-server desync (H1 stuck-prediction scenario)', () => {
      // Another tab voted X=good first; our tab still thinks X is unvoted.
      // We click good → optimistic predicts good. Server (already good)
      // returns state=good idempotently. Counters do not inflate; local
      // state lines up with the server.
      service.submitToggleVote(5, 'good').subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'good' });
      req.flush({ ok: true, state: 'good', click_time: 42 });

      expect(service.goodVotes.has(5)).toBe(true);
      expect(service.clickTimes['5']).toBe(42);
    });

    it('drops a stale /api/votes response that resolves after a fresher one (out-of-order race)', () => {
      service.setFindMode(true);
      // Flood-fill: 1, 2, 3 presumed good, none verified yet.
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [1, 2, 3], bad: [], click_times: {}, learned_scores: {}, verified: [] });

      // A read issued early (e.g. a poll that read the server before the vote
      // commits) is still in flight; it will report the OLD verified set.
      service.loadVotes(); // GET_stale

      // The user verifies id 2; onMediaVoted optimistically marks it and fires
      // its own post-vote read.
      service.setOptimisticVerified(2, true);
      service.loadVotes(); // GET_fresh

      const [staleReq, freshReq] = httpMock.match('/api/votes');
      // The fresher read lands first: the server now reports 2 as verified.
      freshReq.flush({ good: [1, 2, 3], bad: [], click_times: {}, learned_scores: {}, verified: [1, 2] });
      expect(service.verifiedIds.has(2)).toBe(true);

      // The stale read lands later with the pre-vote verified set. It must be
      // dropped, not applied — otherwise it wipes the just-verified id back out
      // of the right pile (the find-verification staleness report).
      staleReq.flush({ good: [1, 2, 3], bad: [], click_times: {}, learned_scores: {}, verified: [1] });
      expect(service.verifiedIds.has(2)).toBe(true);
    });
  });

  describe('failed vote POST rollback', () => {
    it('rolls back the optimistic vote when the POST fails, so polls show server truth', () => {
      service.submitToggleVote(10, 'good').subscribe({ next: () => {}, error: () => {} });
      expect(service.goodVotes.has(10)).toBe(true); // optimistic

      // The POST fails (transient 502 below the offline-breaker threshold).
      httpMock
        .expectOne('/api/medias/10/vote')
        .flush(null, { status: 502, statusText: 'Bad Gateway' });

      // The rollback re-reads server state; the server never saw the vote.
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
      expect(service.goodVotes.has(10)).toBe(false);

      // Regression: the pending entry used to survive the failed POST and
      // re-impose the phantom vote over EVERY subsequent poll.
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
      expect(service.goodVotes.has(10)).toBe(false);
      expect(service.clickTimes['10']).toBeUndefined();
    });

    it('rolls back an optimistic region box when the POST fails', () => {
      service
        .submitToggleVote(11, 'good', [0.1, 0.2, 0.3, 0.4])
        .subscribe({ next: () => {}, error: () => {} });
      expect(service.goodRegionBoxes['11']).toEqual([0.1, 0.2, 0.3, 0.4]);

      httpMock
        .expectOne('/api/medias/11/vote')
        .flush(null, { status: 500, statusText: 'Server Error' });
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {}, good_region_boxes: {} });

      expect(service.goodRegionBoxes['11']).toBeUndefined();

      // A later poll must not resurrect the crop either.
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {}, good_region_boxes: {} });
      expect(service.goodRegionBoxes['11']).toBeUndefined();
    });

    it('a failed undo POST repairs local state from the server', () => {
      service.recordVote(5, 'good', 'foo.wav');
      service.applyOptimisticState(5, 'good');

      service.undo(); // posts target=none optimistically
      expect(service.goodVotes.has(5)).toBe(false);
      httpMock
        .expectOne('/api/medias/5/vote')
        .flush(null, { status: 500, statusText: 'Server Error' });

      // The rollback's loadVotes reflects the server (vote still good).
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [5], bad: [], click_times: { '5': 3 }, learned_scores: {} });
      expect(service.goodVotes.has(5)).toBe(true);

      // Regression: the pendingOptimistic entry from the undo's optimistic
      // apply used to override this reload on every later poll.
      service.loadVotes();
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [5], bad: [], click_times: { '5': 3 }, learned_scores: {} });
      expect(service.goodVotes.has(5)).toBe(true);
    });
  });

  describe('undo / redo stack', () => {
    it('records the previous polarity at click time and POSTs the inverse target on undo', () => {
      // Pretend a vote landed: media 5 is now good.
      service.applyOptimisticState(5, 'good');
      // Capture state BEFORE the next click (which will toggle off).
      service.recordVote(5, 'good', 'foo.wav');
      // Simulate the click that follows: toggle off → state=none.
      service.applyOptimisticState(5, 'none');

      service.undo();
      // The inverse target is the saved previousPolarity ('good').
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'good' });
      req.flush({ ok: true, state: 'good', click_time: 1 });

      expect(service.canRedo()).toBe(true);
      expect(service.canUndo()).toBe(false);
    });

    it('undo of a first-time vote posts target=none', () => {
      service.recordVote(7, 'bad', 'bar.wav'); // previousPolarity = null
      service.applyOptimisticState(7, 'bad');

      service.undo();
      const req = httpMock.expectOne('/api/medias/7/vote');
      expect(req.request.body).toEqual({ target: 'none' });
      req.flush({ ok: true, state: 'none', click_time: null });
    });

    it('redo replays the original clicked direction as an absolute target', () => {
      service.recordVote(7, 'bad', 'bar.wav'); // previousPolarity = null
      service.applyOptimisticState(7, 'bad');

      service.undo();
      httpMock
        .expectOne('/api/medias/7/vote')
        .flush({ ok: true, state: 'none', click_time: null });

      service.redo();
      const req = httpMock.expectOne('/api/medias/7/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      req.flush({ ok: true, state: 'bad', click_time: 2 });

      expect(service.canRedo()).toBe(false);
      expect(service.canUndo()).toBe(true);
    });

    it('a new recordVote clears the redo stack', () => {
      service.recordVote(1, 'good', 'a');
      service.applyOptimisticState(1, 'good');
      service.undo();
      httpMock
        .expectOne('/api/medias/1/vote')
        .flush({ ok: true, state: 'none', click_time: null });
      expect(service.canRedo()).toBe(true);

      service.recordVote(2, 'bad', 'b');
      expect(service.canRedo()).toBe(false);
    });

    it('undo with empty stack is a no-op', () => {
      service.undo();
      httpMock.expectNone('/api/medias/0/vote');
      expect(service.canUndo()).toBe(false);
    });

    it('emits a toast on undo and redo', () => {
      const toasts: { action: string; mediaName: string }[] = [];
      service.toast$.subscribe((t) => toasts.push(t));

      service.recordVote(9, 'good', 'pic.png');
      service.applyOptimisticState(9, 'good');
      service.undo();
      httpMock
        .expectOne('/api/medias/9/vote')
        .flush({ ok: true, state: 'none', click_time: null });

      service.redo();
      httpMock
        .expectOne('/api/medias/9/vote')
        .flush({ ok: true, state: 'good', click_time: 2 });

      expect(toasts).toEqual([
        { action: 'undo', mediaName: 'pic.png' },
        { action: 'redo', mediaName: 'pic.png' },
      ]);
    });

    it('caps the past stack at 20 entries', () => {
      for (let i = 0; i < 25; i++) {
        service.recordVote(i, 'good', `m${i}`);
      }
      let popped = 0;
      while (service.canUndo()) {
        service.undo();
        const req = httpMock.expectOne((r) => r.url.startsWith('/api/medias/') && r.url.endsWith('/vote'));
        req.flush({ ok: true, state: 'none', click_time: null });
        popped++;
        if (popped > 30) throw new Error('runaway');
      }
      expect(popped).toBe(20);
    });

    it('clear() wipes the undo/redo stacks', () => {
      service.recordVote(1, 'good', 'a');
      expect(service.canUndo()).toBe(true);
      service.clear();
      expect(service.canUndo()).toBe(false);
      expect(service.canRedo()).toBe(false);
    });
  });

  describe('submitToggleVoteAndRecord (the H26 fix surface)', () => {
    it('records the undo entry only after the POST succeeds', () => {
      service.submitToggleVoteAndRecord(5, 'good', 'foo.wav').subscribe();
      // Undo stack must be empty while the POST is in flight.
      expect(service.canUndo()).toBe(false);

      const req = httpMock.expectOne('/api/medias/5/vote');
      req.flush({ ok: true, state: 'good', click_time: 1 });

      // After confirmation, the entry should be on the stack.
      expect(service.canUndo()).toBe(true);
    });

    it('does NOT record an undo entry when the POST errors', () => {
      service.submitToggleVoteAndRecord(5, 'good', 'foo.wav').subscribe({
        next: () => {},
        error: () => {},
      });
      const req = httpMock.expectOne('/api/medias/5/vote');
      req.flush(null, { status: 500, statusText: 'Server Error' });
      // The failed POST triggers the optimistic rollback's server re-read.
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {} });

      // The failed POST must not leave a phantom undo entry; Cmd-Z next
      // would otherwise post a "reversal" of a vote that never happened.
      expect(service.canUndo()).toBe(false);
    });

    it('captures previousPolarity before the optimistic flip', () => {
      // Media starts as "good"; user clicks bad (a flip).
      service.applyOptimisticState(5, 'good');

      service.submitToggleVoteAndRecord(5, 'bad', 'foo.wav').subscribe();
      // submitToggleVote inside has already optimistically flipped 5 to bad.
      expect(service.badVotes.has(5)).toBe(true);

      const req = httpMock.expectOne('/api/medias/5/vote');
      req.flush({ ok: true, state: 'bad', click_time: 2 });

      // Undo should now POST target='good' (the previousPolarity before
      // the click, NOT the polarity that the optimistic flip put us in).
      service.undo();
      const undoReq = httpMock.expectOne('/api/medias/5/vote');
      expect(undoReq.request.body).toEqual({ target: 'good' });
      undoReq.flush({ ok: true, state: 'good', click_time: 3 });
    });

    it('clears the redo stack only when the POST succeeds', () => {
      // Set up a redo entry the normal way.
      service.recordVote(1, 'good', 'a');
      service.applyOptimisticState(1, 'good');
      service.undo();
      httpMock
        .expectOne('/api/medias/1/vote')
        .flush({ ok: true, state: 'none', click_time: null });
      expect(service.canRedo()).toBe(true);

      // A new vote whose POST fails must NOT wipe the redo stack; the
      // vote never happened, so the redo entry is still legitimate.
      service.submitToggleVoteAndRecord(2, 'bad', 'b').subscribe({
        next: () => {},
        error: () => {},
      });
      httpMock
        .expectOne('/api/medias/2/vote')
        .flush(null, { status: 500, statusText: 'Server Error' });
      // Drain the rollback's server re-read.
      httpMock
        .expectOne('/api/votes')
        .flush({ good: [], bad: [], click_times: {}, learned_scores: {} });
      expect(service.canRedo()).toBe(true);

      // A new vote whose POST succeeds wipes the redo stack as expected.
      service.submitToggleVoteAndRecord(3, 'good', 'c').subscribe();
      httpMock
        .expectOne('/api/medias/3/vote')
        .flush({ ok: true, state: 'good', click_time: 5 });
      expect(service.canRedo()).toBe(false);
    });

    it('honours regionBox on a good vote', () => {
      service
        .submitToggleVoteAndRecord(5, 'good', 'foo.wav', [0.1, 0.2, 0.3, 0.4])
        .subscribe();
      const req = httpMock.expectOne('/api/medias/5/vote');
      expect(req.request.body).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.3, 0.4] });
      req.flush({ ok: true, state: 'good', click_time: 1 });
    });

    it('redo of a region good vote re-applies the click crop', () => {
      service
        .submitToggleVoteAndRecord(5, 'good', 'foo.wav', [0.1, 0.2, 0.3, 0.4])
        .subscribe();
      httpMock
        .expectOne('/api/medias/5/vote')
        .flush({ ok: true, state: 'good', click_time: 1 });

      service.undo();
      httpMock
        .expectOne('/api/medias/5/vote')
        .flush({ ok: true, state: 'none', click_time: null });

      // Redo must restore the original crop, not POST a box-less good vote.
      service.redo();
      const redoReq = httpMock.expectOne('/api/medias/5/vote');
      expect(redoReq.request.body).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.3, 0.4] });
      redoReq.flush({ ok: true, state: 'good', click_time: 2 });
      expect(service.goodRegionBoxes['5']).toEqual([0.1, 0.2, 0.3, 0.4]);
    });

    it('undo restores the crop of the previous good vote', () => {
      // Media 5 already has a region good vote with a known crop.
      service.applyOptimisticState(5, 'good');
      service['_goodRegionBoxes'].set({ '5': [0.5, 0.6, 0.7, 0.8] });

      // User re-crops the same good vote with a new box.
      service
        .submitToggleVoteAndRecord(5, 'good', 'foo.wav', [0.1, 0.2, 0.3, 0.4])
        .subscribe();
      httpMock
        .expectOne('/api/medias/5/vote')
        .flush({ ok: true, state: 'good', click_time: 1 });

      // Undo must restore the prior crop, not drop it.
      service.undo();
      const undoReq = httpMock.expectOne('/api/medias/5/vote');
      expect(undoReq.request.body).toEqual({ target: 'good', region_box: [0.5, 0.6, 0.7, 0.8] });
      undoReq.flush({ ok: true, state: 'good', click_time: 2 });
      expect(service.goodRegionBoxes['5']).toEqual([0.5, 0.6, 0.7, 0.8]);
    });
  });

  it('goodVotes getter reflects a loaded /api/votes response', () => {
    service.loadVotes();
    httpMock.expectOne('/api/votes').flush(mockVotes);

    expect(service.goodVotes.has(1)).toBe(true);
    expect(service.goodVotes.has(2)).toBe(true);
  });
});
