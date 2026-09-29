import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController, type TestRequest } from '@angular/common/http/testing';
import { FloorCheckModalComponent, type FloorCheckVoted } from './floor-check-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { wireFloor } from '../../../testing/line-floor';
import { KeyboardService, type KeyboardAction } from '../../../services/keyboard.service';
import { MediaStateService } from '../../../services/media-state.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import type { Media } from '../../../models/api.models';

/** A running check's wire state, round *round* of *rounds*, dealing *picks* from the top *candidate*. */
function running(picks: number[], round = 1, rounds = 1, candidate = 32, extra: Record<string, unknown> = {}) {
  return {
    status: 'running',
    min_precision: 0.5,
    round,
    rounds,
    picks_per_round: 5,
    candidate,
    start_candidate: 32 * 2 ** (rounds - 1),
    picks,
    labelled: 0,
    right: 0,
    range: null,
    ...extra,
  };
}

/** A finished check's wire state. */
function finished(status: 'confirmed' | 'short', range: { lo: number; hi: number; labelled: number; right: number }) {
  return { ...running([], 1, 1), status, labelled: range.labelled, right: range.right, range };
}

describe('FloorCheckModalComponent (#4273)', () => {
  let fixture: ComponentFixture<FloorCheckModalComponent>;
  let component: FloorCheckModalComponent;
  let httpMock: HttpTestingController;
  let keyboard: KeyboardService;
  let voted: FloorCheckVoted[];
  let closed: number;

  beforeEach(async () => {
    await configureZoneless({
      imports: [FloorCheckModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
    keyboard = TestBed.inject(KeyboardService);
    keyboard.start();
    // Every pick is a loaded image; the metadata batch is not under test.
    vi.spyOn(TestBed.inject(MediaStateService), 'getMedia').mockImplementation(
      (id: number) => ({ id, media_type: 'image', filename: `pick-${id}.png` }) as Media,
    );
    vi.spyOn(TestBed.inject(MediaMetadataCacheService), 'ensureLoaded').mockImplementation(() => undefined);

    fixture = TestBed.createComponent(FloorCheckModalComponent);
    component = fixture.componentInstance;
    voted = [];
    closed = 0;
    component.voted.subscribe((v) => voted.push(v));
    component.closed.subscribe(() => closed++);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    keyboard.stop();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({});
    });
  });

  const el = () => fixture.nativeElement as HTMLElement;
  const text = () => el().textContent!.replace(/\s+/g, ' ');
  const press = async (key: string) => {
    document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
  };
  const dots = () => Array.from(el().querySelectorAll('.pick-dot')) as HTMLButtonElement[];
  const shownId = () => (el().querySelector('vt-image-viewer img') as HTMLImageElement | null)?.getAttribute('src') ?? '';
  const startReq = () => httpMock.expectOne((req) => req.url === '/api/precision-check/start');
  const votesReq = () => httpMock.expectOne((req) => req.url === '/api/precision-check/votes');

  async function start(picks: number[], rounds = 1, candidate = 32): Promise<void> {
    startReq().flush({ floor: wireFloor('unchecked'), check: running(picks, 1, rounds, candidate) });
    await settleZoneless(fixture);
  }

  async function answer(req: TestRequest, body: object): Promise<void> {
    req.flush(body);
    await settleZoneless(fixture);
  }

  describe('a round', () => {
    it('starts a check on open and shows its picks in the order they were drawn', async () => {
      expect(text()).toContain('Drawing picks');
      await start([17, 4, 29, 8, 11]);
      expect(text()).toContain('Spot check: at least 50% right?');
      expect(text()).toContain('5 picks drawn at random from the top 32. Is each one a match?');
      expect(dots().length).toBe(5);
      expect(component.picks()).toEqual([17, 4, 29, 8, 11]);
      expect(component.currentId()).toBe(17);
    });

    it('presents them as a check, not the ranking: no rank numbers and no scores', async () => {
      await start([17, 4, 29, 8, 11]);
      const stage = el().querySelector('.floor-check')!;
      expect(stage.querySelector('.media-score')).toBeNull();
      expect(stage.textContent).not.toMatch(/#\s*\d|rank|score/i);
      dots().forEach((d) => expect(d.textContent!.trim()).toBe(''));
    });

    it('holds the votes until every pick has one, then sends the round whole', async () => {
      await start([17, 4, 29]);
      await press('ArrowRight');
      await press('ArrowLeft');
      httpMock.expectNone((req) => req.url === '/api/precision-check/votes');
      expect(dots().map((d) => d.getAttribute('data-vote'))).toEqual(['good', 'bad', null]);
      await press('ArrowRight');
      expect(votesReq().request.body).toEqual({
        votes: [
          { id: 17, label: 'good' },
          { id: 4, label: 'bad' },
          { id: 29, label: 'good' },
        ],
      });
    });

    it('steps back with ↓ to change a vote before the round is sent', async () => {
      await start([17, 4, 29]);
      await press('ArrowRight');
      expect(component.currentId()).toBe(4);
      await press('ArrowDown');
      expect(component.currentId()).toBe(17);
      await press('ArrowLeft');
      // Back on the next unvoted pick, with 17 now Bad.
      expect(component.currentId()).toBe(4);
      await press('ArrowRight');
      await press('ArrowRight');
      expect(votesReq().request.body.votes).toEqual([
        { id: 17, label: 'bad' },
        { id: 4, label: 'good' },
        { id: 29, label: 'good' },
      ]);
    });

    it('votes with the Good and Bad buttons too, and moves between picks by their dots', async () => {
      await start([17, 4]);
      dots()[1].click();
      await settleZoneless(fixture);
      expect(component.currentId()).toBe(4);
      (el().querySelector('.btn-bad') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      expect(component.currentId()).toBe(17);
      (el().querySelector('.btn-good') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      expect(votesReq().request.body.votes).toEqual([
        { id: 17, label: 'good' },
        { id: 4, label: 'bad' },
      ]);
    });

    it('shows the pick on screen', async () => {
      await start([17, 4]);
      expect(shownId()).toContain('/api/medias/17/');
      await press('ArrowRight');
      expect(shownId()).toContain('/api/medias/4/');
    });
  });

  describe('the vote keys', () => {
    it('reach the step and never the ranked list behind it', async () => {
      const actions: KeyboardAction[] = [];
      keyboard.action$.subscribe((a) => actions.push(a));
      await start([17, 4]);
      await press('ArrowRight');
      await press('ArrowDown');
      await press('ArrowLeft');
      expect(actions).toEqual([]);
      expect(component.votes().get(17)).toBe('bad');
    });

    it('go back to the list once the step closes', async () => {
      const actions: KeyboardAction[] = [];
      keyboard.action$.subscribe((a) => actions.push(a));
      await start([17]);
      const host = el();
      fixture.destroy();
      // The fixture's host outlives the component in jsdom; the app's `@if` removes it.
      host.remove();
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      expect(actions.map((a) => a.type)).toEqual(['vote']);
    });
  });

  describe('a check of several rounds', () => {
    it('says a short round is not there yet, checks a shorter list, and ends on the result', async () => {
      // 10%: the top 128, up to three rounds of 5.
      startReq().flush({
        floor: wireFloor('unchecked', { minPrecision: 0.1, count: 128 }),
        check: { ...running([90, 3, 51, 7, 64], 1, 3, 128), min_precision: 0.1 },
      });
      await settleZoneless(fixture);
      expect(text()).toContain('Round 1 of up to 3: 5 picks drawn at random from the top 128.');
      expect(el().querySelector('.check-shorter')).toBeNull();
      for (let i = 0; i < 5; i++) await press('ArrowLeft');

      await answer(votesReq(), {
        floor: wireFloor('unchecked', { minPrecision: 0.1, count: 128 }),
        check: { ...running([12, 40, 5, 33, 21], 2, 3, 64), min_precision: 0.1, labelled: 2, right: 0 },
      });
      expect(voted.map((v) => v.finished)).toEqual([false]);
      expect(el().querySelector('.check-shorter')!.textContent).toContain('Not there yet: checking a shorter list');
      expect(text()).toContain('Round 2 of up to 3: 5 picks drawn at random from the top 64.');
      expect(component.currentId()).toBe(12);
      expect(dots().every((d) => d.getAttribute('data-vote') === null)).toBe(true);
      for (let i = 0; i < 5; i++) await press('ArrowLeft');

      await answer(votesReq(), {
        floor: wireFloor('unchecked', { minPrecision: 0.1, count: 128 }),
        check: { ...running([8, 19, 2, 30, 14], 3, 3, 32), min_precision: 0.1, labelled: 3, right: 0 },
      });
      expect(text()).toContain('Round 3 of up to 3: 5 picks drawn at random from the top 32.');
      for (let i = 0; i < 5; i++) await press('ArrowRight');

      const range = { lo: 0.04, hi: 0.67, labelled: 8, right: 5 };
      await answer(votesReq(), {
        floor: { ...wireFloor('short', { minPrecision: 0.1, count: 32 }), range },
        check: { ...finished('short', range), min_precision: 0.1, round: 3, rounds: 3 },
      });
      expect(voted.map((v) => v.finished)).toEqual([false, false, true]);
      expect(el().querySelector('.check-result-headline')!.textContent).toContain(
        'Aimed at 10%: likely 4–67% right (checked 8).',
      );
      expect(text()).toContain('The line keeps the top 32 the check ended on.');
      // A short check names no cause: the copy is true of a sparse corpus and a weak model alike.
      expect(text()).not.toMatch(/sparse|weak|too few|model/i);
      expect(dots().length).toBe(0);
    });

    it('ends on a confirmed floor with the range and the count the check returned', async () => {
      await start([17, 4, 29, 8, 11], 2, 64);
      for (let i = 0; i < 5; i++) await press('ArrowRight');
      const range = { lo: 0.26, hi: 1, labelled: 5, right: 5 };
      await answer(votesReq(), {
        floor: { ...wireFloor('confirmed', { minPrecision: 0.25, count: 64 }), range },
        check: { ...finished('confirmed', range), min_precision: 0.25, rounds: 2, candidate: 64 },
      });
      expect(el().querySelector('.check-result')!.getAttribute('data-status')).toBe('confirmed');
      expect(el().querySelector('.check-result-headline')!.textContent).toContain(
        'At least 25% right: likely 26–100% (checked 5).',
      );
      expect(text()).toContain('The line keeps these 64.');
      expect(voted.at(-1)!.finished).toBe(true);
    });
  });

  describe('closing', () => {
    it('cancels a running check, leaving the state as it was', async () => {
      await start([17, 4]);
      await press('ArrowRight');
      (el().querySelector('.modal-footer .btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock.expectOne((req) => req.url === '/api/precision-check/cancel');
      httpMock.expectNone((req) => req.url === '/api/precision-check/votes');
      expect(closed).toBe(1);
      expect(voted).toEqual([]);
    });

    it('closes a finished check without cancelling anything', async () => {
      await start([17]);
      await press('ArrowRight');
      const range = { lo: 0.55, hi: 1, labelled: 1, right: 1 };
      await answer(votesReq(), { floor: { ...wireFloor('confirmed'), range }, check: finished('confirmed', range) });
      const done = el().querySelector('.modal-footer .btn') as HTMLButtonElement;
      expect(done.textContent!.trim()).toBe('Done');
      done.click();
      await settleZoneless(fixture);
      httpMock.expectNone((req) => req.url === '/api/precision-check/cancel');
      expect(closed).toBe(1);
    });
  });

  describe('when the server says no', () => {
    it('shows why a check could not start', async () => {
      startReq().flush(
        { code: 409, message: 'This candidate was already checked; vote on something first, or re-sort.' },
        { status: 409, statusText: 'Conflict' },
      );
      await settleZoneless(fixture);
      expect(el().querySelector('.error-text')!.textContent).toContain('already checked');
      expect(el().querySelector('.modal-footer .btn')!.textContent!.trim()).toBe('Done');
    });

    it('keeps a round whose votes failed to send, to send again', async () => {
      await start([17]);
      await press('ArrowRight');
      votesReq().flush({ message: 'Network trouble' }, { status: 500, statusText: 'Server Error' });
      await settleZoneless(fixture);
      expect(el().querySelector('.error-text')!.textContent).toContain('Network trouble');
      expect(component.votes().get(17)).toBe('good');
      (el().querySelector('.error-text .btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      expect(votesReq().request.body.votes).toEqual([{ id: 17, label: 'good' }]);
    });
  });
});
