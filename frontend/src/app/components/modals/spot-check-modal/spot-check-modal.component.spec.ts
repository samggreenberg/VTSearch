import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController, type TestRequest } from '@angular/common/http/testing';
import { SpotCheckModalComponent, type SpotCheckVoted } from './spot-check-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { wireBalance } from '../../../testing/line-balance';
import { KeyboardService, type KeyboardAction } from '../../../services/keyboard.service';
import { MediaStateService } from '../../../services/media-state.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import type { Media } from '../../../models/api.models';

/**
 * A running walk's wire state (#4388, #4413): round *round* of a ranking of
 * *rounds* bands, dealing *picks* from band *band* (its 1-based rank positions
 * *lo*–*hi*) while the set under test is the top *candidate*.
 */
function running(
  picks: number[],
  round = 1,
  rounds = 6,
  candidate = 32,
  extra: Record<string, unknown> = {},
  band: { index: number; lo: number; hi: number } | null = { index: 0, lo: 1, hi: 8 },
) {
  return {
    status: 'running',
    beta: 1,
    round,
    rounds,
    picks_per_round: 5,
    candidate,
    start_candidate: 32,
    bands: Math.max(1, Math.ceil(Math.log2(Math.max(candidate, 8) / 8)) + 1),
    band,
    direction: 'start',
    estimate: null,
    fbeta: null,
    recall: null,
    picks,
    labelled: 0,
    right: 0,
    range: null,
    ...extra,
  };
}

/** A finished check's wire state: the walk ended on the peak. */
function finished(range: { lo: number; hi: number; labelled: number; right: number }, recall: { lo: number; hi: number }) {
  return {
    ...running([], 1, 6, 32, {}, null),
    status: 'checked',
    labelled: range.labelled,
    right: range.right,
    range,
    recall: { ...recall, labelled: range.labelled, right: range.right },
    estimate: range.labelled ? range.right / range.labelled : null,
    fbeta: 0.7,
  };
}

/** The wire `balance` of a checked line, with its two ranges. */
function checkedBalance(count: number, range: { lo: number; hi: number; labelled: number; right: number }, recall: { lo: number; hi: number }) {
  return wireBalance('checked', {
    count,
    precision: { ...range, stale: false },
    recall: { ...recall, labelled: range.labelled, right: range.right, stale: false },
    fbeta: 0.7,
  });
}

describe('SpotCheckModalComponent (#4413, #4273)', () => {
  let fixture: ComponentFixture<SpotCheckModalComponent>;
  let component: SpotCheckModalComponent;
  let httpMock: HttpTestingController;
  let keyboard: KeyboardService;
  let voted: SpotCheckVoted[];
  let closed: number;

  beforeEach(async () => {
    await configureZoneless({
      imports: [SpotCheckModalComponent],
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

    fixture = TestBed.createComponent(SpotCheckModalComponent);
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
    startReq().flush({ balance: wireBalance('unchecked'), check: running(picks, 1, rounds, candidate) });
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
      expect(text()).toContain('Spot check');
      // The balance goes unnamed (#4317): the radio on the Threshold spectrum shows it.
      expect(text()).not.toMatch(/Centered|Complete|Correct|aiming|beta|F1/);
      expect(text()).toContain('5 picks drawn at random from the top 8, checking the top 32. Is each one a match?');
      expect(dots().length).toBe(5);
      expect(component.picks()).toEqual([17, 4, 29, 8, 11]);
      expect(component.currentId()).toBe(17);
    });

    it('presents them as a check, not the ranking: no rank numbers and no scores', async () => {
      await start([17, 4, 29, 8, 11]);
      const stage = el().querySelector('.spot-check')!;
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

  describe('a walk of several bands (#4388, #4413)', () => {
    it('audits the starting bands in turn, says when it walks deeper or back past the peak, and ends on the peak', async () => {
      // Balanced: the top 32 in three bands (8, 8, 16), then the walk.
      startReq().flush({
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([90, 3, 51, 7, 64], 1, 6, 32, {}, { index: 0, lo: 1, hi: 8 }),
      });
      await settleZoneless(fixture);
      expect(text()).toContain('5 picks drawn at random from the top 8, checking the top 32.');
      expect(el().querySelector('.check-walk')).toBeNull();
      for (let i = 0; i < 5; i++) await press('ArrowRight');

      // The next band the starting set owes: the walk has not moved, so no note.
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([12, 40, 5, 33, 21], 2, 6, 32, { labelled: 5, right: 5 }, { index: 1, lo: 9, hi: 16 }),
      });
      expect(voted.map((v) => v.finished)).toEqual([false]);
      expect(el().querySelector('.check-walk')).toBeNull();
      expect(text()).toContain('5 picks drawn at random from items 9–16 of the list, checking the top 32.');
      expect(component.currentId()).toBe(12);
      expect(dots().every((d) => d.getAttribute('data-vote') === null)).toBe(true);
      for (let i = 0; i < 5; i++) await press('ArrowRight');
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([8, 19, 2, 30, 14], 3, 6, 32, { labelled: 10, right: 10 }, { index: 2, lo: 17, hi: 32 }),
      });
      for (let i = 0; i < 5; i++) await press('ArrowRight');

      // The estimate did not fall: the walk goes deeper, into the next 32.
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running(
          [41, 60, 35, 52, 48],
          4,
          6,
          64,
          { labelled: 15, right: 15, direction: 'deeper', estimate: 1, fbeta: 0.6, bands: 4 },
          { index: 3, lo: 33, hi: 64 },
        ),
      });
      expect(el().querySelector('.check-walk')!.textContent).toContain('Better so far: checking the next 32.');
      expect(text()).toContain('5 picks drawn at random from items 33–64 of the list, checking the top 64.');
      for (let i = 0; i < 5; i++) await press('ArrowLeft');

      // The top 64 fell: the walk steps back to the top 32, the peak, and ends there.
      const range = { lo: 0.55, hi: 1, labelled: 15, right: 15 };
      const recall = { lo: 0.3, hi: 0.7 };
      await answer(votesReq(), {
        balance: checkedBalance(32, range, recall),
        check: { ...finished(range, recall), round: 4, candidate: 32, direction: 'shallower', bands: 3 },
      });
      expect(voted.map((v) => v.finished)).toEqual([false, false, false, true]);
      expect(el().querySelector('.check-result')!.getAttribute('data-status')).toBe('checked');
      // Beta 1 is an advisory check (#4427): the audited set is named, and the line keeps its own count.
      expect(el().querySelector('.check-result-headline')!.textContent).toContain(
        'Checked the top 32: likely 55–100% right, about half of them found (checked 15).',
      );
      expect(text()).toContain(
        "The line keeps its 32, where your labels put it: the check informs the line and does not move it. " +
          'The ranges are what the picks say about the top 32.',
      );
      expect(dots().length).toBe(0);
    });

    it('says the walk is past the peak when it turns back, and ends checked on a short set too', async () => {
      await start([90, 3, 51, 7, 64], 6, 32);
      for (let i = 0; i < 5; i++) await press('ArrowLeft');
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([12, 40, 5, 33, 21], 2, 6, 32, { labelled: 5, right: 0 }, { index: 1, lo: 9, hi: 16 }),
      });
      for (let i = 0; i < 5; i++) await press('ArrowLeft');
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([8, 19, 2, 30, 14], 3, 6, 32, { labelled: 10, right: 0 }, { index: 2, lo: 17, hi: 32 }),
      });
      for (let i = 0; i < 5; i++) await press('ArrowLeft');
      // The first deeper step fell: the walk turns back to a shorter list.
      await answer(votesReq(), {
        balance: wireBalance('unchecked', { count: 32 }),
        check: running([1, 2, 3, 4, 5], 4, 6, 16, { labelled: 15, right: 0, direction: 'shallower', bands: 2 }, { index: 1, lo: 9, hi: 16 }),
      });
      expect(el().querySelector('.check-walk')!.textContent).toContain('Past the peak: checking a shorter list.');
      for (let i = 0; i < 5; i++) await press('ArrowLeft');
      // It ends on the first band: no new picks, the result, with no verdict of short or met.
      const range = { lo: 0, hi: 0.45, labelled: 5, right: 0 };
      const recall = { lo: 0, hi: 0.2 };
      await answer(votesReq(), {
        balance: checkedBalance(8, range, recall),
        check: { ...finished(range, recall), round: 5, candidate: 8, direction: 'shallower', bands: 1 },
      });
      expect(voted.map((v) => v.finished)).toEqual([false, false, false, true]);
      expect(el().querySelector('.check-result-headline')!.textContent).toContain(
        'Checked the top 8: likely 0–45% right, few of them found (checked 5).',
      );
      expect(text()).toContain("The line keeps its 8, where your labels put it");
      expect(text()).not.toMatch(/\bshort\b|\bfell\b|\bmet\b|confirmed|threshold/i);
      // The copy names no cause: it is true of a sparse corpus and a weak model alike.
      expect(text()).not.toMatch(/sparse|weak|too few|model/i);
      expect(dots().length).toBe(0);
    });

    it('reads a checked result with no ranges as checked', async () => {
      await start([17]);
      await press('ArrowRight');
      await answer(votesReq(), {
        balance: wireBalance('checked', { precision: null, recall: null, fbeta: null }),
        check: { ...running([], 1, 1, 32, {}, null), status: 'checked' },
      });
      expect(el().querySelector('.check-result-headline')!.textContent!.trim()).toBe('Checked the top 32.');
      expect(voted.map((v) => v.finished)).toEqual([true]);
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

    it('cancels a check still drawing its first picks', async () => {
      (el().querySelector('.modal-footer .btn') as HTMLButtonElement).click();
      await settleZoneless(fixture);
      httpMock.expectOne((req) => req.url === '/api/precision-check/cancel');
      expect(closed).toBe(1);
    });

    it('closes a finished check without cancelling anything', async () => {
      await start([17]);
      await press('ArrowRight');
      const range = { lo: 0.55, hi: 1, labelled: 1, right: 1 };
      const recall = { lo: 0.5, hi: 1 };
      await answer(votesReq(), { balance: checkedBalance(32, range, recall), check: finished(range, recall) });
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
