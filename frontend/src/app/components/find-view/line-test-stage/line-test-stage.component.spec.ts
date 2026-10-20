import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LineTestStageComponent } from './line-test-stage.component';
import { LineTestSessionService } from '../../../services/line-test-session.service';
import { PairScopeService } from '../../../services/pair-scope.service';
import { KeyboardService, type KeyboardAction } from '../../../services/keyboard.service';
import { MediaStateService } from '../../../services/media-state.service';
import { VoteStateService } from '../../../services/vote-state.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { estimate, wireDone, wireLineTest, wireTest } from '../../../testing/line-test';
import type { LineTestResponse } from '../../../generated/api-client/models/line-test-response';
import type { Media } from '../../../models/api.models';

describe('LineTestStageComponent (#4524)', () => {
  let fixture: ComponentFixture<LineTestStageComponent>;
  let session: LineTestSessionService;
  let httpMock: HttpTestingController;
  let keyboard: KeyboardService;

  beforeEach(async () => {
    await configureZoneless({
      imports: [LineTestStageComponent],
      providers: [...provideHttpTesting(), PairScopeService, LineTestSessionService],
    }).compileComponents();
    keyboard = TestBed.inject(KeyboardService);
    keyboard.start();
    vi.spyOn(TestBed.inject(VoteStateService), 'loadVotes').mockImplementation(() => undefined);
    vi.spyOn(TestBed.inject(MediaStateService), 'getMedia').mockImplementation(
      (id: number) => ({ id, media_type: 'image', filename: `${id}.png` }) as Media,
    );
    fixture = TestBed.createComponent(LineTestStageComponent);
    session = TestBed.inject(LineTestSessionService);
    httpMock = TestBed.inject(HttpTestingController);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    keyboard.stop();
    // The round's picks are fetched in a batch for the viewer; nothing here reads it.
    httpMock.match('/api/medias/batch').forEach((req) => req.flush([]));
    httpMock.verify();
  });

  const el = () => fixture.nativeElement as HTMLElement;
  const dots = () => Array.from(el().querySelectorAll('.pick-dot')) as HTMLButtonElement[];

  function press(key: string): void {
    document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  }

  async function dealRound(): Promise<void> {
    session.start();
    httpMock.expectOne('/api/line-test/start').flush(wireLineTest(wireTest({ picks: [51, 36, 41] })));
    await settleZoneless(fixture);
  }

  it('waits on the line before a test exists', () => {
    expect(el().querySelector('.stage-heading')!.textContent).toContain('Drawing picks');
    expect(el().querySelector('.pick-dot')).toBeNull();
  });

  it('shows the round\'s picks as dots in draw order, the first on screen, with the phase\'s prompt', async () => {
    await dealRound();
    expect(dots().length).toBe(3);
    expect(dots()[0].classList.contains('current')).toBe(true);
    expect(el().querySelector('.stage-prompt')!.textContent).toContain('Checking the matches: 3 picks drawn at random from items 33–64 of the list');
    expect(el().querySelector('vt-image-viewer')).not.toBeNull();
    // A pick carries no rank and no score.
    expect(el().textContent).not.toMatch(/#51|rank/i);
  });

  it('→ votes Good and moves on; the dot fills with the vote', async () => {
    await dealRound();
    press('ArrowRight');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest({ picks: [36, 41], labelled: 1 })));
    await settleZoneless(fixture);
    expect(dots()[0].dataset['vote']).toBe('good');
    expect(dots()[1].classList.contains('current')).toBe(true);
  });

  it('↓ goes back a pick and takes its vote back', async () => {
    await dealRound();
    press('ArrowLeft');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest({ picks: [36, 41], labelled: 1 })));
    await settleZoneless(fixture);
    press('ArrowDown');
    httpMock.expectOne('/api/line-test/unvote').flush(wireLineTest(wireTest({ picks: [51, 36, 41] })));
    await settleZoneless(fixture);
    expect(dots()[0].classList.contains('current')).toBe(true);
    expect(dots()[0].dataset['vote']).toBeUndefined();
  });

  it('holds the vote keys so nothing reaches the list behind it', async () => {
    await dealRound();
    const actions: KeyboardAction[] = [];
    keyboard.action$.subscribe((a) => actions.push(a));
    press('ArrowRight');
    httpMock.expectOne('/api/line-test/votes').flush(wireLineTest(wireTest({ picks: [36, 41], labelled: 1 })));
    expect(actions).toEqual([]);
  });

  it('clicking a dot shows that pick', async () => {
    await dealRound();
    dots()[2].click();
    await settleZoneless(fixture);
    expect(dots()[2].classList.contains('current')).toBe(true);
  });

  async function done(response: LineTestResponse): Promise<void> {
    session.load();
    httpMock.expectOne('/api/line-test').flush(response);
    await settleZoneless(fixture);
  }

  it('says the test is done, with Toasty pointing at AutoFind and the Review tab (#4807)', async () => {
    await done(wireLineTest(wireDone()));
    expect(el().querySelector('.stage-heading')!.textContent).toContain('Test done');
    expect(el().querySelector('.stage-body')).toBeNull();
    const hint = el().querySelector('vt-toasty-hint');
    expect(hint?.getAttribute('hintId')).toBe('test-done');
    expect(hint?.textContent).toContain('AutoFind');
    expect(hint?.textContent).toContain('Review');
  });

  it('at Done is the verdict, with its exits and the Threshold leaning on the presets (#5100)', async () => {
    const presets = [
      { beta: 0.25, count: 31, precision: estimate(0.9, 0.75, 1), recall: estimate(0.3, 0.2, 0.4), fbeta: estimate(0.5, 0.4, 0.6), found: 'about a quarter of them found' },
      { beta: 1, count: 64, precision: estimate(0.75, 0.55, 0.95), recall: estimate(0.5, 0.3, 0.7), fbeta: estimate(0.6, 0.45, 0.78), found: 'about half of them found' },
      { beta: 4, count: 128, precision: estimate(0.5, 0.35, 0.65), recall: estimate(0.8, 0.6, 0.95), fbeta: estimate(0.7, 0.55, 0.85), found: 'about three quarters of them found' },
    ];
    await done(wireLineTest(wireDone(), { presets }));
    const card = el().querySelector('.verdict-card') as HTMLElement;
    expect(card.querySelector('.section-title')!.textContent).toContain('Verdict');
    expect(card.querySelector('.verdict')!.textContent).toContain(
      'the line keeps the top 64: likely 55–95% of them are right, with about half of them found (30–70% of all the matches)',
    );
    const exits = Array.from(card.querySelectorAll('.exit-btn')).map((b) => b.textContent!.trim());
    expect(exits).toEqual(['Move to AutoFind', 'Add Corrections and retrain']);

    const rows = Array.from(card.querySelectorAll('.presets tbody tr')) as HTMLElement[];
    expect(rows.map((r) => r.querySelector('.preset-btn')!.textContent!.trim())).toEqual(['False Positives', 'Between', 'False Negatives']);
    expect(rows[1].classList.contains('current')).toBe(true);
    expect((rows[1].querySelector('.preset-btn') as HTMLButtonElement).disabled).toBe(true);
    expect(rows[0].textContent).toContain('128');
    expect(rows[0].textContent).toContain('35–65%');

    const said: string[] = [];
    fixture.componentInstance.lean.subscribe((b) => said.push(`lean ${b}`));
    fixture.componentInstance.moveToAutoFind.subscribe(() => said.push('autofind'));
    fixture.componentInstance.addCorrections.subscribe(() => said.push('corrections'));
    (rows[0].querySelector('.preset-btn') as HTMLButtonElement).click();
    (card.querySelector('.exit-btn.btn--primary') as HTMLButtonElement).click();
    (card.querySelector('.exit-btn.btn--secondary') as HTMLButtonElement).click();
    expect(said).toEqual(['lean 4', 'autofind', 'corrections']);
  });

  it('with no class model reads Found in the verdict in words only (#4542)', async () => {
    await done(wireLineTest(wireDone({ class_model: false })));
    const verdict = el().querySelector('.verdict')!.textContent!;
    expect(verdict).toContain('with about half of them found. That is');
    expect(verdict).not.toContain('of all the matches');
  });

  it('says over the verdict that it is stale after corrections, or that the line moved, offering a new test', async () => {
    await done(wireLineTest(wireDone(), { stale: true }));
    expect(el().querySelector('.verdict-card .stale-note')!.textContent).toContain('Out of date');

    await done(wireLineTest(wireDone(), { moved: true, line_count: 128 }));
    const note = el().querySelector('.verdict-card .stale-note')!;
    expect(note.textContent).toContain('measured the top 64');
    expect(note.textContent).toContain('keeps the top 128 now');
    const again: number[] = [];
    fixture.componentInstance.testAgain.subscribe(() => again.push(1));
    (note.querySelector('button') as HTMLButtonElement).click();
    expect(again.length).toBe(1);
  });

  it('says there is nothing to test', async () => {
    session.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest({ phase: 'nothing', picks: [] })));
    await settleZoneless(fixture);
    expect(el().querySelector('.stage-heading')!.textContent).toContain('Nothing to test');
  });
});
