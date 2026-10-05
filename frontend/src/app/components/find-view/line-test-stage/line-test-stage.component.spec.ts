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
import { wireDone, wireLineTest, wireTest } from '../../../testing/line-test';
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

  it('says the test is done, and points at the result and the Review tab', async () => {
    session.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireDone()));
    await settleZoneless(fixture);
    expect(el().querySelector('.stage-heading')!.textContent).toContain('Test done');
    expect(el().querySelector('.stage-body')!.textContent).toContain('Review tab');
  });

  it('says there is nothing to test', async () => {
    session.load();
    httpMock.expectOne('/api/line-test').flush(wireLineTest(wireTest({ phase: 'nothing', picks: [] })));
    await settleZoneless(fixture);
    expect(el().querySelector('.stage-heading')!.textContent).toContain('Nothing to test');
  });
});
