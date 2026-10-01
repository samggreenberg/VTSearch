import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StripeOverviewComponent, STRIPE_MAX_ITEMS } from './stripe-overview.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { BALANCE_STATES, lineBalance } from '../../../testing/line-balance';

describe('StripeOverviewComponent', () => {
  let component: StripeOverviewComponent;
  let fixture: ComponentFixture<StripeOverviewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StripeOverviewComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(StripeOverviewComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should not be visible when no sort order', () => {
    expect(component.visible).toBe(false);
  });

  it('holds its place as an empty, inert track before anything is ranked (#4347)', async () => {
    // The label view snaps the left panel to its grid before the first ranking
    // lands; a strip that only appeared then would narrow the grid by a column.
    const el = fixture.nativeElement as HTMLElement;
    const track = el.querySelector('.stripe-overview');
    expect(track).not.toBeNull();
    expect(track!.classList).toContain('stripe-overview--empty');
    expect(track!.getAttribute('aria-hidden')).toBe('true');
    expect(el.querySelector('.stripe-container')).toBeNull();

    fixture.componentRef.setInput('sortOrder', [{ id: 1, score: 0.9 }]);
    await settleZoneless(fixture);
    expect(el.querySelector('.stripe-overview--empty')).toBeNull();
    expect(el.querySelector('.stripe-container')).not.toBeNull();
  });

  it('should be visible when sort order exists', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ]);
    expect(component.visible).toBe(true);
  });

  it('should generate good dots', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ]);
    fixture.componentRef.setInput('goodVotes', new Set([1]));
    const dots = component.cachedDots();
    const goodDots = dots.filter((d: { top: number; type: string }) => d.type === 'good');
    expect(goodDots.length).toBe(1);
    expect(goodDots[0].top).toBe(0);
  });

  it('should generate bad dots', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ]);
    fixture.componentRef.setInput('badVotes', new Set([2]));
    const dots = component.cachedDots();
    const badDots = dots.filter((d: { top: number; type: string }) => d.type === 'bad');
    expect(badDots.length).toBe(1);
    expect(badDots[0].top).toBe(50);
  });

  it('should generate selected dot', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ]);
    fixture.componentRef.setInput('selectedId', 2);
    const dots = component.cachedDots();
    const selectedDots = dots.filter((d: { top: number; type: string }) => d.type === 'selected');
    expect(selectedDots.length).toBe(1);
  });

  it('should calculate threshold position', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
      { id: 3, score: 0.1 },
    ]);
    fixture.componentRef.setInput('threshold', 0.4);
    expect(component.cachedThresholdPosition()).toBeCloseTo(66.67, 0);
  });

  it('should return null threshold position when no threshold', () => {
    fixture.componentRef.setInput('sortOrder', [{ id: 1, score: 0.9 }]);
    fixture.componentRef.setInput('threshold', null);
    expect(component.cachedThresholdPosition()).toBeNull();
  });

  it('is not oversized at or below the size cap', () => {
    fixture.componentRef.setInput(
      'sortOrder',
      Array.from({ length: 3 }, (_v, i) => ({ id: i, score: 1 - i * 0.1 })),
    );
    expect(component.oversized()).toBe(false);
  });

  it('goes oversized above the size cap and stops building dots/threshold', () => {
    const order = Array.from({ length: STRIPE_MAX_ITEMS + 1 }, (_v, i) => ({
      id: i,
      score: 1 - i / (STRIPE_MAX_ITEMS + 1),
    }));
    fixture.componentRef.setInput('sortOrder', order);
    fixture.componentRef.setInput('threshold', 0.5);
    fixture.componentRef.setInput('goodVotes', new Set([0]));
    fixture.componentRef.setInput('selectedId', 1);
    expect(component.oversized()).toBe(true);
    // Still "visible" (the strip stays in the layout) but inert: no dots and
    // no threshold marker are built, so the O(N) loop is skipped.
    expect(component.visible).toBe(true);
    expect(component.cachedDots()).toEqual([]);
    expect(component.cachedThresholdPosition()).toBeNull();
  });

  it('ignores clicks/keyboard when oversized', () => {
    const order = Array.from({ length: STRIPE_MAX_ITEMS + 1 }, (_v, i) => ({
      id: i,
      score: 1 - i / (STRIPE_MAX_ITEMS + 1),
    }));
    fixture.componentRef.setInput('sortOrder', order);
    let emitted = false;
    component.stripeClick.subscribe(() => (emitted = true));
    component.onStripeClick({
      currentTarget: { getBoundingClientRect: () => ({ top: 0, height: 100 }) },
      clientY: 50,
    } as unknown as MouseEvent);
    component.onStripeKeyboard();
    expect(emitted).toBe(false);
  });

  describe('the line marker in every balance state (#4272, #4273, #4413)', () => {
    async function drawWith(balance: ReturnType<typeof lineBalance> | null): Promise<HTMLElement> {
      fixture.componentRef.setInput('sortOrder', [
        { id: 1, score: 0.9 },
        { id: 2, score: 0.3 },
      ]);
      fixture.componentRef.setInput('threshold', 0.5);
      fixture.componentRef.setInput('balance', balance);
      await settleZoneless(fixture);
      return fixture.nativeElement as HTMLElement;
    }

    it.each(BALANCE_STATES)('draws the same plain marker when %s, and names the state in the strip tooltip', async (status) => {
      const el = await drawWith(lineBalance(status));
      const marker = el.querySelector('.stripe-threshold');
      expect(marker).not.toBeNull();
      expect(marker!.className).toBe('stripe-threshold');
      expect(component.cachedThresholdPosition()).toBe(50);
      const title = el.querySelector('.stripe-overview')!.getAttribute('title')!;
      expect(title).toContain('The line: ');
      expect(title).not.toMatch(/dashed|unpromised/);
    });

    it('draws a plain marker with no detector behind the sort', async () => {
      const el = await drawWith(null);
      expect(el.querySelector('.stripe-threshold')!.className).toBe('stripe-threshold');
      expect(el.querySelector('.stripe-overview')!.getAttribute('title')).not.toContain('The line: ');
    });
  });
});
