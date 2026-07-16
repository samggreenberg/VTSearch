import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StripeOverviewComponent } from './stripe-overview.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

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
});
