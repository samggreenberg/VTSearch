import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StripeOverviewComponent } from './stripe-overview.component';

describe('StripeOverviewComponent', () => {
  let component: StripeOverviewComponent;
  let fixture: ComponentFixture<StripeOverviewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StripeOverviewComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(StripeOverviewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should not be visible when no sort order', () => {
    expect(component.visible).toBeFalse();
  });

  it('should be visible when sort order exists', () => {
    component.sortOrder = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ];
    expect(component.visible).toBeTrue();
  });

  it('should generate good dots', () => {
    component.sortOrder = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ];
    component.goodVotes = new Set([1]);
    component.ngOnChanges();
    const dots = component.cachedDots;
    const goodDots = dots.filter((d: { top: number; type: string }) => d.type === 'good');
    expect(goodDots.length).toBe(1);
    expect(goodDots[0].top).toBe(0);
  });

  it('should generate bad dots', () => {
    component.sortOrder = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ];
    component.badVotes = new Set([2]);
    component.ngOnChanges();
    const dots = component.cachedDots;
    const badDots = dots.filter((d: { top: number; type: string }) => d.type === 'bad');
    expect(badDots.length).toBe(1);
    expect(badDots[0].top).toBe(50);
  });

  it('should generate selected dot', () => {
    component.sortOrder = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ];
    component.selectedId = 2;
    component.ngOnChanges();
    const dots = component.cachedDots;
    const selectedDots = dots.filter((d: { top: number; type: string }) => d.type === 'selected');
    expect(selectedDots.length).toBe(1);
  });

  it('should calculate threshold position', () => {
    component.sortOrder = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
      { id: 3, score: 0.1 },
    ];
    component.threshold = 0.4;
    component.ngOnChanges();
    expect(component.cachedThresholdPosition).toBeCloseTo(66.67, 0);
  });

  it('should return null threshold position when no threshold', () => {
    component.sortOrder = [{ id: 1, score: 0.9 }];
    component.threshold = null;
    component.ngOnChanges();
    expect(component.cachedThresholdPosition).toBeNull();
  });
});
