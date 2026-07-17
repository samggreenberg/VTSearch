import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ProgressBarComponent } from './progress-bar.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('ProgressBarComponent', () => {
  let component: ProgressBarComponent;
  let fixture: ComponentFixture<ProgressBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProgressBarComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(ProgressBarComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should calculate percentage correctly', () => {
    fixture.componentRef.setInput('value', 50);
    fixture.componentRef.setInput('max', 200);
    expect(component.percentage).toBe(25);
  });

  it('should clamp percentage to 100', () => {
    fixture.componentRef.setInput('value', 150);
    fixture.componentRef.setInput('max', 100);
    expect(component.percentage).toBe(100);
  });

  it('should handle zero max', () => {
    fixture.componentRef.setInput('value', 50);
    fixture.componentRef.setInput('max', 0);
    expect(component.percentage).toBe(0);
  });

  // Zoneless: drive inputs via `setInput` (a CD trigger) and assert the rendered
  // DOM after `whenStable()` — no manual `detectChanges`.
  it('should render progressbar role', async () => {
    await settleZoneless(fixture);
    const track = fixture.nativeElement.querySelector('.progress-track');
    expect(track.getAttribute('role')).toBe('progressbar');
  });

  it('should set width style from percentage', async () => {
    fixture.componentRef.setInput('value', 75);
    fixture.componentRef.setInput('max', 100);
    await settleZoneless(fixture);
    const fill = fixture.nativeElement.querySelector('.progress-fill') as HTMLElement;
    expect(fill.style.width).toBe('75%');
  });

  it('should apply indeterminate class', async () => {
    fixture.componentRef.setInput('indeterminate', true);
    await settleZoneless(fixture);
    const fill = fixture.nativeElement.querySelector('.progress-fill');
    expect(fill.classList).toContain('indeterminate');
  });

  it('should not have aria-valuenow when indeterminate', async () => {
    fixture.componentRef.setInput('indeterminate', true);
    await settleZoneless(fixture);
    const track = fixture.nativeElement.querySelector('.progress-track');
    expect(track.getAttribute('aria-valuenow')).toBeNull();
  });

  it('should not apply the smooth modifier by default', async () => {
    await settleZoneless(fixture);
    const fill = fixture.nativeElement.querySelector('.progress-fill');
    expect(fill.classList).not.toContain('progress-fill--smooth');
  });

  it('should apply the smooth modifier when [smooth] is set', async () => {
    fixture.componentRef.setInput('smooth', true);
    await settleZoneless(fixture);
    const fill = fixture.nativeElement.querySelector('.progress-fill');
    expect(fill.classList).toContain('progress-fill--smooth');
  });

  describe('fillColor', () => {
    it('should be reddest at 0%', () => {
      fixture.componentRef.setInput('value', 0);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--text-warning) 0%, var(--color-bad))');
    });

    it('should be yellowest at 50%', () => {
      fixture.componentRef.setInput('value', 50);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--text-warning) 100%, var(--color-bad))');
    });

    it('should be greenest at 100%', () => {
      fixture.componentRef.setInput('value', 100);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--color-good) 100%, var(--text-warning))');
    });

    it('should interpolate continuously between red and yellow below 50%', () => {
      fixture.componentRef.setInput('value', 25);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--text-warning) 50%, var(--color-bad))');
    });

    it('should interpolate continuously between yellow and green above 50%', () => {
      fixture.componentRef.setInput('value', 75);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--color-good) 50%, var(--text-warning))');
    });

    it('should be null while indeterminate', () => {
      fixture.componentRef.setInput('indeterminate', true);
      expect(component.fillColor).toBeNull();
    });

    it('should apply the fill color as the background style', async () => {
      fixture.componentRef.setInput('value', 75);
      fixture.componentRef.setInput('max', 100);
      await settleZoneless(fixture);
      const fill = fixture.nativeElement.querySelector('.progress-fill') as HTMLElement;
      expect(fill.style.background).toContain('color-mix');
    });
  });

  describe('polarity: high-bad', () => {
    beforeEach(() => {
      fixture.componentRef.setInput('polarity', 'high-bad');
    });

    it('should be greenest at 0% (empty is good)', () => {
      fixture.componentRef.setInput('value', 0);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--color-good) 100%, var(--text-warning))');
    });

    it('should be yellowest at 50%', () => {
      fixture.componentRef.setInput('value', 50);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--text-warning) 100%, var(--color-bad))');
    });

    it('should be reddest at 100% (full is bad)', () => {
      fixture.componentRef.setInput('value', 100);
      fixture.componentRef.setInput('max', 100);
      expect(component.fillColor).toBe('color-mix(in srgb, var(--text-warning) 0%, var(--color-bad))');
    });

    it('mirrors the high-good gradient', () => {
      fixture.componentRef.setInput('value', 25);
      fixture.componentRef.setInput('max', 100);
      // 25% under high-bad colors as 75% would under high-good.
      expect(component.fillColor).toBe('color-mix(in srgb, var(--color-good) 50%, var(--text-warning))');
    });

    it('should be null while indeterminate', () => {
      fixture.componentRef.setInput('indeterminate', true);
      expect(component.fillColor).toBeNull();
    });
  });
});
