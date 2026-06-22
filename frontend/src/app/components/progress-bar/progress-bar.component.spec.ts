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
    component.value = 50;
    component.max = 200;
    expect(component.percentage).toBe(25);
  });

  it('should clamp percentage to 100', () => {
    component.value = 150;
    component.max = 100;
    expect(component.percentage).toBe(100);
  });

  it('should handle zero max', () => {
    component.value = 50;
    component.max = 0;
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
});
