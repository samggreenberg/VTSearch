import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AutopilotCompleteModalComponent } from './autopilot-complete-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('AutopilotCompleteModalComponent', () => {
  let fixture: ComponentFixture<AutopilotCompleteModalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotCompleteModalComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotCompleteModalComponent);
    fixture.componentRef.setInput('heading', 'Detector Trained');
    fixture.componentRef.setInput('detail', 'Every quality indicator is green.');
    fixture.componentRef.setInput('nextSteps', 'Keep labeling, or head to the Dashboard.');
    await settleZoneless(fixture);
  });

  function footerButtons(): HTMLButtonElement[] {
    return [...fixture.nativeElement.querySelectorAll('.modal-footer button')];
  }

  it('renders the per-phase copy', () => {
    expect(fixture.nativeElement.querySelector('.modal-header h2').textContent)
      .toContain('Detector Trained');
    const body = fixture.nativeElement.querySelector('.modal-body').textContent;
    expect(body).toContain('Every quality indicator is green.');
    expect(body).toContain('Keep labeling, or head to the Dashboard.');
  });

  it('offers the stay option before the leave option', () => {
    expect(footerButtons().map((b) => b.textContent!.trim()))
      .toEqual(['Continue Training', 'Head to Dashboard']);
  });

  it('lets the stay label be renamed per phase', async () => {
    fixture.componentRef.setInput('stayLabel', 'Stay Here');
    await settleZoneless(fixture);
    expect(footerButtons()[0].textContent!.trim()).toBe('Stay Here');
  });

  it('emits stay, never a navigation, when the user declines the hand-off', () => {
    const stay = vi.fn();
    const leave = vi.fn();
    fixture.componentInstance.stay.subscribe(stay);
    fixture.componentInstance.goToDashboard.subscribe(leave);

    footerButtons()[0].click();

    expect(stay).toHaveBeenCalledTimes(1);
    expect(leave).not.toHaveBeenCalled();
  });

  it('emits goToDashboard when the user takes the hand-off', () => {
    const leave = vi.fn();
    fixture.componentInstance.goToDashboard.subscribe(leave);

    footerButtons()[1].click();

    expect(leave).toHaveBeenCalledTimes(1);
  });

  it('treats dismissing the dialog as staying, not as leaving', async () => {
    // Escape / backdrop click must never be a shortcut out of the Train window:
    // the only way to the Dashboard is the explicit button.
    const stay = vi.fn();
    const leave = vi.fn();
    fixture.componentInstance.stay.subscribe(stay);
    fixture.componentInstance.goToDashboard.subscribe(leave);

    fixture.nativeElement.querySelector('.modal-backdrop')
      .dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settleZoneless(fixture);

    expect(stay).toHaveBeenCalledTimes(1);
    expect(leave).not.toHaveBeenCalled();
  });

  it('offers no close button — both outcomes are explicit', () => {
    expect(fixture.nativeElement.querySelector('.modal-close')).toBeNull();
  });
});
