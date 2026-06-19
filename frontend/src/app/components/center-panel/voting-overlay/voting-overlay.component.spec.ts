import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VotingOverlayComponent } from './voting-overlay.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('VotingOverlayComponent', () => {
  let component: VotingOverlayComponent;
  let fixture: ComponentFixture<VotingOverlayComponent>;

  beforeEach(async () => {
    await configureZoneless({
      imports: [VotingOverlayComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(VotingOverlayComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render Good and Bad buttons', () => {
    const buttons = fixture.nativeElement.querySelectorAll('button');
    expect(buttons.length).toBe(2);
    expect(buttons[0].textContent.trim()).toBe('Bad');
    expect(buttons[1].textContent.trim()).toBe('Good');
  });

  it('should emit good on Good click', () => {
    let emitted: string | undefined;
    component.voted.subscribe((v: string) => (emitted = v));
    fixture.nativeElement.querySelector('.btn-good').click();
    expect(emitted).toBe('good');
  });

  it('should emit bad on Bad click', () => {
    let emitted: string | undefined;
    component.voted.subscribe((v: string) => (emitted = v));
    fixture.nativeElement.querySelector('.btn-bad').click();
    expect(emitted).toBe('bad');
  });

  it('should apply voted class when isGood is true', async () => {
    fixture.componentRef.setInput('isGood', true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.btn-good').classList.contains('voted')).toBe(true);
  });

  it('should apply voted class when isBad is true', async () => {
    fixture.componentRef.setInput('isBad', true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.btn-bad').classList.contains('voted')).toBe(true);
  });

  it('should not emit when disabled', async () => {
    fixture.componentRef.setInput('disabled', true);
    await settleZoneless(fixture);
    let emitted = false;
    component.voted.subscribe(() => (emitted = true));
    fixture.nativeElement.querySelector('.btn-good').click();
    expect(emitted).toBe(false);
  });

  it('should drop focus from the clicked button so Shift cannot re-ring it', () => {
    // The button keeps DOM focus after a mouse click; pressing a modifier later
    // would promote it to :focus-visible and paint a ring. Blurring on click
    // prevents that, keeping Shift a pure cursor modifier during region voting.
    const good = fixture.nativeElement.querySelector('.btn-good') as HTMLButtonElement;
    good.focus();
    good.click();
    expect(document.activeElement).not.toBe(good);
  });

  it('should hide the first-vote hint by default', () => {
    expect(fixture.nativeElement.querySelector('.vote-hint')).toBeNull();
  });

  it('should render the first-vote hint text when showHint is true', async () => {
    fixture.componentRef.setInput('showHint', true);
    await settleZoneless(fixture);
    const hint = fixture.nativeElement.querySelector('.vote-hint');
    expect(hint).toBeTruthy();
    expect(hint.textContent.trim()).toContain('Use');
    expect(hint.textContent).toContain('Autopilot');
  });

  // Zoneless staleness canary: the flash class is added in a bound `(click)`
  // (which schedules CD on its own) but cleared by a `setTimeout`. The reset is
  // the zoneless-sensitive write — it only repaints because `goodFlash` is a
  // signal read in the template. Click, confirm the flash paints, then let the
  // 300ms timer fire and confirm the class is removed with no manual pump.
  it('flashes on vote and clears the flash class after the timer (zoneless canary)', async () => {
    const good = fixture.nativeElement.querySelector('.btn-good') as HTMLButtonElement;
    good.click();
    await settleZoneless(fixture);
    expect(good.classList.contains('vote-flash')).toBe(true);

    await new Promise<void>((resolve) => setTimeout(resolve, 350));
    await fixture.whenStable();
    expect(good.classList.contains('vote-flash')).toBe(false);
  });
});
