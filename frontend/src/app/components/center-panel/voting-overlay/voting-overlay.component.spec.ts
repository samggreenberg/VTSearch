import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VotingOverlayComponent } from './voting-overlay.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('VotingOverlayComponent', () => {
  let component: VotingOverlayComponent;
  let fixture: ComponentFixture<VotingOverlayComponent>;

  beforeEach(async () => {
    await configureZoneless({
      imports: [VotingOverlayComponent],
      // Toasty's hint reads the user's settings.
      providers: [...provideHttpTesting()],
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

  it('has no hint over the buttons by default', () => {
    expect(fixture.nativeElement.querySelector('vt-toasty-hint')).toBeNull();
  });

  it("puts Toasty's start-voting hint over the buttons when asked (#4680)", async () => {
    fixture.componentRef.setInput('hint', 'start-voting');
    await settleZoneless(fixture);
    const hint = fixture.nativeElement.querySelector('vt-toasty-hint');
    expect(hint).toBeTruthy();
    expect(hint.classList).toContain('toasty-hint--above');
    expect(hint.textContent).toContain('Autopilot picked this item for you');
  });

  it("says what each Autopilot phase is after, with its counts, and what a test's pick is (#4807)", async () => {
    const text = () => (fixture.nativeElement.querySelector('vt-toasty-hint')?.textContent ?? '').replace(/\s+/g, ' ');
    fixture.componentRef.setInput('hint', 'find-goods');
    fixture.componentRef.setInput('hintCounts', { goods: 1, goodsNeeded: 3, bads: 0, badsNeeded: 4 });
    await settleZoneless(fixture);
    expect(text()).toContain('Autopilot needs 3 Goods to start training');
    expect(text()).toContain("you've found 1");

    fixture.componentRef.setInput('hint', 'find-bads');
    fixture.componentRef.setInput('hintCounts', { goods: 3, goodsNeeded: 3, bads: 2, badsNeeded: 4 });
    await settleZoneless(fixture);
    expect(text()).toContain('Autopilot needs 4 Bads too, and has 2');

    fixture.componentRef.setInput('hint', 'keep-voting');
    await settleZoneless(fixture);
    expect(text()).toContain('retrains on every vote');

    fixture.componentRef.setInput('hint', 'test-vote');
    await settleZoneless(fixture);
    expect(text()).toContain("picks from both sides of your detector's line");

    fixture.componentRef.setInput('hint', null);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('vt-toasty-hint')).toBeNull();
  });

  it("names a picked item's label, the button that changes it, and the way back (#4824)", async () => {
    const text = () => (fixture.nativeElement.querySelector('vt-toasty-hint')?.textContent ?? '').replace(/\s+/g, ' ');
    fixture.componentRef.setInput('hint', 'revote');
    fixture.componentRef.setInput('isGood', true);
    await settleZoneless(fixture);
    expect(text()).toContain('You already labeled this item Good. Click Bad (←) to change it');
    expect(text()).toContain("click Autopilot's current step on the left (↑) to go back to its pick");

    fixture.componentRef.setInput('isGood', false);
    fixture.componentRef.setInput('isBad', true);
    await settleZoneless(fixture);
    expect(text()).toContain('You already labeled this item Bad. Click Good (→) to change it');
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
