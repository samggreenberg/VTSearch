import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VotingOverlayComponent } from './voting-overlay.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { STOP_CAP_VOTES, TYPICAL_DONE_VOTES } from '../../../services/autopilot-state.service';

/** `keep-voting`'s one line with numbers in it (#5539), the exception to #5143's rule. */
const KEEP_VOTING_HOW_LONG = /Most detectors are trained within \d+ votes, and I never ask for more than \d+\./;

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
    expect(hint.textContent).toContain('I picked this for you');
  });

  it("says what each Autopilot phase is after, in plain words, and what a test's pick is (#4807, #5143, #5538)", async () => {
    const text = () => (fixture.nativeElement.querySelector('vt-toasty-hint')?.textContent ?? '').replace(/\s+/g, ' ');
    const phases = [
      ['start-voting', 'Click Good if'],
      ['find-goods', "I'm looking for Goods now. Keep clicking."],
      ['find-bads', "I'm looking for Bads now. Keep clicking."],
      ['find-more', "I'm looking for more Goods now. Keep clicking."],
      ['keep-voting', "I'm sharpening your detector now. Keep clicking."],
      ['spot-check', "I'm double-checking your detector now. Keep clicking."],
    ] as const;
    for (const [hint, says] of phases) {
      fixture.componentRef.setInput('hint', hint);
      await settleZoneless(fixture);
      expect(text()).toContain(says);
      // Autopilot is for clicking: no counts, targets or keys to take in
      // (#5143), bar the one line on how long a run takes (#5539).
      const rest = hint === 'keep-voting' ? text().replace(KEEP_VOTING_HOW_LONG, '') : text();
      expect(rest).not.toMatch(/\d|arrow key/);
      // Toasty speaks as Autopilot, in the first person (#5538).
      expect(text()).not.toMatch(/Autopilot|VTSearch/);
    }

    fixture.componentRef.setInput('hint', 'test-vote');
    await settleZoneless(fixture);
    expect(text()).toContain("Tune tests your detector's line with picks from both sides of it");

    fixture.componentRef.setInput('hint', null);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('vt-toasty-hint')).toBeNull();
  });

  it('says, while Autopilot sharpens the detector, how many votes a run takes (#5539)', async () => {
    fixture.componentRef.setInput('hint', 'keep-voting');
    await settleZoneless(fixture);
    const text = fixture.nativeElement.querySelector('vt-toasty-hint').textContent.replace(/\s+/g, ' ');
    expect(text).toContain(
      `Most detectors are trained within ${TYPICAL_DONE_VOTES} votes, and I never ask for more than ${STOP_CAP_VOTES}.`,
    );
    expect(text).toContain('Most detectors are trained within 50 votes, and I never ask for more than 200.');
  });

  it("names a picked item's label, the button that changes it, and the way back (#4824)", async () => {
    const text = () => (fixture.nativeElement.querySelector('vt-toasty-hint')?.textContent ?? '').replace(/\s+/g, ' ');
    fixture.componentRef.setInput('hint', 'revote');
    fixture.componentRef.setInput('isGood', true);
    await settleZoneless(fixture);
    expect(text()).toContain('You already labeled this Good. Click Bad to change it');
    expect(text()).toContain('or my step on the left to carry on.');

    fixture.componentRef.setInput('isGood', false);
    fixture.componentRef.setInput('isBad', true);
    await settleZoneless(fixture);
    expect(text()).toContain('You already labeled this Bad. Click Good to change it');
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

  describe('read-only (a received detector, #5578)', () => {
    beforeEach(async () => {
      fixture.componentRef.setInput('readOnly', true);
      fixture.componentRef.setInput('hint', 'start-voting');
      await settleZoneless(fixture);
    });

    it('disables both buttons, says why, and offers Fork', () => {
      const el = fixture.nativeElement as HTMLElement;
      const good = el.querySelector('.btn-good') as HTMLButtonElement;
      const bad = el.querySelector('.btn-bad') as HTMLButtonElement;
      expect(good.disabled).toBe(true);
      expect(bad.disabled).toBe(true);
      expect(good.title).toContain('Read-only: fork to edit');
      expect(el.querySelector('.read-only-hint')?.textContent).toContain('Read-only: fork to edit');
      // Toasty's hints step aside: none of them can be acted on.
      expect(el.querySelector('vt-toasty-hint')).toBeNull();

      let forked = 0;
      component.fork.subscribe(() => forked++);
      (el.querySelector('.fork-btn') as HTMLButtonElement).click();
      expect(forked).toBe(1);
    });

    it('casts no vote even when called directly', () => {
      let emitted = 0;
      component.voted.subscribe(() => emitted++);
      component.onVoteGood();
      component.onVoteBad();
      expect(emitted).toBe(0);
    });

    it('opens again once the detector is no longer read-only', async () => {
      fixture.componentRef.setInput('readOnly', false);
      await settleZoneless(fixture);
      const el = fixture.nativeElement as HTMLElement;
      expect((el.querySelector('.btn-good') as HTMLButtonElement).disabled).toBe(false);
      expect(el.querySelector('.read-only-hint')).toBeNull();
    });
  });
});
