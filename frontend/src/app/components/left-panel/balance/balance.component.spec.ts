import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BalanceComponent } from './balance.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { BALANCE_STATES, CHECKED_PRECISION, CHECKED_RECALL, lineBalance } from '../../../testing/line-balance';
import type { LineBalance } from '../../../utils/line-balance';

describe('BalanceComponent (#4413, #4317)', () => {
  let component: BalanceComponent;
  let fixture: ComponentFixture<BalanceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BalanceComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(BalanceComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  async function show(
    value: number | null,
    balance: LineBalance | null = null,
    returned: number | null = null,
    busy = false,
  ) {
    fixture.componentRef.setInput('value', value);
    fixture.componentRef.setInput('balance', balance);
    fixture.componentRef.setInput('returned', returned);
    fixture.componentRef.setInput('busy', busy);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  const root = () => fixture.nativeElement as HTMLElement;
  const radios = () => Array.from(root().querySelectorAll('input[type="radio"]')) as HTMLInputElement[];
  const checkedValue = () => radios().find((r) => r.checked)?.value ?? null;
  const state = () => root().querySelector('.balance-state');
  const stateText = () => root().querySelector('.balance-state-text');
  const hints = () => root().querySelectorAll('vt-field-hint-icon');

  describe('the spectrum (#4317)', () => {
    it('reads "Threshold:", from False Positives to False Negatives', () => {
      expect(root().querySelector('.balance-label')!.textContent!.trim()).toBe('Threshold:');
      const ends = Array.from(root().querySelectorAll('.spectrum-bar span')).map((e) => e.textContent!.trim());
      expect(ends).toEqual(['False Positives', 'False Negatives']);
      expect(root().textContent).not.toContain('Lean');
      expect(root().querySelector('select')).toBeNull();
    });

    it('offers three radios under it, beta 2 / 1 / 0.5 left to right, starting at the balanced middle', () => {
      expect(radios().map((r) => r.value)).toEqual(['2', '1', '0.5']);
      expect(checkedValue()).toBe('1');
      // One grid column per third of the spectrum, each radio centred in its own.
      expect(root().querySelectorAll('.spectrum-radios > .spectrum-radio').length).toBe(3);
    });

    it('names no balance and numbers none: each radio says where it sits only in its tooltip', () => {
      expect(root().querySelector('.balance-spectrum')!.textContent!.replace(/\s+/g, '')).toBe('FalsePositivesFalseNegatives');
      expect(root().querySelector('.balance-head')!.textContent).not.toMatch(/\d/);
      const titles = Array.from(root().querySelectorAll('.spectrum-radio')).map((l) => l.getAttribute('title'));
      expect(titles[0]).toMatch(/^Toward false positives/);
      expect(titles[1]).toBe('Between the two');
      expect(titles[2]).toMatch(/^Toward false negatives/);
      expect(radios().map((r) => r.getAttribute('aria-label'))).toEqual(titles);
    });

    it('is one radio group, labelled by its heading', () => {
      const group = root().querySelector('[role="radiogroup"]')!;
      expect(root().querySelector(`#${group.getAttribute('aria-labelledby')}`)!.textContent!.trim()).toBe('Threshold:');
      expect(new Set(radios().map((r) => r.name)).size).toBe(1);
    });

    it('is never a range slider, whose arrow keys would move it as they cast votes', () => {
      expect(root().querySelector('input[type="range"]')).toBeNull();
    });

    it('follows the value through every transition', async () => {
      for (const value of [0.5, 2, 1]) {
        await show(value);
        expect(checkedValue()).toBe(String(value));
      }
    });

    it('emits the picked balance as a beta', () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      radios()[0].click();
      expect(emitted).toHaveBeenCalledExactlyOnceWith(2);
    });

    it('shows the balance the host holds, not the click: a dropped pick leaves it where it was', async () => {
      await show(1);
      radios()[2].click();
      await settleZoneless(fixture);
      // The host never took the pick (Find drops one mid-pass).
      expect(checkedValue()).toBe('1');
      await show(0.5);
      expect(checkedValue()).toBe('0.5');
    });

    it('does not re-emit the balance it already shows', () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      radios()[1].click();
      expect(emitted).not.toHaveBeenCalled();
    });

    it('hands focus back after a pick, so the arrow keys vote again', () => {
      radios()[2].focus();
      expect(document.activeElement).toBe(radios()[2]);
      radios()[2].click();
      expect(document.activeElement).not.toBe(radios()[2]);
    });
  });

  /** A stored balance off the list: one set through the CLI or the API. */
  describe('a stored balance off the list', () => {
    it.each<[number, number]>([
      [4, 2],
      [1.5, 2],
      [0.75, 1],
      [0.25, 0.5],
    ])('shows %s on the nearest radio in log space, %s, and snaps to it', async (stored, snapped) => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(stored);
      expect(checkedValue()).toBe(String(snapped));
      expect(emitted).toHaveBeenCalledExactlyOnceWith(snapped);
    });

    it('waits for a running sort before it snaps', async () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(4, null, null, true);
      expect(emitted).not.toHaveBeenCalled();
      expect(checkedValue()).toBe('2');

      fixture.componentRef.setInput('busy', false);
      await settleZoneless(fixture);
      expect(emitted).toHaveBeenCalledExactlyOnceWith(2);
    });

    it('leaves a preset alone', async () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      for (const value of [2, 1, 0.5]) await show(value);
      expect(emitted).not.toHaveBeenCalled();
    });
  });

  describe('the state line', () => {
    it('is absent with no line from the detector', async () => {
      await show(1, null);
      expect(state()).toBeNull();
    });

    it('says a checked line is checked, with the share right, how many were found in words, and the count kept', async () => {
      await show(0.5, lineBalance('checked', { beta: 0.5 }), 1234);
      expect(state()!.getAttribute('data-status')).toBe('green');
      expect(stateText()!.textContent).toContain('Checked · likely 55–100% right, about half of them found (checked 5) · 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('5 random picks from the 32 items the line keeps');
      expect(stateText()!.getAttribute('title')).toContain("the set where the check's balance peaked");
    });

    it('reads the count off the result, never the preset', async () => {
      await show(2, lineBalance('checked', { beta: 2, count: 64 }));
      expect(stateText()!.textContent).toContain('64 kept');
    });

    it('says an unchecked line keeps the starting candidate', async () => {
      await show(2, lineBalance('unchecked', { beta: 2, count: 128 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Top 128 kept, unchecked');
      expect(stateText()!.getAttribute('title')).toContain('nothing has measured how much of it is right');
      expect(stateText()!.textContent).not.toMatch(/likely|%\s*right/);
    });

    it('never shows red: nothing falls short of a balance', async () => {
      for (const status of BALANCE_STATES) {
        await show(1, lineBalance(status), 300);
        expect(state()!.getAttribute('data-status')).not.toBe('red');
        expect(stateText()!.textContent).not.toMatch(/short|confirmed/i);
      }
    });

    it('prices a check off the schedule, rounds and all, on the check button', async () => {
      await show(2, lineBalance('unchecked', { beta: 2, count: 128, schedule: { candidate: 128, rounds: 5, picks: 5 } }));
      expect(root().querySelector('.balance-check-btn')!.getAttribute('title')).toContain('5 random picks a band, walking the list from the top 128');
      expect(root().querySelector('.balance-check-btn')!.getAttribute('title')).toContain('while the balance keeps improving');
    });

    it('notes a stale checked range only in the tooltip', async () => {
      const fresh = lineBalance('checked');
      await show(1, fresh, 300);
      const freshText = stateText()!.textContent;
      const freshMarkup = state()!.outerHTML.replace(/title="[^"]*"/g, '');
      await show(1, lineBalance('checked', { precision: { ...CHECKED_PRECISION, stale: true }, recall: { ...CHECKED_RECALL, stale: true } }), 300);
      expect(stateText()!.textContent).toBe(freshText);
      expect(state()!.outerHTML.replace(/title="[^"]*"/g, '')).toBe(freshMarkup);
      expect(stateText()!.getAttribute('title')).toContain('Measured before your later votes');
    });

    it.each(BALANCE_STATES)('never shows an estimate from the model (%s)', async (status) => {
      await show(1, lineBalance(status), 300);
      expect(stateText()!.textContent).not.toMatch(/estimated|F-?beta|F1/i);
    });

    it('describes the line it was cut at, not a pick still on its way to the server', async () => {
      await show(0.5, lineBalance('unchecked', { beta: 1, count: 40 }), 40);
      expect(checkedValue()).toBe('0.5');
      expect(stateText()!.textContent).toContain('Top 40 kept');
    });
  });

  describe('the check affordance (#4273)', () => {
    const checkBtn = () => (fixture.nativeElement as HTMLElement).querySelector('.balance-check-btn') as HTMLButtonElement | null;

    it('is absent with no line to check', async () => {
      await show(1, null);
      expect(checkBtn()).toBeNull();
    });

    it.each(BALANCE_STATES)('offers a check in the %s state, and runs it on click', async (status) => {
      await show(1, lineBalance(status), 32);
      const emitted = vi.spyOn(component.check, 'emit');
      expect(checkBtn()!.textContent!.trim()).toBe('Check 5 picks');
      checkBtn()!.click();
      expect(emitted).toHaveBeenCalledOnce();
    });

    it('reads the pick count off the schedule: 5 a band at the false-negatives end too (#4388)', async () => {
      await show(0.5, lineBalance('checked', { beta: 0.5, schedule: { candidate: 32, rounds: 3, picks: 5 } }));
      expect(checkBtn()!.textContent!.trim()).toBe('Check 5 picks');
    });

    it('is held while the host cannot run a check', async () => {
      fixture.componentRef.setInput('checkable', false);
      await show(1, lineBalance('unchecked'));
      expect(checkBtn()!.disabled).toBe(true);
    });

    it.each(BALANCE_STATES)('is absent where the host offers no check, Find (%s, #4317)', async (status) => {
      fixture.componentRef.setInput('offerCheck', false);
      await show(1, lineBalance(status), 32);
      expect(checkBtn()).toBeNull();
      expect(stateText()).not.toBeNull();
    });
  });

  describe('the "what does this mean" hint', () => {
    it('sits beside "Threshold", with or without a state line', async () => {
      for (const balance of [null, lineBalance('unchecked')]) {
        await show(1, balance, 10);
        expect(hints().length).toBe(1);
        expect(hints()[0].closest('.balance-head')).not.toBeNull();
      }
      expect(hints()[0].querySelector('.field-hint-icon')!.getAttribute('aria-label')).toBe('What the threshold means');
    });

    it('is one sentence pair, not a page (#4317)', () => {
      expect(component.hint.length).toBeLessThan(200);
      expect(component.hint.match(/\./g)!.length).toBeLessThanOrEqual(2);
    });

    it('opens below the control, so the left panel cannot clip it', () => {
      expect(hints()[0].querySelector('.field-hint-icon')!.classList).toContain('field-hint-icon--below-block');
    });
  });
});
