import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PrecisionFloorComponent } from './precision-floor.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { FLOOR_STATES, lineFloor } from '../../../testing/line-floor';
import type { LineFloor } from '../../../utils/line-floor';

describe('PrecisionFloorComponent (#4246, #4317)', () => {
  let component: PrecisionFloorComponent;
  let fixture: ComponentFixture<PrecisionFloorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PrecisionFloorComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(PrecisionFloorComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  async function show(
    value: number | null,
    floor: LineFloor | null = null,
    returned: number | null = null,
    busy = false,
  ) {
    fixture.componentRef.setInput('value', value);
    fixture.componentRef.setInput('floor', floor);
    fixture.componentRef.setInput('returned', returned);
    fixture.componentRef.setInput('busy', busy);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  const root = () => fixture.nativeElement as HTMLElement;
  const radios = () => Array.from(root().querySelectorAll('input[type="radio"]')) as HTMLInputElement[];
  const checkedValue = () => radios().find((r) => r.checked)?.value ?? null;
  const state = () => root().querySelector('.floor-state');
  const stateText = () => root().querySelector('.floor-state-text');
  const hints = () => root().querySelectorAll('vt-field-hint-icon');

  describe('the spectrum (#4317)', () => {
    it('reads "Threshold", from False Positives to False Negatives', () => {
      expect(root().querySelector('.floor-label')!.textContent!.trim()).toBe('Threshold');
      const ends = Array.from(root().querySelectorAll('.spectrum-bar span')).map((e) => e.textContent!.trim());
      expect(ends).toEqual(['False Positives', 'False Negatives']);
      expect(root().textContent).not.toContain('Lean');
      expect(root().querySelector('select')).toBeNull();
    });

    it('offers three radios under it, left to right, starting at the middle default', () => {
      expect(radios().map((r) => r.value)).toEqual(['0.1', '0.5', '0.9']);
      expect(checkedValue()).toBe('0.5');
      // One grid column per third of the spectrum, each radio centred in its own.
      expect(root().querySelectorAll('.spectrum-radios > .spectrum-radio').length).toBe(3);
    });

    it('names no floor and numbers none: each radio says where it sits only in its tooltip', () => {
      expect(root().querySelector('.floor-spectrum')!.textContent!.replace(/\s+/g, '')).toBe('FalsePositivesFalseNegatives');
      expect(root().querySelector('.floor-head')!.textContent).not.toMatch(/\d/);
      const titles = Array.from(root().querySelectorAll('.spectrum-radio')).map((l) => l.getAttribute('title'));
      expect(titles[0]).toMatch(/^Toward false positives/);
      expect(titles[2]).toMatch(/^Toward false negatives/);
      expect(radios().map((r) => r.getAttribute('aria-label'))).toEqual(titles);
    });

    it('is one radio group, labelled by its heading', () => {
      const group = root().querySelector('[role="radiogroup"]')!;
      expect(root().querySelector(`#${group.getAttribute('aria-labelledby')}`)!.textContent!.trim()).toBe('Threshold');
      expect(new Set(radios().map((r) => r.name)).size).toBe(1);
    });

    it('is never a range slider, whose arrow keys would move it as they cast votes', () => {
      expect(root().querySelector('input[type="range"]')).toBeNull();
    });

    it('follows the value through every transition', async () => {
      for (const value of [0.9, 0.1, 0.5]) {
        await show(value);
        expect(checkedValue()).toBe(String(value));
      }
    });

    it('emits the picked floor as a fraction', () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      radios()[0].click();
      expect(emitted).toHaveBeenCalledExactlyOnceWith(0.1);
    });

    it('shows the floor the host holds, not the click: a dropped pick leaves it where it was', async () => {
      await show(0.5);
      radios()[2].click();
      await settleZoneless(fixture);
      // The host never took the pick (Find drops one mid-pass).
      expect(checkedValue()).toBe('0.5');
      await show(0.9);
      expect(checkedValue()).toBe('0.9');
    });

    it('does not re-emit the floor it already shows', () => {
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

  /** A stored floor off the list (#4298): an old 25% / 75% pick, or one set through the CLI or the API. */
  describe('a stored floor off the list', () => {
    it.each<[number, number]>([
      [0.25, 0.1],
      [0.75, 0.9],
      [0.6, 0.5],
    ])('shows %s on the nearest radio, %s, and snaps to it', async (stored, snapped) => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(stored);
      expect(checkedValue()).toBe(String(snapped));
      expect(emitted).toHaveBeenCalledExactlyOnceWith(snapped);
    });

    it('waits for a running sort before it snaps', async () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(0.25, null, null, true);
      expect(emitted).not.toHaveBeenCalled();
      expect(checkedValue()).toBe('0.1');

      fixture.componentRef.setInput('busy', false);
      await settleZoneless(fixture);
      expect(emitted).toHaveBeenCalledExactlyOnceWith(0.1);
    });

    it('leaves a preset alone', async () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      for (const value of [0.1, 0.5, 0.9]) await show(value);
      expect(emitted).not.toHaveBeenCalled();
    });
  });

  describe('the state line', () => {
    it('is absent with no line from the detector', async () => {
      await show(0.5, null);
      expect(state()).toBeNull();
    });

    it('says a confirmed floor holds, with the range the check found and the count kept', async () => {
      await show(0.9, lineFloor('confirmed', { minPrecision: 0.9 }), 1234);
      expect(state()!.getAttribute('data-status')).toBe('green');
      expect(stateText()!.textContent).toContain('Confirmed · likely 55–100% right (checked 5) · 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('5 random picks from the 32 items the line keeps');
    });

    it('reads the count off the result, never the preset', async () => {
      await show(0.1, lineFloor('confirmed', { minPrecision: 0.1, count: 64 }));
      expect(stateText()!.textContent).toContain('64 kept');
    });

    it('says how close a short check got, naming no cause, with the top 32 kept', async () => {
      await show(0.9, lineFloor('short', { minPrecision: 0.9 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('red');
      expect(stateText()!.textContent).toContain('Fell short · likely 11–73% right (checked 5) · top 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('short of the threshold');
      expect(stateText()!.getAttribute('title')).not.toMatch(/sparse|weak|evidence|too few/i);
    });

    it('says an unchecked line keeps the starting candidate', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Top 128 kept, unchecked');
      expect(stateText()!.getAttribute('title')).toContain('nothing has measured how much of it is right');
      expect(stateText()!.textContent).not.toMatch(/likely|%\s*right/);
    });

    it('prices a check off the schedule, rounds and all, on the check button', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }));
      expect(root().querySelector('.floor-check-btn')!.getAttribute('title')).toContain('5 random picks a round, in up to 3 rounds');
    });

    it.each(['short', 'confirmed'] as const)('notes a stale %s range only in the tooltip', async (status) => {
      const fresh = lineFloor(status);
      await show(0.5, fresh, 300);
      const freshText = stateText()!.textContent;
      const freshMarkup = state()!.outerHTML.replace(/title="[^"]*"/g, '');
      await show(0.5, lineFloor(status, { range: { ...fresh.range!, stale: true } }), 300);
      expect(stateText()!.textContent).toBe(freshText);
      expect(state()!.outerHTML.replace(/title="[^"]*"/g, '')).toBe(freshMarkup);
      expect(stateText()!.getAttribute('title')).toContain('Measured before your later votes');
    });

    it.each(FLOOR_STATES)('never shows an estimate from the model (%s)', async (status) => {
      await show(0.5, lineFloor(status), 300);
      expect(stateText()!.textContent).not.toMatch(/about|estimated/i);
    });

    it('describes the line it was cut at, not a pick still on its way to the server', async () => {
      await show(0.9, lineFloor('short', { minPrecision: 0.5, count: 40 }), 40);
      expect(checkedValue()).toBe('0.9');
      expect(stateText()!.textContent).toContain('top 40 kept');
    });
  });

  describe('the check affordance (#4273)', () => {
    const checkBtn = () => (fixture.nativeElement as HTMLElement).querySelector('.floor-check-btn') as HTMLButtonElement | null;

    it('is absent with no line to check', async () => {
      await show(0.5, null);
      expect(checkBtn()).toBeNull();
    });

    it.each(FLOOR_STATES)('offers a check in the %s state, and runs it on click', async (status) => {
      await show(0.5, lineFloor(status), 32);
      const emitted = vi.spyOn(component.check, 'emit');
      expect(checkBtn()!.textContent!.trim()).toBe('Check 5 picks');
      checkBtn()!.click();
      expect(emitted).toHaveBeenCalledOnce();
    });

    it('reads the pick count off the schedule: 29 at the false-negatives end', async () => {
      await show(0.9, lineFloor('confirmed', { minPrecision: 0.9, schedule: { candidate: 32, rounds: 1, picks: 29 } }));
      expect(checkBtn()!.textContent!.trim()).toBe('Check 29 picks');
    });

    it('is held while the host cannot run a check', async () => {
      fixture.componentRef.setInput('checkable', false);
      await show(0.5, lineFloor('unchecked'));
      expect(checkBtn()!.disabled).toBe(true);
    });

    it.each(FLOOR_STATES)('is absent where the host offers no check, Find (%s, #4317)', async (status) => {
      fixture.componentRef.setInput('offerCheck', false);
      await show(0.5, lineFloor(status), 32);
      expect(checkBtn()).toBeNull();
      expect(stateText()).not.toBeNull();
    });
  });

  describe('the "what does this mean" hint', () => {
    it('sits beside "Threshold", with or without a state line', async () => {
      for (const floor of [null, lineFloor('unchecked')]) {
        await show(0.5, floor, 10);
        expect(hints().length).toBe(1);
        expect(hints()[0].closest('.floor-head')).not.toBeNull();
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
