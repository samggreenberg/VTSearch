import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PrecisionFloorComponent } from './precision-floor.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { FLOOR_STATES, lineFloor } from '../../../testing/line-floor';
import type { LineFloor } from '../../../utils/line-floor';

describe('PrecisionFloorComponent (#4246)', () => {
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

  const select = () => (fixture.nativeElement as HTMLElement).querySelector('select') as HTMLSelectElement;
  const optionLabels = () => Array.from(select().options).map((o) => o.textContent!.trim());
  const state = () => (fixture.nativeElement as HTMLElement).querySelector('.floor-state');
  const stateText = () => (fixture.nativeElement as HTMLElement).querySelector('.floor-state-text');
  const hints = () => (fixture.nativeElement as HTMLElement).querySelectorAll('vt-field-hint-icon');

  describe('the picker', () => {
    it('offers the three named floors, starting at the Centered default (#4298)', () => {
      expect(optionLabels()).toEqual(['Complete', 'Centered', 'Correct']);
      expect(select().value).toBe('0.5');
    });

    it('is a select, never a range slider whose arrow keys would cast votes', () => {
      expect(select()).not.toBeNull();
      expect((fixture.nativeElement as HTMLElement).querySelector('input[type="range"]')).toBeNull();
    });

    it('reads as "Lean: [name]", with no number on the picker (#4298)', async () => {
      const el = await show(0.9);
      expect(el.querySelector('.floor-picker')!.textContent!.replace(/\s+/g, ' ')).toContain('Lean:');
      expect(el.querySelector('.floor-picker')!.textContent).not.toMatch(/\d/);
      expect(select().value).toBe('0.9');
      expect(select().selectedOptions[0].textContent!.trim()).toBe('Correct');
    });

    it('follows the value through every transition', async () => {
      for (const [value, shown] of [
        [0.9, '0.9'],
        [0.1, '0.1'],
        [0.5, '0.5'],
      ] as const) {
        await show(value);
        expect(select().value).toBe(shown);
      }
    });

    it('emits the picked floor as a fraction', () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      select().value = '0.1';
      select().dispatchEvent(new Event('change'));
      expect(emitted).toHaveBeenCalledWith(0.1);
    });

    it('hands focus back after a pick, so the arrow keys vote again', () => {
      select().focus();
      expect(document.activeElement).toBe(select());
      select().value = '0.9';
      select().dispatchEvent(new Event('change'));
      expect(document.activeElement).not.toBe(select());
    });
  });

  /** A stored floor off the list (#4298): an old 25% / 75% pick, or one set through the CLI or the API. */
  describe('a stored floor off the list', () => {
    it.each<[number, string, number]>([
      [0.25, 'Complete', 0.1],
      [0.75, 'Correct', 0.9],
      [0.6, 'Centered', 0.5],
    ])('shows %s as the nearest preset, %s, and snaps to it', async (stored, name, snapped) => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(stored);
      expect(optionLabels()).toEqual(['Complete', 'Centered', 'Correct']);
      expect(select().selectedOptions[0].textContent!.trim()).toBe(name);
      expect(emitted).toHaveBeenCalledExactlyOnceWith(snapped);
    });

    it('waits for a running sort before it snaps', async () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      await show(0.25, null, null, true);
      expect(emitted).not.toHaveBeenCalled();
      expect(select().selectedOptions[0].textContent!.trim()).toBe('Complete');

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
      expect(stateText()!.textContent).toContain('Aimed at Correct: likely 11–73% right (checked 5) · top 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('short of Correct');
      expect(stateText()!.getAttribute('title')).not.toMatch(/sparse|weak|evidence|too few/i);
    });

    it('says an unchecked line keeps the starting candidate', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Top 128 kept, unchecked · aiming at Complete');
      expect(stateText()!.getAttribute('title')).toContain('nothing has measured how much of it is right');
      expect(stateText()!.textContent).not.toMatch(/likely|%\s*right/);
    });

    it('prices an unchecked check off the schedule, rounds and all', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }));
      expect(stateText()!.getAttribute('title')).toContain('5 random picks a round, in up to 3 rounds');
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
      await show(0.9, lineFloor('short', { minPrecision: 0.5 }), 40);
      expect(select().value).toBe('0.9');
      expect(stateText()!.textContent).toContain('Aimed at Centered');
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

    it('reads the pick count off the schedule: 29 at Correct', async () => {
      await show(0.9, lineFloor('confirmed', { minPrecision: 0.9, schedule: { candidate: 32, rounds: 1, picks: 29 } }));
      expect(checkBtn()!.textContent!.trim()).toBe('Check 29 picks');
    });

    it('is held while the host cannot run a check', async () => {
      fixture.componentRef.setInput('checkable', false);
      await show(0.5, lineFloor('unchecked'));
      expect(checkBtn()!.disabled).toBe(true);
    });
  });

  describe('the "what does this mean" hint', () => {
    it('sits on the picker line while there is no state line', () => {
      expect(hints().length).toBe(1);
      expect(hints()[0].closest('.floor-picker')).not.toBeNull();
      expect(hints()[0].querySelector('.field-hint-icon')!.getAttribute('aria-label')).toBe(
        'What the precision floor means',
      );
    });

    it('moves to the end of the state line, leaving the picker line to projected controls', async () => {
      await show(0.5, lineFloor('unchecked'), 10);
      expect(hints().length).toBe(1);
      expect(hints()[0].closest('.floor-state')).not.toBeNull();
    });

    it('opens below the control, so the left panel cannot clip it', () => {
      expect(hints()[0].querySelector('.field-hint-icon')!.classList).toContain('field-hint-icon--below-block');
    });
  });
});
