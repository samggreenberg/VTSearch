import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PrecisionFloorComponent } from './precision-floor.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { NO_PROMISE_STATES, lineFloor } from '../../../testing/line-floor';
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
      expect(stateText()!.getAttribute('title')).toContain('fell short');
      expect(stateText()!.getAttribute('title')).not.toMatch(/sparse|weak|evidence/);
    });

    it('says an unchecked line keeps the starting candidate', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Top 128 kept, unchecked · aiming at Complete');
      expect(stateText()!.getAttribute('title')).toContain('Run a check');
    });

    it('notes a stale range only in the tooltip', async () => {
      const stale = { lo: 0.11, hi: 0.73, labelled: 5, right: 2, stale: true };
      await show(0.5, lineFloor('short', { range: stale }), 300);
      expect(stateText()!.textContent).toContain('likely 11–73% right (checked 5)');
      expect(stateText()!.textContent).not.toContain('later votes');
      expect(stateText()!.getAttribute('title')).toContain('Measured before your later votes');
    });

    it.each(NO_PROMISE_STATES)('never shows an estimate from the model (%s)', async (status) => {
      await show(0.5, lineFloor(status), 300);
      expect(stateText()!.textContent).not.toMatch(/about|estimated/i);
    });

    it('describes the line it was cut at, not a pick still on its way to the server', async () => {
      await show(0.9, lineFloor('short', { minPrecision: 0.5 }), 40);
      expect(select().value).toBe('0.9');
      expect(stateText()!.textContent).toContain('Aimed at Centered');
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
