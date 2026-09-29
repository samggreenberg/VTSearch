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

    it('reads as "Lean: [name]", with no number anywhere on the control (#4298)', async () => {
      const el = await show(0.9);
      expect(el.querySelector('.floor-picker')!.textContent!.replace(/\s+/g, ' ')).toContain('Lean:');
      expect(el.querySelector('.precision-floor')!.textContent).not.toMatch(/\d/);
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

    it('says a promised floor holds, with the count', async () => {
      await show(0.9, lineFloor('promised', { minPrecision: 0.9 }), 1234);
      expect(state()!.getAttribute('data-status')).toBe('green');
      expect(stateText()!.textContent).toContain('Promise kept · 1,234 returned');
      expect(stateText()!.getAttribute('title')).toContain('keeps its Correct promise on the 1,234 items it returns');
    });

    it('omits the count when it is unknown', async () => {
      await show(0.5, lineFloor('promised'));
      expect(stateText()!.textContent!.trim()).toBe('Promise kept');
    });

    it('says an unreachable floor shows the default cut', async () => {
      await show(0.9, lineFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('red');
      expect(stateText()!.textContent).toContain("Can't reach Correct on this dataset · showing the default cut");
      expect(stateText()!.getAttribute('title')).toContain('No cut on this dataset reaches Correct');
    });

    it('says a floor short of evidence shows the default cut, with the Good votes it has', async () => {
      await show(0.5, lineFloor('insufficient_evidence'), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Not enough evidence yet (3 of 10 Good votes) · showing the default cut');
      expect(stateText()!.getAttribute('title')).toContain('Hard picks');
    });

    it.each(NO_PROMISE_STATES)('never shows the estimate behind the verdict (%s)', async (status) => {
      await show(0.5, lineFloor(status), 300);
      expect(stateText()!.textContent).not.toMatch(/about|estimated/i);
    });

    it('describes the line it was cut at, not a pick still on its way to the server', async () => {
      await show(0.9, lineFloor('unreachable', { minPrecision: 0.5 }), 40);
      expect(select().value).toBe('0.9');
      expect(stateText()!.textContent).toContain("Can't reach Centered");
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
      await show(0.5, lineFloor('insufficient_evidence'), 10);
      expect(hints().length).toBe(1);
      expect(hints()[0].closest('.floor-state')).not.toBeNull();
    });

    it('opens below the control, so the left panel cannot clip it', () => {
      expect(hints()[0].querySelector('.field-hint-icon')!.classList).toContain('field-hint-icon--below-block');
    });
  });
});
