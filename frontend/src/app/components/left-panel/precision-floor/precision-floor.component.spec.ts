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

  async function show(value: number | null, floor: LineFloor | null = null, returned: number | null = null) {
    fixture.componentRef.setInput('value', value);
    fixture.componentRef.setInput('floor', floor);
    fixture.componentRef.setInput('returned', returned);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  const select = () => (fixture.nativeElement as HTMLElement).querySelector('select') as HTMLSelectElement;
  const optionLabels = () => Array.from(select().options).map((o) => o.textContent!.trim());
  const state = () => (fixture.nativeElement as HTMLElement).querySelector('.floor-state');
  const stateText = () => (fixture.nativeElement as HTMLElement).querySelector('.floor-state-text');
  const hints = () => (fixture.nativeElement as HTMLElement).querySelectorAll('vt-field-hint-icon');

  describe('the picker', () => {
    it('offers the four floors #4220 priced, starting at the 50% default', () => {
      expect(optionLabels()).toEqual(['25%', '50%', '75%', '90%']);
      expect(select().value).toBe('0.5');
    });

    it('is a select, never a range slider whose arrow keys would cast votes', () => {
      expect(select()).not.toBeNull();
      expect((fixture.nativeElement as HTMLElement).querySelector('input[type="range"]')).toBeNull();
    });

    it('reads as "at least X right"', async () => {
      const el = await show(0.75);
      expect(el.querySelector('.floor-picker')!.textContent!.replace(/\s+/g, ' ')).toContain('At least');
      expect(el.querySelector('.floor-picker')!.textContent).toContain('right');
      expect(select().value).toBe('0.75');
    });

    it('shows a stored floor that is not a preset as its own option, in order', async () => {
      await show(0.6);
      expect(optionLabels()).toEqual(['25%', '50%', '60%', '75%', '90%']);
      expect(select().value).toBe('0.6');
    });

    it('shows a detector with no floor as "No floor", which cannot be picked back', async () => {
      await show(null);
      expect(optionLabels()).toEqual(['No floor', '25%', '50%', '75%', '90%']);
      expect(select().value).toBe('none');
      expect(select().options[0].disabled).toBe(true);
    });

    it('follows the value through every transition, "No floor" included', async () => {
      for (const [value, shown] of [
        [0.9, '0.9'],
        [null, 'none'],
        [0.75, '0.75'],
        [0.6, '0.6'],
        [0.25, '0.25'],
        [null, 'none'],
        [0.5, '0.5'],
      ] as const) {
        await show(value);
        expect(select().value).toBe(shown);
      }
    });

    it('emits the picked floor as a fraction', () => {
      const emitted = vi.spyOn(component.valueChange, 'emit');
      select().value = '0.25';
      select().dispatchEvent(new Event('change'));
      expect(emitted).toHaveBeenCalledWith(0.25);
    });

    it('hands focus back after a pick, so the arrow keys vote again', () => {
      select().focus();
      expect(document.activeElement).toBe(select());
      select().value = '0.75';
      select().dispatchEvent(new Event('change'));
      expect(document.activeElement).not.toBe(select());
    });

    it('never emits "No floor"', async () => {
      await show(null);
      const emitted = vi.spyOn(component.valueChange, 'emit');
      select().value = 'none';
      select().dispatchEvent(new Event('change'));
      expect(emitted).not.toHaveBeenCalled();
    });
  });

  describe('the state line', () => {
    it('is absent with no line from the detector', async () => {
      await show(0.5, null);
      expect(state()).toBeNull();
    });

    it('says a promised floor holds, with the count', async () => {
      await show(0.75, lineFloor('promised', { minPrecision: 0.75 }), 1234);
      expect(state()!.getAttribute('data-status')).toBe('green');
      expect(stateText()!.textContent).toContain('At least 75% right · 1,234 returned');
      expect(stateText()!.getAttribute('title')).toContain('At least 75% of the 1,234 items the line returns');
    });

    it('omits the count when it is unknown', async () => {
      await show(0.5, lineFloor('promised'));
      expect(stateText()!.textContent!.trim()).toBe('At least 50% right');
    });

    it('says an unreachable floor shows the default cut', async () => {
      await show(0.9, lineFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('red');
      expect(stateText()!.textContent).toContain("Can't reach 90% on this dataset · showing the default cut");
      expect(stateText()!.getAttribute('title')).toContain('No cut reaches 90% precision');
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

    it('says a detector with no floor makes no promise', async () => {
      await show(null, lineFloor(null), 300);
      expect(state()!.getAttribute('data-status')).toBe('none');
      expect(stateText()!.textContent).toContain('No floor set');
    });

    it('describes the line it was cut at, not a pick still on its way to the server', async () => {
      await show(0.9, lineFloor('promised', { minPrecision: 0.5 }), 40);
      expect(select().value).toBe('0.9');
      expect(stateText()!.textContent).toContain('At least 50% right');
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
