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
    it('offers the five preset floors, starting at the 50% default', () => {
      expect(optionLabels()).toEqual(['10%', '25%', '50%', '75%', '90%']);
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
      expect(optionLabels()).toEqual(['10%', '25%', '50%', '60%', '75%', '90%']);
      expect(select().value).toBe('0.6');
    });

    it('orders a stored floor below every preset first', async () => {
      await show(0.05);
      expect(optionLabels()).toEqual(['5%', '10%', '25%', '50%', '75%', '90%']);
      expect(select().value).toBe('0.05');
    });

    it('follows the value through every transition', async () => {
      for (const [value, shown] of [
        [0.9, '0.9'],
        [0.75, '0.75'],
        [0.6, '0.6'],
        [0.25, '0.25'],
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
  });

  describe('the state line', () => {
    it('is absent with no line from the detector', async () => {
      await show(0.5, null);
      expect(state()).toBeNull();
    });

    it('says a confirmed floor holds, with the range the check found and the count kept', async () => {
      await show(0.75, lineFloor('confirmed', { minPrecision: 0.75 }), 1234);
      expect(state()!.getAttribute('data-status')).toBe('green');
      expect(stateText()!.textContent).toContain('At least 75% right · likely 55–100% (checked 5) · 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('5 random picks from the 32 items the line keeps');
    });

    it('reads the count off the result, never the preset', async () => {
      await show(0.1, lineFloor('confirmed', { minPrecision: 0.1, count: 64 }));
      expect(stateText()!.textContent).toContain('64 kept');
    });

    it('says how close a short check got, naming no cause, with the top 32 kept', async () => {
      await show(0.9, lineFloor('short', { minPrecision: 0.9 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('red');
      expect(stateText()!.textContent).toContain('Aimed at 90%: likely 11–73% right (checked 5) · top 32 kept');
      expect(stateText()!.getAttribute('title')).toContain('fell short');
      expect(stateText()!.getAttribute('title')).not.toMatch(/sparse|weak|evidence/);
    });

    it('says an unchecked line keeps the starting candidate', async () => {
      await show(0.1, lineFloor('unchecked', { minPrecision: 0.1, count: 128 }), 300);
      expect(state()!.getAttribute('data-status')).toBe('yellow');
      expect(stateText()!.textContent).toContain('Top 128 kept, unchecked · aiming at 10%');
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
      await show(0.9, lineFloor('confirmed', { minPrecision: 0.5 }), 40);
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
      await show(0.5, lineFloor('unchecked'), 10);
      expect(hints().length).toBe(1);
      expect(hints()[0].closest('.floor-state')).not.toBeNull();
    });

    it('opens below the control, so the left panel cannot clip it', () => {
      expect(hints()[0].querySelector('.field-hint-icon')!.classList).toContain('field-hint-icon--below-block');
    });
  });
});
