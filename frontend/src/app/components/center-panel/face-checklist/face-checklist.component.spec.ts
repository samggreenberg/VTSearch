import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FACE_CHECKS, FaceChecklistComponent } from './face-checklist.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/**
 * The checklist strip of the Face compare layout (#5643): five things to
 * compare in diagnostic order, what not to weigh, and a fitness notice when
 * either face is a hard view.
 */
describe('FaceChecklistComponent', () => {
  let fixture: ComponentFixture<FaceChecklistComponent>;

  beforeEach(async () => {
    configureZoneless({ imports: [FaceChecklistComponent] });
    fixture = TestBed.createComponent(FaceChecklistComponent);
    await settleZoneless(fixture);
  });

  afterEach(() => fixture.destroy());

  async function verdicts(candidate: boolean | null, reference: boolean | null): Promise<void> {
    fixture.componentRef.setInput('candidateHardView', candidate);
    fixture.componentRef.setInput('referenceHardView', reference);
    await settleZoneless(fixture);
  }

  function rows(): string[] {
    const items = Array.from(fixture.nativeElement.querySelectorAll('.checklist-part')) as HTMLElement[];
    return items.map((el) => el.textContent!.trim());
  }

  function notice(): string | null {
    const el = fixture.nativeElement.querySelector('.checklist-fitness') as HTMLElement | null;
    return el ? el.textContent!.trim() : null;
  }

  it('lists the five parts most telling first: marks and ears, down to mouth and outline', () => {
    expect(rows()).toEqual(['Marks', 'Ears', 'Nose', 'Eyes and brows', 'Mouth, chin, face outline']);
    expect(FACE_CHECKS.length).toBe(5);
    // A numbered list, so the order reads as a ranking.
    expect(fixture.nativeElement.querySelector('ol.checklist-rows')).not.toBeNull();
  });

  it('names what not to weigh', () => {
    expect(fixture.nativeElement.querySelector('.checklist-ignore')!.textContent).toContain(
      "Don't weigh hair, beard, makeup or skin tone.",
    );
  });

  it('is a prompt, not a form: nothing to tick and nothing to press', () => {
    expect(fixture.nativeElement.querySelectorAll('input, button, select').length).toBe(0);
  });

  it('has no notice while neither face is a hard view, or there is no verdict', async () => {
    expect(notice()).toBeNull();
    await verdicts(false, false);
    expect(notice()).toBeNull();
    await verdicts(null, false);
    expect(notice()).toBeNull();
  });

  it('says which face is a hard view, in one line, and offers no way past the vote', async () => {
    await verdicts(true, null);
    expect(notice()).toBe('This face is a hard view (turned, covered or small): expect a weaker comparison.');
    await verdicts(false, true);
    expect(notice()).toBe('The reference is a hard view (turned, covered or small): expect a weaker comparison.');
    await verdicts(true, true);
    expect(notice()).toBe('Both faces are hard views (turned, covered or small): expect a weaker comparison.');
    expect(notice()).not.toMatch(/skip|move on|next/i);
  });
});
