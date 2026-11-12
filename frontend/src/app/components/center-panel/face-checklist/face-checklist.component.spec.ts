import { ComponentFixture, TestBed } from '@angular/core/testing';

import { computed, signal } from '@angular/core';

import { FACE_CHECKS, FaceChecklistComponent } from './face-checklist.component';
import { FaceRect, FaceShape, FaceView, LinkLead, ViewLink, viewOfRect } from '../image-viewer/face-link';
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

  it('is a prompt, not a form: nothing to tick, and nothing to press while the panes are not linked', async () => {
    expect(fixture.nativeElement.querySelectorAll('input, button, select').length).toBe(0);
    // A link whose faces are not matched (no landmarks, or Wider) is the same.
    fixture.componentRef.setInput('link', new ViewLink());
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelectorAll('input, button, select').length).toBe(0);
    expect(rows().length).toBe(5);
  });

  describe('with the two panes linked (#5647)', () => {
    const HOME: FaceView = { cx: 0, cy: 0.5, scale: 20 };
    const ROOMY: FaceRect = { x0: -2.5, y0: -2.5, x1: 2.5, y1: 3 };
    /** A face looking at the camera, on an image with *bounds*. */
    const face = (bounds: FaceRect = ROOMY): FaceShape => ({
      points: [
        [-0.5, 0],
        [0.5, 0],
        [0, 0.55],
        [-0.4, 1],
        [0.4, 1],
      ],
      bounds,
    });

    const view = signal<FaceView>(HOME);
    const candidate = signal<FaceShape | null>(face());
    const reference = signal<FaceShape | null>(face());
    const linked = signal(true);
    let asked: FaceRect[];
    let homed: number;

    /** A stand-in for the candidate's viewer: a 100 px pane that shows what it is asked to. */
    const lead: LinkLead = {
      leadState: computed(() => ({ kind: 'on' as const, frame: { x: 0, y: 0, unit: 20 }, view: view() })),
      faceShape: candidate,
      shownZoom: signal(1),
      showFaceView: (v) => view.set(v),
      showFaceRect: (rect) => {
        asked.push(rect);
        view.set(viewOfRect(rect, 100, 100, 0)!);
      },
      homeView: () => {
        homed += 1;
        view.set(HOME);
      },
      canPan: () => false,
    };

    beforeEach(async () => {
      asked = [];
      homed = 0;
      view.set(HOME);
      candidate.set(face());
      reference.set(face());
      linked.set(true);
      const link = new ViewLink();
      link.attach(lead);
      link.attachFollower({ linked, faceShape: reference });
      fixture.componentRef.setInput('link', link);
      await settleZoneless(fixture);
    });

    function buttons(): HTMLButtonElement[] {
      return Array.from(fixture.nativeElement.querySelectorAll('button.checklist-zoom'));
    }

    async function press(part: string): Promise<void> {
      buttons()
        .find((b) => b.querySelector('.checklist-part')!.textContent!.trim() === part)!
        .click();
      await settleZoneless(fixture);
    }

    function pressed(): string[] {
      return buttons()
        .filter((b) => b.getAttribute('aria-pressed') === 'true')
        .map((b) => b.querySelector('.checklist-part')!.textContent!.trim());
    }

    it('makes every row a button, in the same order with the same words', () => {
      expect(buttons().length).toBe(5);
      expect(rows()).toEqual(['Marks', 'Ears', 'Nose', 'Eyes and brows', 'Mouth, chin, face outline']);
      expect(buttons().every((b) => !b.disabled && b.type === 'button')).toBe(true);
      expect(pressed()).toEqual([]);
    });

    it('zooms both panes to the part a row names, and back out on a second press', async () => {
      await press('Nose');
      expect(asked.length).toBe(1);
      // Round the nose tip, between the eyes and the mouth.
      expect(asked[0].x0).toBeLessThan(0);
      expect(asked[0].x1).toBeGreaterThan(0);
      expect(asked[0].y1).toBeLessThan(1);
      expect(pressed()).toEqual(['Nose']);
      await press('Nose');
      expect(asked.length).toBe(1);
      expect(homed).toBe(1);
      expect(pressed()).toEqual([]);
    });

    it('steps from one ear to the other, then back out', async () => {
      await press('Ears');
      await press('Ears');
      expect(asked.length).toBe(2);
      expect(asked[0].x1).toBeLessThan(0);
      expect(asked[1].x0).toBeGreaterThan(0);
      expect(pressed()).toEqual(['Ears']);
      await press('Ears');
      expect(homed).toBe(1);
      expect(pressed()).toEqual([]);
    });

    it('goes straight to another row\'s part, from whichever is in view', async () => {
      await press('Ears');
      await press('Eyes and brows');
      expect(homed).toBe(0);
      expect(pressed()).toEqual(['Eyes and brows']);
      // Back to the ears starts at the first again.
      await press('Ears');
      expect(asked[2]).toEqual(asked[0]);
    });

    it('forgets the row in view once the user zooms or pans off it', async () => {
      await press('Nose');
      view.update((v) => ({ ...v, cx: v.cx + 0.2 }));
      await settleZoneless(fixture);
      expect(pressed()).toEqual([]);
      // So the next press zooms to the nose again, not back out.
      await press('Nose');
      expect(asked.length).toBe(2);
      expect(homed).toBe(0);
    });

    it('greys a row whose part is off either image', async () => {
      // The reference's crop is cut close at the cheeks: no ears.
      reference.set(face({ x0: -1, y0: -2.5, x1: 1, y1: 3 }));
      await settleZoneless(fixture);
      const ears = buttons()[1];
      expect(ears.disabled).toBe(true);
      expect(ears.title).toBe('Not in view on both faces');
      expect(buttons().filter((b) => b.disabled).length).toBe(1);
      ears.click();
      expect(asked).toEqual([]);
    });

    it('turns back into plain text when the panes stop being linked', async () => {
      linked.set(false);
      await settleZoneless(fixture);
      expect(buttons().length).toBe(0);
      expect(rows().length).toBe(5);
    });
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
