import { Component, ChangeDetectionStrategy } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { PointerArrowComponent, RectLike, pointerArrowGeometry } from './pointer-arrow.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

function rect(left: number, top: number, width: number, height: number): RectLike {
  return { left, top, width, height, right: left + width, bottom: top + height };
}

describe('pointerArrowGeometry', () => {
  const origin = { left: 0, top: 0 };

  it('bends from the source side up into a target above and to the right', () => {
    // Empty-state message in the middle of a panel, `+` button top-right.
    const g = pointerArrowGeometry(rect(400, 300, 200, 20), rect(900, 20, 30, 30), origin);
    // Leaves the message's right edge (+8px gap) at its vertical middle, and
    // arrives 8px under the button's bottom-centre.
    expect(g?.shaft).toBe('M 608 310 Q 915 310 915 58');
    // An upward chevron: both wings sit below the tip.
    expect(g?.head).toBe('M 910 66.7 L 915 58 L 920 66.7');
  });

  it('leaves from the left side when the target is off to the left', () => {
    const g = pointerArrowGeometry(rect(400, 300, 200, 20), rect(100, 20, 30, 30), origin);
    expect(g?.shaft.startsWith('M 392 310 Q 115 310 115 58')).toBe(true);
  });

  it('drops out of the bottom as an S when the target is below and overlapping', () => {
    // Train hint centred above the Train button.
    const g = pointerArrowGeometry(rect(400, 500, 200, 20), rect(380, 600, 120, 40), origin);
    expect(g?.shaft).toBe('M 500 528 C 500 560 440 560 440 592');
    // A downward chevron: both wings sit above the tip.
    expect(g?.head).toBe('M 435 583.3 L 440 592 L 445 583.3');
  });

  it('translates into the host coordinate space', () => {
    const g = pointerArrowGeometry(rect(400, 300, 200, 20), rect(900, 20, 30, 30), { left: 100, top: 10 });
    expect(g?.shaft).toBe('M 508 300 Q 815 300 815 48');
  });

  it('draws nothing for an unrendered endpoint or vertically overlapping boxes', () => {
    expect(pointerArrowGeometry(rect(0, 0, 0, 0), rect(900, 20, 30, 30), origin)).toBeNull();
    expect(pointerArrowGeometry(rect(400, 300, 200, 20), rect(0, 0, 0, 0), origin)).toBeNull();
    expect(pointerArrowGeometry(rect(400, 300, 200, 20), rect(700, 305, 30, 10), origin)).toBeNull();
  });
});

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  imports: [PointerArrowComponent],
  template: `
    <div class="box">
      <button #target>+</button>
      <p #source>No datasets yet.</p>
      <vt-pointer-arrow [from]="source" [to]="target" />
    </div>
  `,
})
class HostComponent {}

describe('PointerArrowComponent', () => {
  // The prototype stub below would otherwise leak into later spec files
  // (the suite runs with `isolate: false`).
  afterEach(() => vi.restoreAllMocks());

  it('measures its endpoints after render and draws the arrow', async () => {
    TestBed.configureTestingModule({ imports: [HostComponent], providers: [...provideZoneless()] });
    // jsdom does no layout, so hand each element the box a browser would.
    const boxes: Record<string, RectLike> = {
      BUTTON: rect(900, 20, 30, 30),
      P: rect(400, 300, 200, 20),
      'VT-POINTER-ARROW': rect(0, 0, 1000, 700),
    };
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      return (boxes[this.tagName] ?? rect(0, 0, 0, 0)) as DOMRect;
    });
    const fixture = TestBed.createComponent(HostComponent);
    const el = fixture.nativeElement as HTMLElement;
    await settleZoneless(fixture);

    const host = el.querySelector('vt-pointer-arrow') as HTMLElement;
    expect(host.getAttribute('aria-hidden')).toBe('true');
    expect(host.querySelector('.pointer-arrow__shaft')?.getAttribute('d')).toBe('M 608 310 Q 915 310 915 58');
    expect(host.querySelector('.pointer-arrow__head')?.getAttribute('d')).toBe('M 910 66.7 L 915 58 L 920 66.7');
  });
});
