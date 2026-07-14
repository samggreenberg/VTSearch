import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PanelResizeDirective } from './panel-resize.directive';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Coverage for the vertical-divider drag directive used by `vt-label-view`.
 * The width math (`computeWidth`) and the mousedown→move→up lifecycle are the
 * whole surface; jsdom does no layout, so the layout container's bounding rect
 * is stubbed to a fixed 1000px-wide box and the drag is driven with synthetic
 * mouse events on the divider and `document`.
 */
@Component({
  standalone: true,
  imports: [PanelResizeDirective],
  template: `
    <div #layout class="layout">
      <div
        class="divider"
        [vtPanelResize]="side()"
        [layoutEl]="layout"
        [minWidth]="minWidth"
        [opposingWidth]="opposingWidth()"
        [centerMin]="centerMin"
        [dividerTotal]="dividerTotal"
        (widthChange)="widths.push($event)"
        (resizeEnd)="ends.push($event)"
      ></div>
    </div>
  `,
})
class HostComponent {
  // Signals for the fields a test mutates mid-run: a plain-field write would not
  // schedule change detection under zoneless, so the bound input would go stale.
  side = signal<'left' | 'right'>('left');
  opposingWidth = signal(0);
  minWidth = 100;
  centerMin = 100;
  dividerTotal = 16;
  widths: number[] = [];
  ends: number[] = [];
}

// A 1000px-wide layout anchored at x=0: raw left width is `clientX`, raw right
// width is `1000 - clientX`. With the default clamps below, the usable range is
// [100, 1000 - 16 - 100 - 0] = [100, 884].
const RECT = {
  width: 1000,
  left: 0,
  right: 1000,
  top: 0,
  bottom: 0,
  x: 0,
  y: 0,
  toJSON: () => ({}),
} as DOMRect;

describe('PanelResizeDirective', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;
  let divider: HTMLElement;

  beforeEach(async () => {
    await configureZoneless({ imports: [HostComponent] }).compileComponents();
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    await settleZoneless(fixture);
    const layout = fixture.nativeElement.querySelector('.layout') as HTMLElement;
    layout.getBoundingClientRect = () => RECT;
    divider = fixture.nativeElement.querySelector('.divider') as HTMLElement;
  });

  afterEach(() => fixture.destroy());

  function mouseDown(clientX: number): void {
    divider.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX }));
  }
  function docMove(clientX: number): void {
    document.dispatchEvent(new MouseEvent('mousemove', { clientX }));
  }
  function docUp(): void {
    document.dispatchEvent(new MouseEvent('mouseup', {}));
  }

  it('mousedown preventDefault-s, flags dragging, and seeds the width', async () => {
    const event = new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX: 300 });
    divider.dispatchEvent(event);
    await settleZoneless(fixture);
    expect(event.defaultPrevented).toBe(true);
    expect(divider.classList.contains('dragging')).toBe(true);
  });

  it('a click with no intervening move emits the seeded width on resizeEnd (no jump to min)', async () => {
    mouseDown(300);
    docUp();
    await settleZoneless(fixture);
    expect(host.widths).toEqual([]); // no move → no widthChange
    expect(host.ends).toEqual([300]); // seeded from mousedown, not 0/min
    expect(divider.classList.contains('dragging')).toBe(false);
  });

  it('emits the clamped width on every move for the left side', async () => {
    mouseDown(200);
    docMove(300); // inside range
    docMove(50); // below minWidth → clamped up to 100
    docMove(950); // above max (884) → clamped down to 884
    await settleZoneless(fixture);
    expect(host.widths).toEqual([300, 100, 884]);
    docUp();
    await settleZoneless(fixture);
    expect(host.ends).toEqual([884]); // final width from the last move
  });

  it('measures width from the right edge when bound to the right side', async () => {
    host.side.set('right');
    await settleZoneless(fixture);
    mouseDown(300); // raw = 1000 - 300 = 700
    docMove(250); // raw = 750
    await settleZoneless(fixture);
    expect(host.widths).toEqual([750]);
    docUp();
    await settleZoneless(fixture);
    expect(host.ends).toEqual([750]); // resizeEnd carries the last move, not the seed
  });

  it('honors the opposingWidth clamp shrinking the available maximum', async () => {
    host.opposingWidth.set(400); // max = 1000 - 16 - 100 - 400 = 484
    await settleZoneless(fixture);
    mouseDown(200);
    docMove(950); // raw 950 clamped to 484
    await settleZoneless(fixture);
    expect(host.widths).toEqual([484]);
  });

  it('stops emitting after mouseup (drag listeners detached)', async () => {
    mouseDown(200);
    docMove(300);
    docUp();
    await settleZoneless(fixture);
    host.widths.length = 0;
    docMove(400); // no active drag / listener removed
    await settleZoneless(fixture);
    expect(host.widths).toEqual([]);
  });

  it('detaches document listeners on destroy (no emit for a stray move)', async () => {
    mouseDown(200);
    await settleZoneless(fixture);
    host.widths.length = 0;
    fixture.destroy();
    docMove(500);
    expect(host.widths).toEqual([]);
  });
});
