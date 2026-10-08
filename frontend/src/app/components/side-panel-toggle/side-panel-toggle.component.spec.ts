import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SidePanelToggleComponent } from './side-panel-toggle.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('SidePanelToggleComponent (#4673)', () => {
  let fixture: ComponentFixture<SidePanelToggleComponent>;

  beforeEach(async () => {
    await configureZoneless({ imports: [SidePanelToggleComponent] }).compileComponents();
    fixture = TestBed.createComponent(SidePanelToggleComponent);
    fixture.componentRef.setInput('label', 'Labels');
  });

  async function render(side: 'left' | 'right', collapsed: boolean): Promise<HTMLElement> {
    fixture.componentRef.setInput('side', side);
    fixture.componentRef.setInput('collapsed', collapsed);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  it('folded, is one strip-wide button naming the panel, its arrow pointing into the view', async () => {
    const el = await render('right', true);
    const strip = el.querySelector('button.side-strip') as HTMLButtonElement;
    expect(strip).not.toBeNull();
    expect(strip.title).toBe('Show labels panel');
    expect(strip.textContent).toContain('◀');
    expect(strip.textContent).toContain('Labels');
    expect(el.classList.contains('folded')).toBe(true);

    expect((await render('left', true)).querySelector('.side-strip')!.textContent).toContain('▶');
  });

  it('open, is a small arrow pointing toward the edge it folds to', async () => {
    const right = await render('right', false);
    expect(right.querySelector('.side-strip')).toBeNull();
    const btn = right.querySelector('.side-bar button.collapse-toggle') as HTMLButtonElement;
    expect(btn.title).toBe('Hide labels panel');
    expect(btn.textContent).toContain('▶');
    expect(right.querySelector('.side-bar--right')).not.toBeNull();

    const left = await render('left', false);
    expect(left.querySelector('.side-bar button')!.textContent).toContain('◀');
    expect(left.querySelector('.side-bar--right')).toBeNull();
  });

  it('reports the click either way', async () => {
    const seen: number[] = [];
    fixture.componentInstance.toggle.subscribe(() => seen.push(1));

    (await render('right', true)).querySelector<HTMLButtonElement>('.side-strip')!.click();
    (await render('right', false)).querySelector<HTMLButtonElement>('.collapse-toggle')!.click();
    expect(seen.length).toBe(2);
  });
});
