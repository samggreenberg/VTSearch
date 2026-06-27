import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ContextMenuComponent } from './context-menu.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('ContextMenuComponent', () => {
  let fixture: ComponentFixture<ContextMenuComponent>;
  let component: ContextMenuComponent;
  let dismissCount: number;
  const outsideEls: HTMLElement[] = [];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ContextMenuComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(ContextMenuComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('x', 10);
    fixture.componentRef.setInput('y', 20);
    fixture.componentRef.setInput('items', [{ id: 'a', label: 'Action A' }]);
    // The capture-phase listeners are on `document`, so the menu host must be
    // connected for dispatched clicks to propagate through the document.
    document.body.appendChild(fixture.nativeElement);
    await settleZoneless(fixture);

    dismissCount = 0;
    component.dismissed.subscribe(() => (dismissCount += 1));
  });

  afterEach(() => {
    outsideEls.forEach((el) => el.remove());
    outsideEls.length = 0;
    fixture.destroy();
  });

  /** Append a detached button to the body and return it. */
  function makeOutsideButton(stopProp: boolean): HTMLElement {
    const btn = document.createElement('button');
    if (stopProp) {
      // Mirrors a dashboard card's ⋯ overflow button, which stops the click
      // from bubbling so the row isn't selected.
      btn.addEventListener('click', (e) => e.stopPropagation());
    }
    document.body.appendChild(btn);
    outsideEls.push(btn);
    return btn;
  }

  it('should dismiss on an outside click', () => {
    makeOutsideButton(false).click();
    expect(dismissCount).toBe(1);
  });

  it('should dismiss on an outside click even when the target stops propagation', () => {
    // Regression: clicking another card's ⋯ button (which calls
    // stopPropagation) used to leave this menu open because the bubble-phase
    // document listener never fired.
    makeOutsideButton(true).click();
    expect(dismissCount).toBe(1);
  });

  it('should dismiss on an outside right-click', () => {
    const btn = makeOutsideButton(false);
    btn.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(dismissCount).toBe(1);
  });

  it('should not dismiss when clicking inside the menu', () => {
    const item = fixture.nativeElement.querySelector('.menu-item') as HTMLElement;
    item.click();
    expect(dismissCount).toBe(0);
  });

  it('should dismiss on Escape', () => {
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(dismissCount).toBe(1);
  });

  it('should stop listening once destroyed', () => {
    fixture.destroy();
    makeOutsideButton(true).click();
    expect(dismissCount).toBe(0);
  });
});
