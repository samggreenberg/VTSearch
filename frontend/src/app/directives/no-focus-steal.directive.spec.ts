import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoFocusStealDirective } from './no-focus-steal.directive';
import { configureZoneless } from '../testing/zoneless-testbed';
import { settleZoneless } from '../testing/settle-resource';

@Component({
  standalone: true,
  imports: [NoFocusStealDirective],
  template: `
    <div vtNoFocusSteal>
      <button type="button" class="host-btn">Re-project</button>
      <div role="button" tabindex="0" class="role-btn">Entry</div>
      <input class="host-input" />
    </div>
  `,
})
class HostComponent {}

describe('NoFocusStealDirective', () => {
  let fixture: ComponentFixture<HostComponent>;

  beforeEach(async () => {
    await configureZoneless({ imports: [HostComponent] }).compileComponents();
    fixture = TestBed.createComponent(HostComponent);
    await settleZoneless(fixture);
  });

  /** Dispatch a cancelable mousedown and report whether default was prevented. */
  function pressMouseDown(el: Element): boolean {
    const event = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
    el.dispatchEvent(event);
    return event.defaultPrevented;
  }

  it('prevents default on mousedown over a real <button> (blocks focus steal)', () => {
    const button = fixture.nativeElement.querySelector('.host-btn') as HTMLButtonElement;
    expect(pressMouseDown(button)).toBe(true);
  });

  it('leaves the mousedown default intact for role="button" elements that need focus', () => {
    const roleBtn = fixture.nativeElement.querySelector('.role-btn') as HTMLElement;
    expect(pressMouseDown(roleBtn)).toBe(false);
  });

  it('leaves other focusable controls (inputs) untouched', () => {
    const input = fixture.nativeElement.querySelector('.host-input') as HTMLInputElement;
    expect(pressMouseDown(input)).toBe(false);
  });

  it('keeps the click on a guarded button working (focus stays off it)', () => {
    const button = fixture.nativeElement.querySelector('.host-btn') as HTMLButtonElement;
    let clicked = false;
    button.addEventListener('click', () => (clicked = true));
    // A real click issues mousedown (prevented) then a click (not prevented).
    pressMouseDown(button);
    button.click();
    expect(clicked).toBe(true);
    expect(document.activeElement).not.toBe(button);
  });
});
