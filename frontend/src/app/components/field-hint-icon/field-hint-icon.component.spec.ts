import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FieldHintIconComponent } from './field-hint-icon.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('FieldHintIconComponent', () => {
  let fixture: ComponentFixture<FieldHintIconComponent>;
  let component: FieldHintIconComponent;

  const iconSpan = (): HTMLElement =>
    fixture.nativeElement.querySelector('.field-hint-icon') as HTMLElement;
  const tooltip = (): HTMLElement | null =>
    fixture.nativeElement.querySelector('.field-hint-tooltip');

  beforeEach(async () => {
    await configureZoneless({ imports: [FieldHintIconComponent] }).compileComponents();
    fixture = TestBed.createComponent(FieldHintIconComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('hint', 'Explains the field');
    await settleZoneless(fixture);
  });

  it('creates', () => {
    expect(component).toBeTruthy();
    expect(tooltip()).toBeNull();
  });

  it('shows the tooltip immediately on click (no hover delay)', async () => {
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);

    expect(component.visible()).toBe(true);
    expect(tooltip()?.textContent).toContain('Explains the field');
  });

  it('toggles the tooltip off on a second click', async () => {
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(true);

    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(false);
    expect(tooltip()).toBeNull();
  });

  it('keeps a click-opened tooltip open when the pointer leaves the icon', async () => {
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(true);

    iconSpan().dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
    await settleZoneless(fixture);

    // A click pins the tooltip; moving the mouse off the icon must not close it.
    expect(component.visible()).toBe(true);
    expect(tooltip()).not.toBeNull();
  });

  it('closes a click-opened tooltip when clicking elsewhere', async () => {
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(true);

    document.body.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settleZoneless(fixture);

    expect(component.visible()).toBe(false);
    expect(tooltip()).toBeNull();
  });

  it('shows after a delay on hover, then hides on mouse leave', async () => {
    vi.useFakeTimers();
    try {
      iconSpan().dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
      // Not shown yet: the hover has a delay.
      expect(component.visible()).toBe(false);

      vi.advanceTimersByTime(600);
      expect(component.visible()).toBe(true);

      iconSpan().dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
      expect(component.visible()).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not leave the tooltip stuck open after a hover+click sequence', async () => {
    // Hover to open, then click, then click again to dismiss, then move away.
    iconSpan().dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(true);

    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(false);
  });

  it('closes the tooltip on Escape', async () => {
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    expect(component.visible()).toBe(true);

    iconSpan().dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }),
    );
    await settleZoneless(fixture);
    expect(component.visible()).toBe(false);
  });
});
