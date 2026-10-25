import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FieldHintIconComponent, fieldHintPlacement } from './field-hint-icon.component';
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

  // Specs here spy on `HTMLElement.prototype`; spec files share one module
  // graph (`isolate: false`), so the spies must not outlive the test.
  afterEach(() => vi.restoreAllMocks());

  it('creates', () => {
    expect(component).toBeTruthy();
    expect(tooltip()).toBeNull();
  });

  it('fixes the tooltip to the window, kept inside its left edge (#5226)', async () => {
    // An icon 20px from the window's left edge, as the Add Dataset dialog's
    // Multi-Dataset hint is: centred on it, the tooltip would start left of 0.
    vi.spyOn(iconSpan(), 'getBoundingClientRect').mockReturnValue(
      DOMRect.fromRect({ x: 20, y: 300, width: 13, height: 13 }),
    );
    // jsdom lays nothing out; give the tooltip its real width.
    vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(260);
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);

    const tip = tooltip()!;
    expect(tip.getAttribute('popover')).toBe('manual');
    expect(tip.style.left).toBe('8px');
    expect(tip.style.top).toBe('296px');
  });

  it('follows the icon when something under it scrolls', async () => {
    const rect = vi.spyOn(iconSpan(), 'getBoundingClientRect');
    rect.mockReturnValue(DOMRect.fromRect({ x: 400, y: 300, width: 13, height: 13 }));
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);
    const before = tooltip()!.style.top;

    rect.mockReturnValue(DOMRect.fromRect({ x: 400, y: 200, width: 13, height: 13 }));
    document.body.dispatchEvent(new Event('scroll'));

    expect(tooltip()!.style.top).not.toBe(before);
  });

  it('takes the enclosing block\'s width for below-block', async () => {
    fixture.componentRef.setInput('placement', 'below-block');
    await settleZoneless(fixture);
    // jsdom has no offsetParent, so the host's parent stands in for the block.
    const block = fixture.nativeElement.parentElement as HTMLElement;
    vi.spyOn(block, 'getBoundingClientRect').mockReturnValue(
      DOMRect.fromRect({ x: 0, y: 100, width: 240, height: 40 }),
    );
    iconSpan().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settleZoneless(fixture);

    expect(tooltip()!.style.width).toBe('240px');
    expect(tooltip()!.style.top).toBe('144px');
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

describe('fieldHintPlacement', () => {
  const viewport = { width: 1000, height: 800 };
  const tip = { width: 260, height: 60 };
  const icon = (x: number, y: number) => DOMRect.fromRect({ x, y, width: 13, height: 13 });

  it('centres the tooltip above the icon when it fits', () => {
    expect(fieldHintPlacement(icon(500, 400), tip, viewport, 'above')).toEqual({
      left: 500 + 6.5 - 130,
      top: 400 - 4 - 60,
      side: 'above',
    });
  });

  it('slides right rather than run off the left edge (#5226)', () => {
    expect(fieldHintPlacement(icon(20, 400), tip, viewport, 'above').left).toBe(8);
  });

  it('slides left rather than run off the right edge', () => {
    expect(fieldHintPlacement(icon(980, 400), tip, viewport, 'above').left).toBe(1000 - 8 - 260);
  });

  it('flips below when there is no room above', () => {
    const p = fieldHintPlacement(icon(500, 20), tip, viewport, 'above');
    expect(p.side).toBe('below');
    expect(p.top).toBe(20 + 13 + 4);
  });

  it('flips above when there is no room below', () => {
    const p = fieldHintPlacement(icon(500, 760), tip, viewport, 'below');
    expect(p.side).toBe('above');
    expect(p.top).toBe(760 - 4 - 60);
  });

  it('keeps the preferred side, clamped, when neither side fits', () => {
    const tall = { width: 260, height: 790 };
    expect(fieldHintPlacement(icon(500, 400), tall, viewport, 'above')).toEqual({
      left: 500 + 6.5 - 130,
      top: 8,
      side: 'above',
    });
  });
});
