import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ClipboardCopyComponent } from './clipboard-copy.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('ClipboardCopyComponent', () => {
  let fixture: ComponentFixture<ClipboardCopyComponent>;

  beforeEach(async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: () => Promise.resolve() },
      configurable: true,
    });

    await configureZoneless({
      imports: [ClipboardCopyComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ClipboardCopyComponent);
    fixture.componentRef.setInput('columns', [{ key: 'a', label: 'A' }]);
    fixture.componentRef.setInput('rows', [{ a: '1' }]);
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('shows the copy label by default', () => {
    const btn = fixture.nativeElement.querySelector('button.btn') as HTMLButtonElement;
    expect(btn.textContent?.trim()).toBe('Copy');
    expect(btn.disabled).toBe(false);
  });

  // #5103: each dropdown carries a visible label tied to it by `for`/`id`,
  // rather than naming itself only through a hover tooltip.
  function labelsFor(el: HTMLElement): string[] {
    return Array.from(el.querySelectorAll('select')).map((select) => {
      const label = el.querySelector(`label[for="${select.id}"]`);
      return label?.textContent?.trim() ?? '';
    });
  }

  it('labels the delimiter dropdown in table mode', () => {
    expect(labelsFor(fixture.nativeElement)).toEqual(['Delimiter:']);
  });

  it('labels the column and separator dropdowns in list mode', async () => {
    fixture.componentRef.setInput('mode', 'list');
    await settleZoneless(fixture);
    expect(labelsFor(fixture.nativeElement)).toEqual(['Column:', 'Separator:']);
  });

  it('gives each instance its own label ids', () => {
    const other = TestBed.createComponent(ClipboardCopyComponent);
    expect(other.componentInstance.delimiterId).not.toBe(fixture.componentInstance.delimiterId);
  });

  // Zoneless staleness canary: `copy()` runs from a bound `(click)`, but the
  // `flash('Copied!')` that sets the feedback label happens in a post-`await`
  // continuation (after `navigator.clipboard.writeText`), OUTSIDE the click's
  // CD-scheduling stack. It only repaints because `buttonText` is a signal.
  it('flashes "Copied!" after the async write resolves (zoneless canary)', async () => {
    const btn = fixture.nativeElement.querySelector('button.btn') as HTMLButtonElement;
    btn.click();
    await settleZoneless(fixture);
    expect(btn.textContent?.trim()).toBe('Copied!');
  });

  it('clears the flash label after the timer (zoneless canary)', async () => {
    const btn = fixture.nativeElement.querySelector('button.btn') as HTMLButtonElement;
    btn.click();
    await settleZoneless(fixture);
    expect(btn.textContent?.trim()).toBe('Copied!');

    await new Promise<void>((resolve) => setTimeout(resolve, 2100));
    await fixture.whenStable();
    expect(btn.textContent?.trim()).toBe('Copy');
  });
});
