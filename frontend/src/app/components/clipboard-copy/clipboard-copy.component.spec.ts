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
