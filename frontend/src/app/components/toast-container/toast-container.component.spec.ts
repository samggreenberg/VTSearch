import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ToastContainerComponent } from './toast-container.component';
import { ToastService } from '../../services/toast.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

describe('ToastContainerComponent', () => {
  let fixture: ComponentFixture<ToastContainerComponent>;
  let toast: ToastService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ToastContainerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ToastContainerComponent);
    toast = TestBed.inject(ToastService);
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  // Zoneless staleness canary: toasts are bound through `| async`, which calls
  // `markForCheck` on emit. Push a toast through the production channel
  // (`ToastService.error`, which `next`s the subject from a plain method call —
  // no bound listener in the stack), settle, and assert it renders with no
  // manual `detectChanges`.
  it('renders a toast pushed through the service (zoneless canary)', async () => {
    expect(fixture.nativeElement.querySelector('.toast')).toBeNull();

    toast.error({ message: 'Something broke' });
    await settleZoneless(fixture);

    const el = fixture.nativeElement.querySelector('.toast');
    expect(el).toBeTruthy();
    expect(el.textContent).toContain('Something broke');
  });

  it("shows Toasty's face for the level: sad, surprised, happy (#4680)", async () => {
    toast.error({ message: 'Broke' });
    toast.warning({ message: 'Careful' });
    toast.success({ message: 'Done' });
    await settleZoneless(fixture);
    const faces = [...fixture.nativeElement.querySelectorAll('.toast__toasty')].map((img: HTMLImageElement) =>
      img.getAttribute('src'),
    );
    expect(faces).toEqual(['toasty-sad.png', 'toasty-surprised.png', 'logo.png']);
  });

  it('removes a toast when dismissed (zoneless canary)', async () => {
    const id = toast.error({ message: 'Transient' });
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.toast')).toBeTruthy();

    toast.dismiss(id);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.toast')).toBeNull();
  });

  it('shows a Dismiss-all control only when more than one toast is stacked', async () => {
    toast.error({ message: 'First' });
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.toast-stack__dismiss-all')).toBeNull();

    toast.error({ message: 'Second' });
    await settleZoneless(fixture);
    const btn = fixture.nativeElement.querySelector('.toast-stack__dismiss-all');
    expect(btn).toBeTruthy();
    expect(btn.textContent).toContain('Dismiss all (2)');
  });

  it('clears the whole stack when Dismiss all is clicked', async () => {
    toast.error({ message: 'First' });
    toast.error({ message: 'Second' });
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelectorAll('.toast').length).toBe(2);

    fixture.nativeElement.querySelector('.toast-stack__dismiss-all').click();
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelectorAll('.toast').length).toBe(0);
  });

  describe('item lists (#4232)', () => {
    const buttonLabelled = (label: string): HTMLButtonElement | undefined =>
      Array.from<HTMLButtonElement>(fixture.nativeElement.querySelectorAll('.toast__btn')).find(
        (b) => b.textContent?.trim() === label,
      );

    it('offers no Details or Copy list on a toast without items', async () => {
      toast.warning({ message: 'Plain warning', detail: 'Nothing to list' });
      await settleZoneless(fixture);

      expect(buttonLabelled('Details')).toBeUndefined();
      expect(buttonLabelled('Copy list')).toBeUndefined();
    });

    it('keeps the list collapsed until Details is clicked, then shows every item', async () => {
      toast.warning({ message: 'Dropped 3 item(s)', items: ['a.wav', 'b.wav', 'c.wav'] });
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.toast__items')).toBeNull();

      const details = buttonLabelled('Details');
      expect(details).toBeTruthy();
      expect(details?.getAttribute('aria-expanded')).toBe('false');
      details?.click();
      await settleZoneless(fixture);

      const rows = Array.from<HTMLElement>(fixture.nativeElement.querySelectorAll('.toast__items li'));
      expect(rows.map((li) => li.textContent?.trim())).toEqual(['a.wav', 'b.wav', 'c.wav']);
      expect(buttonLabelled('Hide details')?.getAttribute('aria-expanded')).toBe('true');
    });

    it('copies the items one per line and confirms with Copied!', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
      try {
        toast.warning({ message: 'Dropped 2 item(s)', items: ['a.wav', 'b.wav'] });
        await settleZoneless(fixture);

        buttonLabelled('Copy list')?.click();
        await settleZoneless(fixture);

        expect(writeText).toHaveBeenCalledWith('a.wav\nb.wav');
        expect(buttonLabelled('Copied!')).toBeTruthy();
      } finally {
        Reflect.deleteProperty(navigator, 'clipboard');
      }
    });
  });
});
