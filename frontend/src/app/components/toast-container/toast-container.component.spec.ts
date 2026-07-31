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

  describe('countdown toasts', () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it('renders the remaining seconds and repaints each tick', async () => {
      vi.useFakeTimers();
      toast.success({
        message: 'Done!',
        countdown: { label: 'Taking you back to the Dashboard in', seconds: 5, onExpire: () => {} },
      });
      TestBed.tick();

      const line = () => fixture.nativeElement.querySelector('.toast__countdown');
      expect(line().textContent).toContain('Taking you back to the Dashboard in');
      expect(line().textContent).toContain('5');

      await vi.advanceTimersByTimeAsync(1000);
      TestBed.tick();
      expect(line().textContent).toContain('4');
    });

    it('runs the expiry handler once the countdown empties, then drops the toast', async () => {
      vi.useFakeTimers();
      const onExpire = vi.fn();
      toast.success({ message: 'Done!', countdown: { label: 'Leaving in', seconds: 2, onExpire } });
      TestBed.tick();

      await vi.advanceTimersByTimeAsync(2000);
      TestBed.tick();
      expect(onExpire).toHaveBeenCalledTimes(1);
      expect(fixture.nativeElement.querySelector('.toast')).toBeNull();
    });

    it('cancels the expiry handler when the toast is dismissed first', async () => {
      vi.useFakeTimers();
      const onExpire = vi.fn();
      const id = toast.success({
        message: 'Done!',
        countdown: { label: 'Leaving in', seconds: 3, onExpire },
      });
      TestBed.tick();

      toast.dismiss(id);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(onExpire).not.toHaveBeenCalled();
    });

    it('does not auto-dismiss a countdown toast out from under its own timer', async () => {
      vi.useFakeTimers();
      const onExpire = vi.fn();
      // Plain success toasts auto-dismiss at 5s; a 10s countdown must outlive
      // that, or the user watches the message vanish mid-count.
      toast.success({ message: 'Done!', countdown: { label: 'Leaving in', seconds: 10, onExpire } });
      TestBed.tick();

      await vi.advanceTimersByTimeAsync(6000);
      TestBed.tick();
      expect(fixture.nativeElement.querySelector('.toast')).toBeTruthy();
      expect(onExpire).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(4000);
      expect(onExpire).toHaveBeenCalledTimes(1);
    });
  });
});
