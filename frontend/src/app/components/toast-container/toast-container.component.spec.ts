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
});
