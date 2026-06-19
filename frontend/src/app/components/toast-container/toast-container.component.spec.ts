import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ToastContainerComponent } from './toast-container.component';
import { ToastService } from '../../services/toast.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('ToastContainerComponent', () => {
  let fixture: ComponentFixture<ToastContainerComponent>;
  let toast: ToastService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ToastContainerComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
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
});
