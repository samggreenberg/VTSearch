import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DialogHostComponent } from './dialog-host.component';
import { VtDialogService } from '../../services/dialog.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('DialogHostComponent', () => {
  let component: DialogHostComponent;
  let fixture: ComponentFixture<DialogHostComponent>;
  let dialogService: VtDialogService;

  beforeEach(async () => {
    await configureZoneless({
      imports: [DialogHostComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DialogHostComponent);
    component = fixture.componentInstance;
    dialogService = TestBed.inject(VtDialogService);
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should resolve dialog on button click', async () => {
    const promise = dialogService.confirm('Are you sure?');
    await settleZoneless(fixture);

    expect(dialogService.dialogOpen()).toBe(true);

    component.onButtonClick(true);
    const result = await promise;
    expect(result).toBe(true);
    expect(dialogService.dialogOpen()).toBe(false);
  });

  it('should resolve with false on close', async () => {
    const promise = dialogService.confirm('Delete?');
    await settleZoneless(fixture);

    component.onClosed();
    const result = await promise;
    expect(result).toBe(false);
  });

  it('should resolve a prompt with null (not false) on Escape/backdrop close', async () => {
    // Regression: onClosed() hard-coded resolve(false); prompt() callers are
    // typed `string | null` and crashed on `false.trim()` / `false.split()`.
    const promise = dialogService.prompt('Name?');
    await settleZoneless(fixture);

    component.onClosed();
    const result = await promise;
    expect(result).toBeNull();
  });

  it('should settle a superseded dialog as cancelled instead of stranding it', async () => {
    // Regression: a second show() overwrote activeResolve, so the first
    // caller's await hung forever with no error.
    const first = dialogService.confirm('First?');
    await settleZoneless(fixture);
    const second = dialogService.confirm('Second?');
    await settleZoneless(fixture);

    expect(await first).toBe(false);

    component.onButtonClick(true);
    expect(await second).toBe(true);
  });

  it('should not show close button on dialog modal', async () => {
    const promise = dialogService.confirm('Delete?');
    await settleZoneless(fixture);

    const closeBtn = fixture.nativeElement.querySelector('.modal-close');
    expect(closeBtn).toBeNull();

    component.onButtonClick(true);
    await promise;
  });

  // Zoneless staleness canary: the dialog state is signalized precisely so a
  // dialog opened from a NON-event callback (a `.then()` continuation, a timer)
  // still schedules change detection and actually renders. Drive `confirm()`
  // from inside a `setTimeout` — an unpatched callback with no bound listener in
  // the stack — and assert the modal paints with no manual `detectChanges`.
  it('renders a dialog opened from a non-event callback (zoneless canary)', async () => {
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeNull();

    setTimeout(() => dialogService.confirm('Opened from a timer'));
    await settleZoneless(fixture);

    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop');
    expect(backdrop).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.dialog-message').textContent).toContain(
      'Opened from a timer',
    );
  });

  // The button labels are driven by the `dialogButtons` signal; a prompt with an
  // input must repaint the text field too, again from the plain method call.
  it('renders the prompt input and OK/Cancel buttons (zoneless canary)', async () => {
    dialogService.prompt('Name?', 'seed');
    await settleZoneless(fixture);

    const input = fixture.nativeElement.querySelector('input.form-input') as HTMLInputElement;
    expect(input).toBeTruthy();
    expect(input.value).toBe('seed');
    const labels = Array.from(
      fixture.nativeElement.querySelectorAll('[modal-footer] button'),
    ).map((b) => (b as HTMLElement).textContent?.trim());
    expect(labels).toEqual(['Cancel', 'OK']);
  });
});
