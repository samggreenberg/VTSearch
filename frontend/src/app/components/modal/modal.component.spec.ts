import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { ModalComponent, topOpenModal } from './modal.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

@Component({
  standalone: true,
  imports: [ModalComponent],
  template: `
    <button class="trigger">Trigger</button>
    <vt-modal
      [title]="'Test Modal'"
      [open]="isOpen()"
      [showCloseButton]="showCloseButton()"
      [initialFocus]="initialFocus()"
      (closed)="onClose()"
    >
      <p>Body content</p>
      <button modal-footer>Footer button</button>
    </vt-modal>
  `,
})
class TestHostComponent {
  // Signals so the host drives the modal through the zoneless CD path (a plain
  // field write on the top-level host would not schedule change detection).
  readonly isOpen = signal(false);
  readonly showCloseButton = signal(true);
  readonly initialFocus = signal<'first-control' | 'dialog'>('first-control');
  closeCalled = false;
  onClose(): void {
    this.closeCalled = true;
  }
}

@Component({
  standalone: true,
  imports: [ModalComponent],
  template: `
    <vt-modal [title]="'Outer'" [open]="true" (closed)="outerClosed = true">
      <p>Outer body</p>
      <button class="outer-trigger">Open inner</button>
    </vt-modal>
    @if (innerVisible()) {
      <vt-modal [title]="'Inner'" [open]="true" (closed)="innerClosed = true">
        <p>Inner body</p>
      </vt-modal>
    }
  `,
})
class StackedHostComponent {
  // Mirrors the real nesting pattern (New Detector → crop modal, Settings →
  // importer): the inner modal is created already-open by an @if.
  readonly innerVisible = signal(false);
  outerClosed = false;
  innerClosed = false;
}

describe('ModalComponent', () => {
  let fixture: ComponentFixture<TestHostComponent>;
  let host: TestHostComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    host = fixture.componentInstance;
  });

  it('should not render when closed', async () => {
    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeNull();
  });

  it('should render when open', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.modal-header h2')?.textContent).toContain('Test Modal');
  });

  it('should project body content', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-body p')?.textContent).toContain('Body content');
  });

  it('should emit closed on close button click', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    fixture.nativeElement.querySelector('.modal-close').click();
    expect(host.closeCalled).toBe(true);
  });

  it('should emit closed on backdrop click', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    fixture.nativeElement.querySelector('.modal-backdrop').click();
    expect(host.closeCalled).toBe(true);
  });

  it('should not close on content click', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    fixture.nativeElement.querySelector('.modal-content').click();
    expect(host.closeCalled).toBe(false);
  });

  it('should close on Escape dispatched at document level (focus not in the modal)', async () => {
    // Regression: modals open from a button, so focus stays on the button and a
    // backdrop-scoped (keydown) never fired. The document-level handler must
    // close an open modal regardless of where focus is.
    host.isOpen.set(true);
    await settleZoneless(fixture);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(host.closeCalled).toBe(true);
  });

  it('should ignore Escape when the modal is closed', async () => {
    host.isOpen.set(false);
    await settleZoneless(fixture);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(host.closeCalled).toBe(false);
  });

  it('should hide close button when showCloseButton is false', async () => {
    host.isOpen.set(true);
    host.showCloseButton.set(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-close')).toBeNull();
  });

  it('should show close button by default', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-close')).toBeTruthy();
  });
});

describe('ModalComponent as a native <dialog>', () => {
  // jsdom has <dialog> but none of its methods; src/test-setup.ts stands in
  // showModal()/close() as attribute toggles. So these specs pin the wiring
  // (a modal <dialog>, opened modally, focus handed back), and chromium is the
  // check for what the platform does with it: initial focus, inertness, the
  // top layer.
  let fixture: ComponentFixture<TestHostComponent>;
  let host: TestHostComponent;

  const dialogEl = () => fixture.nativeElement.querySelector('dialog') as HTMLDialogElement | null;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    host = fixture.componentInstance;
  });

  afterEach(() => vi.restoreAllMocks());

  it('renders the dialog as a <dialog> opened with showModal()', async () => {
    const showModal = vi.spyOn(HTMLDialogElement.prototype, 'showModal');
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const dialog = dialogEl();
    expect(dialog?.classList.contains('modal-backdrop')).toBe(true);
    expect(dialog?.open).toBe(true);
    // Modally, not show(): that is what makes the page behind inert.
    expect(showModal).toHaveBeenCalledTimes(1);
    expect(showModal.mock.contexts[0]).toBe(dialog);
    expect(dialog?.getAttribute('aria-label')).toBe('Test Modal');
  });

  it('leaves Escape to vt-modal (closedby="none") and refuses a native cancel', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const dialog = dialogEl()!;
    expect(dialog.getAttribute('closedby')).toBe('none');
    const cancel = new Event('cancel', { cancelable: true });
    dialog.dispatchEvent(cancel);
    expect(cancel.defaultPrevented).toBe(true);
    expect(host.closeCalled).toBe(false);
  });

  it('claims the Escape keypress it acts on, so the browser does not close the element too', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const esc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(esc);
    expect(host.closeCalled).toBe(true);
    expect(esc.defaultPrevented).toBe(true);
  });

  it('does not claim Escape while closed', async () => {
    await settleZoneless(fixture);
    const esc = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(esc);
    expect(esc.defaultPrevented).toBe(false);
  });

  it('when the browser closes the element anyway, keeps it modal and asks the parent to close', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const dialog = dialogEl()!;
    dialog.close();
    // Still showing until the parent, which owns [open], takes it away.
    expect(dialog.open).toBe(true);
    expect(host.closeCalled).toBe(true);
  });

  it('hands focus back to the trigger when the dialog closes', async () => {
    const trigger = fixture.nativeElement.querySelector('.trigger') as HTMLButtonElement;
    trigger.focus();
    host.isOpen.set(true);
    await settleZoneless(fixture);
    // The platform moves focus in on showModal(); jsdom does not, so do it here.
    (fixture.nativeElement.querySelector('.modal-close') as HTMLButtonElement).focus();
    expect(document.activeElement).not.toBe(trigger);

    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(dialogEl()).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('hands focus back when the parent takes the whole modal away', async () => {
    const trigger = fixture.nativeElement.querySelector('.trigger') as HTMLButtonElement;
    trigger.focus();
    host.isOpen.set(true);
    await settleZoneless(fixture);
    (fixture.nativeElement.querySelector('.modal-close') as HTMLButtonElement).focus();

    // Removing the element drops focus to <body>; ngOnDestroy runs after.
    dialogEl()!.remove();
    fixture.debugElement.query(By.directive(ModalComponent)).injector.get(ModalComponent).ngOnDestroy();
    expect(document.activeElement).toBe(trigger);
  });

  it('leaves focus alone when something else took it before the dialog closed', async () => {
    const trigger = fixture.nativeElement.querySelector('.trigger') as HTMLButtonElement;
    trigger.focus();
    host.isOpen.set(true);
    await settleZoneless(fixture);
    const elsewhere = document.createElement('button');
    document.body.appendChild(elsewhere);
    try {
      elsewhere.focus();
      host.isOpen.set(false);
      await settleZoneless(fixture);
      expect(document.activeElement).toBe(elsewhere);
    } finally {
      elsewhere.remove();
    }
  });

  it('leaves initial focus to the first control by default', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const box = fixture.nativeElement.querySelector('.modal-content') as HTMLElement;
    expect(box.hasAttribute('autofocus')).toBe(false);
    expect(box.hasAttribute('tabindex')).toBe(false);
  });

  it("with initialFocus 'dialog', makes the box itself the initial focus (#4721)", async () => {
    host.initialFocus.set('dialog');
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const box = fixture.nativeElement.querySelector('.modal-content') as HTMLElement;
    // showModal() focuses the first `autofocus` element inside the dialog, here
    // one that can take focus but is not in the Tab order.
    expect(box.hasAttribute('autofocus')).toBe(true);
    expect(box.getAttribute('tabindex')).toBe('-1');
  });

  it('exposes the dialog box and the <dialog> element', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const modal = fixture.debugElement.query(By.directive(ModalComponent)).componentInstance as ModalComponent;
    expect(modal.box()?.nativeElement).toBe(fixture.nativeElement.querySelector('.modal-content'));
    expect(modal.dialog()?.nativeElement).toBe(dialogEl());
  });

  it('renders no <dialog> while closed', async () => {
    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(dialogEl()).toBeNull();
  });
});

describe('ModalComponent stacking', () => {
  let fixture: ComponentFixture<StackedHostComponent>;
  let host: StackedHostComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StackedHostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(StackedHostComponent);
    host = fixture.componentInstance;
  });

  it('Escape closes only the topmost open modal, not the whole stack', async () => {
    // Regression: every instance listens on document, so one Esc used to
    // dismiss the outer flow (and its form state) along with the inner view.
    await settleZoneless(fixture);
    host.innerVisible.set(true);
    await settleZoneless(fixture);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(host.innerClosed).toBe(true);
    expect(host.outerClosed).toBe(false);
  });

  it("names the topmost open modal, for Toasty's hints (#4807)", async () => {
    await settleZoneless(fixture);
    const modals = () => fixture.debugElement.queryAll(By.directive(ModalComponent)).map((d) => d.componentInstance);
    expect(topOpenModal()).toBe(modals()[0]);
    host.innerVisible.set(true);
    await settleZoneless(fixture);
    expect(topOpenModal()).toBe(modals()[1]);
    host.innerVisible.set(false);
    await settleZoneless(fixture);
    expect(topOpenModal()).toBe(modals()[0]);
    fixture.destroy();
    expect(topOpenModal()).toBeNull();
  });

  it('hands focus back into the outer dialog when the inner one goes', async () => {
    await settleZoneless(fixture);
    const outerTrigger = fixture.nativeElement.querySelector('.outer-trigger') as HTMLButtonElement;
    outerTrigger.focus();
    host.innerVisible.set(true);
    await settleZoneless(fixture);
    const dialogs = fixture.nativeElement.querySelectorAll('dialog') as NodeListOf<HTMLDialogElement>;
    expect(dialogs.length).toBe(2);
    expect([...dialogs].every((d) => d.open)).toBe(true);
    (dialogs[1].querySelector('.modal-close') as HTMLButtonElement).focus();

    host.innerVisible.set(false);
    await settleZoneless(fixture);
    expect(document.activeElement).toBe(outerTrigger);
  });

  it('after the inner modal is destroyed, Escape reaches the outer modal', async () => {
    await settleZoneless(fixture);
    host.innerVisible.set(true);
    await settleZoneless(fixture);
    host.innerVisible.set(false);
    await settleZoneless(fixture);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(host.outerClosed).toBe(true);
  });
});
