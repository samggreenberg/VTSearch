import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ModalComponent } from './modal.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

@Component({
  standalone: true,
  imports: [ModalComponent],
  template: `
    <vt-modal
      [title]="'Test Modal'"
      [open]="isOpen()"
      [showCloseButton]="showCloseButton()"
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

@Component({
  standalone: true,
  imports: [ModalComponent],
  template: `
    <button class="trigger">Open</button>
    <vt-modal [title]="'Focusable'" [open]="isOpen()" (closed)="isOpen.set(false)">
      <input class="body-input" />
    </vt-modal>
  `,
})
class FocusHostComponent {
  readonly isOpen = signal(false);
}

describe('ModalComponent focus management', () => {
  let fixture: ComponentFixture<FocusHostComponent>;
  let host: FocusHostComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FocusHostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(FocusHostComponent);
    host = fixture.componentInstance;
    // Attach to the live document so `document.activeElement` tracks focus():
    // jsdom only updates it for elements connected to the document.
    document.body.appendChild(fixture.nativeElement);
  });

  afterEach(() => {
    fixture.nativeElement.remove();
  });

  it('moves initial focus into the dialog when opened', async () => {
    await settleZoneless(fixture);
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop');
    expect(backdrop.contains(document.activeElement)).toBe(true);
    // cdkFocusInitial targets the heading so screen readers land on the title.
    expect(document.activeElement).toBe(fixture.nativeElement.querySelector('.modal-header h2'));
  });

  it('restores focus to the triggering element when closed', async () => {
    const trigger = fixture.nativeElement.querySelector('.trigger') as HTMLButtonElement;
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(document.activeElement).not.toBe(trigger);

    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(document.activeElement).toBe(trigger);
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
