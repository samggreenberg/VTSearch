import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { CdkTrapFocus } from '@angular/cdk/a11y';
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

describe('ModalComponent enter/leave lifecycle', () => {
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

  afterEach(() => {
    // A couple of tests force the app-level reduced-motion class on <html>;
    // never leak it into sibling specs.
    document.documentElement.classList.remove('animations-off');
  });

  it('keeps the subtree mounted and marks it leaving when closed (leave animation runs)', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeTruthy();

    // Close: the node must NOT be torn out synchronously — it stays mounted
    // with `.is-leaving` so the leave keyframes can play. settleZoneless only
    // flushes one macrotask, not the 300ms teardown timer, so this asserts the
    // mid-leave state deterministically.
    host.isOpen.set(false);
    await settleZoneless(fixture);
    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop');
    expect(backdrop).toBeTruthy();
    expect(backdrop.classList.contains('is-leaving')).toBe(true);
  });

  it('tears the subtree down immediately on close under reduced motion', async () => {
    document.documentElement.classList.add('animations-off');
    host.isOpen.set(true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeTruthy();

    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeNull();
  });

  it('re-opening before teardown cancels the leave and drops the leaving flag', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.modal-backdrop').classList.contains('is-leaving')).toBe(true);

    // Re-open while the leave timer is still pending: the node stays, but is
    // no longer leaving, and (crucially) the pending teardown is cancelled so
    // the modal doesn't vanish mid-use.
    host.isOpen.set(true);
    await settleZoneless(fixture);
    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop');
    expect(backdrop).toBeTruthy();
    expect(backdrop.classList.contains('is-leaving')).toBe(false);
  });
});

describe('ModalComponent origin-aware scale-in', () => {
  let fixture: ComponentFixture<TestHostComponent>;
  let host: TestHostComponent;
  let trigger: HTMLButtonElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    host = fixture.componentInstance;

    // A stand-in "launching button" outside the modal. jsdom has no layout
    // engine, so getBoundingClientRect returns zeros; stub a real rect.
    trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.getBoundingClientRect = () =>
      ({ left: 100, top: 200, width: 40, height: 20, right: 140, bottom: 220, x: 100, y: 200, toJSON: () => ({}) }) as DOMRect;
  });

  afterEach(() => {
    trigger.remove();
  });

  function stubContentRect(): HTMLElement {
    const content = fixture.nativeElement.querySelector('.modal-content') as HTMLElement;
    content.getBoundingClientRect = () =>
      ({ left: 60, top: 60, width: 400, height: 300, right: 460, bottom: 360, x: 60, y: 60, toJSON: () => ({}) }) as DOMRect;
    return content;
  }

  it('points transform-origin at the launching button (focusin relatedTarget)', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop') as HTMLElement;
    const content = stubContentRect();

    backdrop.dispatchEvent(new FocusEvent('focusin', { relatedTarget: trigger, bubbles: true }));

    // button center (120, 210) minus content top-left (60, 60) = (60px, 150px)
    expect(content.style.transformOrigin).toBe('60px 150px');
  });

  it('captures the origin only once — later focusin events do not re-point it', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop') as HTMLElement;
    const content = stubContentRect();

    backdrop.dispatchEvent(new FocusEvent('focusin', { relatedTarget: trigger, bubbles: true }));
    expect(content.style.transformOrigin).toBe('60px 150px');

    // A second trigger elsewhere must be ignored (tabbing within the modal, etc.).
    const other = document.createElement('button');
    document.body.appendChild(other);
    other.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 10, height: 10, right: 10, bottom: 10, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    backdrop.dispatchEvent(new FocusEvent('focusin', { relatedTarget: other, bubbles: true }));
    expect(content.style.transformOrigin).toBe('60px 150px');
    other.remove();
  });

  it('leaves transform-origin at default when focus comes from within the modal', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);
    const backdrop = fixture.nativeElement.querySelector('.modal-backdrop') as HTMLElement;
    const content = stubContentRect();

    // relatedTarget inside the backdrop → no meaningful origin → center scale.
    backdrop.dispatchEvent(new FocusEvent('focusin', { relatedTarget: content, bubbles: true }));
    expect(content.style.transformOrigin).toBe('');
  });
});

describe('ModalComponent focus management', () => {
  // We assert the CdkTrapFocus wiring rather than live focus movement: the
  // directive's auto-capture (initial focus in, Tab trap, restore-to-trigger on
  // close) is exercised by CDK's own tests, and its InteractivityChecker treats
  // every element as invisible under jsdom (no layout engine), so it refuses to
  // move focus headlessly. Verifying the trap is attached and auto-capturing
  // guards against the wiring being dropped; the real behavior only surfaces in
  // a browser.
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

  it('attaches an auto-capturing focus trap to the open dialog', async () => {
    host.isOpen.set(true);
    await settleZoneless(fixture);

    const trapEl = fixture.debugElement.query(By.directive(CdkTrapFocus));
    expect(trapEl).toBeTruthy();
    // The trap wraps the whole dialog so Tab cannot escape to the page behind it.
    expect((trapEl.nativeElement as HTMLElement).classList.contains('modal-backdrop')).toBe(true);
    // Auto-capture is what pulls focus in on open and restores it to the trigger
    // on close.
    expect(trapEl.injector.get(CdkTrapFocus).autoCapture).toBe(true);
  });

  it('does not render a focus trap while the dialog is closed', async () => {
    host.isOpen.set(false);
    await settleZoneless(fixture);
    expect(fixture.debugElement.query(By.directive(CdkTrapFocus))).toBeNull();
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
