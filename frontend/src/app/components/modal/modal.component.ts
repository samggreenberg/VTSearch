import { ChangeDetectionStrategy, Component, HostListener, OnDestroy, effect, input, output, signal, untracked } from '@angular/core';
import { CdkTrapFocus } from '@angular/cdk/a11y';
import { prefersReducedMotion } from '../../utils/reduced-motion';

/**
 * Stack of currently-open modal instances, in the order they opened.
 *
 * Modals stack in real flows (New Detector → media-crop, Settings →
 * importer picker, any modal → dialog-host confirm/prompt), and every
 * instance listens for Escape on `document`. Without the stack, one
 * keypress dismissed every open modal at once — losing the outer form's
 * state. Only the most-recently-opened modal reacts to Escape.
 */
const openModalStack: ModalComponent[] = [];

/**
 * How long the leave animation runs before the modal subtree is torn down,
 * in milliseconds. Kept in lockstep with `--transition-slow` (0.3s) in
 * `_variables.scss`, which the `.modal-backdrop`/`.modal-content` open/close
 * keyframes use. If that token changes, change this to match.
 */
const LEAVE_ANIMATION_MS = 300;

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-modal',
  standalone: true,
  imports: [CdkTrapFocus],
  templateUrl: './modal.component.html',
  styleUrl: './modal.component.scss',
})
export class ModalComponent implements OnDestroy {
  readonly title = input('');
  readonly open = input(false);
  readonly showCloseButton = input(true);
  readonly closed = output<void>();

  /**
   * Whether the modal subtree is in the DOM. Tracks `open()` on entry but
   * *lags* it on exit: when `open()` flips false we keep the node mounted and
   * play the leave animation, tearing it down only after `LEAVE_ANIMATION_MS`.
   * Without this lag the `@if` would rip the node out instantly and no leave
   * animation could ever run (see the modal enter/exit hook in
   * `docs/plans/ui-motion-vocabulary.md`).
   */
  readonly rendered = signal(false);

  /** True while the leave animation is playing; drives the `.is-leaving` class. */
  readonly leaving = signal(false);

  /** Pending teardown timer id for the leave animation, cleared on re-open. */
  private leaveTimer: ReturnType<typeof setTimeout> | undefined;

  /**
   * Guards origin capture to the first `focusin` after each open, so tabbing
   * within the modal doesn't re-point `transform-origin` at an inner control.
   */
  private originCaptured = false;

  constructor() {
    // Track this instance's place in the open-modal stack. Runs on every
    // open() change; instances created already-open (the common `@if` +
    // `[open]="true"` pattern) register at creation time, which matches
    // their visual stacking order. Stack membership follows the *logical*
    // open state, not `rendered`: a modal mid-leave is already closing, so
    // it should no longer capture Escape.
    effect(() => {
      const isOpen = this.open();
      untracked(() => {
        const idx = openModalStack.indexOf(this);
        if (isOpen) {
          if (idx === -1) openModalStack.push(this);
        } else if (idx !== -1) {
          openModalStack.splice(idx, 1);
        }
      });
    });

    // Drive the enter/leave render lifecycle off `open()`.
    effect(() => {
      const isOpen = this.open();
      untracked(() => {
        if (isOpen) {
          this.enter();
        } else if (this.rendered()) {
          this.leave();
        }
      });
    });
  }

  ngOnDestroy(): void {
    const idx = openModalStack.indexOf(this);
    if (idx !== -1) openModalStack.splice(idx, 1);
    if (this.leaveTimer !== undefined) clearTimeout(this.leaveTimer);
  }

  /** Mount (or re-mount) the subtree and let the enter animation play. */
  private enter(): void {
    if (this.leaveTimer !== undefined) {
      clearTimeout(this.leaveTimer);
      this.leaveTimer = undefined;
    }
    this.originCaptured = false;
    this.leaving.set(false);
    this.rendered.set(true);
  }

  /** Play the leave animation, then tear the subtree down. */
  private leave(): void {
    this.leaving.set(true);
    // Under reduced motion the animation is suppressed to ~instant, so tear
    // down synchronously rather than lingering an invisible, static node.
    if (prefersReducedMotion()) {
      this.rendered.set(false);
      this.leaving.set(false);
      return;
    }
    this.leaveTimer = setTimeout(() => {
      this.leaveTimer = undefined;
      this.rendered.set(false);
      this.leaving.set(false);
    }, LEAVE_ANIMATION_MS);
  }

  /**
   * Origin-aware scale-in: point the content's `transform-origin` at the
   * control that launched the modal, so it scales *from* that button.
   *
   * `cdkTrapFocus`'s auto-capture pulls focus into the dialog on open, and the
   * resulting `focusin` carries `relatedTarget` = the element that just lost
   * focus, i.e. the launching button. Reading it here needs no wiring at any of
   * the ~25 call sites, and fires before the first paint so the origin is set
   * before the scale animation renders. When there's no usable origin (focus
   * came from within, or from nothing measurable) we leave `transform-origin`
   * at its default center, which is the plain backdrop-fade + center-scale
   * baseline.
   */
  onFocusIn(event: FocusEvent): void {
    if (this.originCaptured) return;
    this.originCaptured = true;

    const related = event.relatedTarget as HTMLElement | null;
    const backdrop = event.currentTarget as HTMLElement | null;
    if (!related || !backdrop || backdrop.contains(related)) return;

    const content = backdrop.querySelector<HTMLElement>('.modal-content');
    if (!content) return;

    const from = related.getBoundingClientRect();
    if (!from.width || !from.height) return; // detached / zero-size trigger

    const box = content.getBoundingClientRect();
    const originX = from.left + from.width / 2 - box.left;
    const originY = from.top + from.height / 2 - box.top;
    content.style.transformOrigin = `${originX}px ${originY}px`;
  }

  onBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.close();
    }
  }

  close(): void {
    this.closed.emit();
  }

  /**
   * Close on Escape. Listening on `document` (not the backdrop) is required
   * because opening a modal from a button leaves focus on that button, so a
   * `(keydown)` bound to the never-focused backdrop never fired — Esc did
   * nothing until the user clicked into the modal. The `open()` guard ensures
   * only a currently-open modal reacts. Mirrors the `context-menu` /
   * `browse-bin-popup` `document:keydown.escape` pattern.
   *
   * Only the topmost open modal closes: with stacked modals every instance's
   * document listener fires for the same keypress, and closing them all at
   * once destroyed the outer flow's state along with the inner view.
   */
  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open() && openModalStack[openModalStack.length - 1] === this) {
      this.close();
    }
  }
}
