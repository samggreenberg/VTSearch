
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  computed,
  inject,
  input,
  signal,
} from '@angular/core';

const HOVER_DELAY_MS = 500;

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-field-hint-icon',
  standalone: true,
  imports: [],
  template: `
    <span
      class="field-hint-icon"
      tabindex="0"
      role="img"
      [attr.aria-label]="ariaLabel() || hint()"
      (mouseenter)="onMouseEnter()"
      (mouseleave)="onMouseLeave()"
      (focus)="onMouseEnter()"
      (blur)="onBlur()"
      (click)="onClick($event)"
      (keydown.escape)="onEscape($event)"
      >
      <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" focusable="false">
        <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.3" />
        <text
          x="8"
          y="11.5"
          font-size="9"
          font-weight="600"
          text-anchor="middle"
          fill="currentColor"
        >?</text>
      </svg>
      @if (visible()) {
        <span class="field-hint-tooltip" role="tooltip">{{ hint() }}</span>
      }
    </span>
    `,
  styles: [
    `
      .field-hint-icon {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-left: var(--space-2xs);
        color: var(--text-muted);
        cursor: help;
        vertical-align: middle;
        line-height: 1;
      }
      .field-hint-icon:hover,
      .field-hint-icon:focus-visible {
        color: var(--text-primary);
        outline: none;
      }
      .field-hint-tooltip {
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        width: max-content;
        max-width: 260px;
        padding: 6px 8px;
        background: var(--bg-surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 4px;
        font-size: 12px;
        font-weight: 400;
        line-height: 1.4;
        white-space: normal;
        text-align: left;
        cursor: default;
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
      }
    `,
  ],
})
export class FieldHintIconComponent {
  private readonly elementRef = inject<ElementRef<HTMLElement>>(ElementRef);

  readonly hint = input('');
  readonly ariaLabel = input('');

  /** Sticky state set by clicking the icon; stays open until dismissed. */
  private readonly pinned = signal(false);
  /** Transient state driven by hover/focus; cleared as soon as you leave. */
  private readonly hovered = signal(false);

  /** The tooltip shows if it's pinned by a click OR currently hovered. */
  readonly visible = computed(() => this.pinned() || this.hovered());

  private hoverTimer: ReturnType<typeof setTimeout> | null = null;

  onMouseEnter(): void {
    // Already showing (pinned or hovered): nothing to schedule.
    if (this.visible()) return;
    this.clearTimer();
    this.hoverTimer = setTimeout(() => {
      this.hoverTimer = null;
      this.hovered.set(true);
    }, HOVER_DELAY_MS);
  }

  onMouseLeave(): void {
    // Only the transient hover ends here. A click-pinned tooltip stays open so
    // the user can move the pointer off the icon to read it.
    this.clearTimer();
    this.hovered.set(false);
  }

  onClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.clearTimer();
    // Toggle immediately: show if hidden, dismiss if already showing (from
    // either a previous click or an in-progress hover).
    const willShow = !this.visible();
    this.hovered.set(false);
    this.pinned.set(willShow);
  }

  onBlur(): void {
    this.hide();
  }

  onEscape(event: Event): void {
    // Only consume Escape when we actually have a tooltip to close, so an
    // Escape on an idle icon still bubbles up to close a parent modal.
    if (this.visible()) {
      event.stopPropagation();
    }
    this.hide();
  }

  hide(): void {
    this.clearTimer();
    this.hovered.set(false);
    this.pinned.set(false);
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.pinned()) return;
    if (!this.elementRef.nativeElement.contains(event.target as Node)) {
      this.hide();
    }
  }

  private clearTimer(): void {
    if (this.hoverTimer !== null) {
      clearTimeout(this.hoverTimer);
      this.hoverTimer = null;
    }
  }
}
