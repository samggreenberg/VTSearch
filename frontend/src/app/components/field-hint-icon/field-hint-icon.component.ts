
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  afterRenderEffect,
  computed,
  effect,
  inject,
  input,
  signal,
  viewChild,
} from '@angular/core';
import type { RectLike } from '../toasty-hint/toasty-hint.component';

const HOVER_DELAY_MS = 500;

/** Clearance (px) between the icon, or its block, and the tooltip. */
const GAP = 4;
/** The closest (px) the tooltip comes to the edge of the window. */
const EDGE = 8;

/** Which side of its anchor the tooltip sits on. */
export type FieldHintSide = 'above' | 'below';

/** Where the tooltip goes, in viewport px, and which side it ended up on. */
export interface FieldHintPlacement {
  left: number;
  top: number;
  side: FieldHintSide;
}

/**
 * Place a tooltip against `anchor` (viewport coordinates, as
 * `getBoundingClientRect()` returns them): centred on it, on the preferred
 * `side`, flipped to the other side when the preferred one runs off the window
 * and the other does not, then pushed back inside the window's edges. A
 * tooltip near the left of a dialog box therefore slides right rather than
 * running off the window (#5226), and one on the top row of the window opens
 * below instead. `tip` is the tooltip's own size and `viewport` the window's.
 */
export function fieldHintPlacement(
  anchor: RectLike,
  tip: { width: number; height: number },
  viewport: { width: number; height: number },
  side: FieldHintSide,
): FieldHintPlacement {
  const aboveTop = anchor.top - GAP - tip.height;
  const belowTop = anchor.bottom + GAP;
  const fitsAbove = aboveTop >= EDGE;
  const fitsBelow = belowTop + tip.height <= viewport.height - EDGE;
  if (side === 'above' && !fitsAbove && fitsBelow) side = 'below';
  else if (side === 'below' && !fitsBelow && fitsAbove) side = 'above';
  const clamp = (v: number, size: number, room: number): number =>
    Math.max(EDGE, Math.min(v, room - EDGE - size));
  return {
    left: clamp((anchor.left + anchor.right) / 2 - tip.width / 2, tip.width, viewport.width),
    top: clamp(side === 'above' ? aboveTop : belowTop, tip.height, viewport.height),
    side,
  };
}

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
        <span
          #tip
          class="field-hint-tooltip"
          [class.field-hint-tooltip--block]="placement() === 'below-block'"
          role="tooltip"
          popover="manual"
        >{{ hint() }}</span>
      }
    </span>
    `,
  styles: [
    `
      .field-hint-icon {
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
      /* Fixed to the window and, where the browser has popovers, raised into
         the top layer, so no dialog body or panel that clips its overflow can
         cut it off (#5226). \`place()\` sets \`left\` / \`top\`; the inset
         and margin reset undo the popover's own centring. */
      .field-hint-tooltip {
        position: fixed;
        inset: auto;
        margin: 0;
        overflow: visible;
        z-index: var(--z-tooltip);
        width: max-content;
        max-width: min(260px, calc(100vw - 16px));
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
      /* \`below-block\`: \`place()\` sets the width to the block's. */
      .field-hint-tooltip--block {
        max-width: calc(100vw - 16px);
      }
    `,
  ],
})
export class FieldHintIconComponent {
  private readonly elementRef = inject<ElementRef<HTMLElement>>(ElementRef);

  readonly hint = input('');
  readonly ariaLabel = input('');
  /**
   * Where the tooltip opens. `above` (the default) centres it over the icon,
   * which suits a form row in a modal. `below-block` hangs it under the
   * host's nearest *positioned* ancestor, as wide as that ancestor, so it
   * leaves the control that block holds in view (the left panel's Threshold
   * spectrum): give the block `position: relative`. Either way the tooltip
   * flips to the other side when its own runs off the window, and slides back
   * inside the window's edges (see {@link fieldHintPlacement}).
   */
  readonly placement = input<'above' | 'below-block'>('above');

  /** The tooltip, while it shows. */
  private readonly tip = viewChild<ElementRef<HTMLElement>>('tip');

  /** Sticky state set by clicking the icon; stays open until dismissed. */
  private readonly pinned = signal(false);
  /** Transient state driven by hover/focus; cleared as soon as you leave. */
  private readonly hovered = signal(false);

  /** The tooltip shows if it's pinned by a click OR currently hovered. */
  readonly visible = computed(() => this.pinned() || this.hovered());

  private hoverTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    // Place the tooltip when it renders, and again when its text or placement
    // changes its size.
    afterRenderEffect(() => {
      this.hint();
      this.placement();
      const tip = this.tip()?.nativeElement;
      if (tip) this.place(tip);
    });
    // While it shows, follow the icon as whatever holds it scrolls, and as
    // the window resizes. Captured, because a dialog body's `scroll` does
    // not bubble.
    effect((onCleanup) => {
      const tip = this.tip()?.nativeElement;
      if (!tip) return;
      const follow = (): void => this.place(tip);
      window.addEventListener('scroll', follow, { capture: true, passive: true });
      window.addEventListener('resize', follow, { passive: true });
      onCleanup(() => {
        window.removeEventListener('scroll', follow, { capture: true });
        window.removeEventListener('resize', follow);
      });
    });
  }

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

  /**
   * Raise the tooltip into the top layer (once per showing) and put it beside
   * its anchor: the icon, or for `below-block` the block, whose width it
   * takes. Writes the tooltip's style directly; no template binding reads it.
   */
  private place(tip: HTMLElement): void {
    if (typeof tip.showPopover === 'function' && !tip.matches(':popover-open')) {
      tip.showPopover();
    }
    const host = this.elementRef.nativeElement;
    const block = this.placement() === 'below-block';
    // jsdom has no layout, so no `offsetParent`; the parent stands in there.
    const anchor = block
      ? ((host.offsetParent as HTMLElement | null) ?? host.parentElement ?? host)
      : (host.querySelector<HTMLElement>('.field-hint-icon') ?? host);
    const rect = anchor.getBoundingClientRect();
    tip.style.width = block ? `${rect.width}px` : '';
    const root = document.documentElement;
    const { left, top } = fieldHintPlacement(
      rect,
      { width: tip.offsetWidth, height: tip.offsetHeight },
      { width: root.clientWidth || window.innerWidth, height: root.clientHeight || window.innerHeight },
      block ? 'below' : 'above',
    );
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  private clearTimer(): void {
    if (this.hoverTimer !== null) {
      clearTimeout(this.hoverTimer);
      this.hoverTimer = null;
    }
  }
}
