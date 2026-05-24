import { NgIf } from '@angular/common';
import { Component, ElementRef, HostListener, Input, signal } from '@angular/core';

const HOVER_DELAY_MS = 500;

@Component({
  selector: 'vt-field-hint-icon',
  standalone: true,
  imports: [NgIf],
  template: `
    <span
      class="field-hint-icon"
      tabindex="0"
      role="img"
      [attr.aria-label]="ariaLabel || hint"
      (mouseenter)="onMouseEnter()"
      (mouseleave)="onMouseLeave()"
      (focus)="onMouseEnter()"
      (blur)="hide()"
      (click)="onClick($event)"
      (keydown.escape)="hide()"
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
      <span class="field-hint-tooltip" *ngIf="visible()" role="tooltip">{{ hint }}</span>
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
        border: 1px solid var(--border-color);
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
  @Input() hint = '';
  @Input() ariaLabel = '';

  readonly visible = signal(false);
  private hoverTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly elementRef: ElementRef<HTMLElement>) {}

  onMouseEnter(): void {
    if (this.visible()) return;
    this.clearTimer();
    this.hoverTimer = setTimeout(() => {
      this.hoverTimer = null;
      this.visible.set(true);
    }, HOVER_DELAY_MS);
  }

  onMouseLeave(): void {
    this.clearTimer();
    this.visible.set(false);
  }

  onClick(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.clearTimer();
    this.visible.set(true);
  }

  hide(): void {
    this.clearTimer();
    this.visible.set(false);
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.visible()) return;
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
