import { Component, Input } from '@angular/core';

@Component({
  selector: 'vt-field-hint-icon',
  standalone: true,
  template: `
    <span
      class="field-hint-icon"
      [title]="hint"
      tabindex="0"
      role="img"
      [attr.aria-label]="ariaLabel || hint"
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
    `,
  ],
})
export class FieldHintIconComponent {
  @Input() hint = '';
  @Input() ariaLabel = '';
}
