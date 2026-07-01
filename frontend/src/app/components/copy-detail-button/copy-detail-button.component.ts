import { ChangeDetectionStrategy, Component, OnDestroy, input, signal } from '@angular/core';

import { IconComponent } from '../icon/icon.component';

/**
 * A small icon-only button placed beside a single metadata detail (in the
 * VTSBrowse bin-popup panel and the center-panel Train/Find tray). Clicking it
 * copies that detail's plaintext value to the clipboard and briefly swaps the
 * copy glyph for a check as confirmation. The hover title reads
 * "Copy <category name> to clipboard".
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-copy-detail-button',
  standalone: true,
  imports: [IconComponent],
  template: `
    <button
      type="button"
      class="copy-detail-btn"
      [class.copied]="copied()"
      (click)="copy($event)"
      [title]="'Copy ' + label() + ' to clipboard'"
      [attr.aria-label]="'Copy ' + label() + ' to clipboard'">
      <vt-icon [type]="copied() ? 'check' : 'copy'" [size]="13" />
    </button>
  `,
  styleUrl: './copy-detail-button.component.scss',
})
export class CopyDetailButtonComponent implements OnDestroy {
  /** Plaintext value copied to the clipboard. */
  readonly value = input('');
  /** Category name woven into the hover title (e.g. "MD5", "Media Type"). */
  readonly label = input('');

  /** Signal so the {@link flash} ``setTimeout`` reset repaints the glyph back
   *  under zoneless change detection. */
  readonly copied = signal(false);

  private timer: ReturnType<typeof setTimeout> | null = null;

  ngOnDestroy(): void {
    if (this.timer) clearTimeout(this.timer);
  }

  async copy(event: Event): Promise<void> {
    // The metadata item may sit inside a clickable/selectable surface; keep the
    // copy click from bubbling into it.
    event.stopPropagation();
    const text = this.value();
    try {
      await navigator.clipboard.writeText(text);
      this.flash();
    } catch {
      // Clipboard API can reject in insecure contexts; fall back to execCommand.
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        this.flash();
      } catch {
        // give up silently
      }
      document.body.removeChild(ta);
    }
  }

  private flash(): void {
    this.copied.set(true);
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.copied.set(false);
      this.timer = null;
    }, 1500);
  }
}
