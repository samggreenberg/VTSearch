import { Component, OnDestroy, inject, signal } from '@angular/core';

import { AsyncPipe } from '@angular/common';
import { Toast, ToastService } from '../../services/toast.service';

/**
 * Stacked toast renderer mounted once in ``AppComponent``. Binds
 * ``ToastService.toasts$`` through the ``async`` pipe and renders each one.
 * Toasts with a rich ``errorContext`` get the expandable Details + Copy debug
 * info actions (preserves the old global-error-banner UX).
 */
@Component({
  selector: 'vt-toast-container',
  standalone: true,
  imports: [AsyncPipe],
  templateUrl: './toast-container.component.html',
  styleUrl: './toast-container.component.scss',
})
export class ToastContainerComponent implements OnDestroy {
  protected toastService = inject(ToastService);

  expandedId: number | null = null;
  /** Signal so the ``setTimeout`` reset in {@link markCopied} repaints the
   *  "Copied!" label back to its default under zoneless change detection. */
  readonly copiedId = signal<number | null>(null);

  private copiedTimer?: ReturnType<typeof setTimeout>;

  ngOnDestroy(): void {
    if (this.copiedTimer) clearTimeout(this.copiedTimer);
  }

  trackById(_: number, t: Toast): number {
    return t.id;
  }

  toggleDetails(t: Toast): void {
    this.expandedId = this.expandedId === t.id ? null : t.id;
  }

  dismiss(t: Toast): void {
    this.toastService.dismiss(t.id);
  }

  /** Fire a toast's action button. The toast is dismissed regardless of
   *  whether the handler throws so the user never sees a stuck toast
   *  with a clicked action. */
  runAction(t: Toast): void {
    try {
      t.action?.onClick();
    } finally {
      this.toastService.dismiss(t.id);
    }
  }

  async copyDebugInfo(t: Toast): Promise<void> {
    const text = this.toastService.formatForClipboard(t);
    try {
      await navigator.clipboard.writeText(text);
      this.markCopied(t.id);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        this.markCopied(t.id);
      } catch {
        // give up silently; details still visible in the toast
      }
      document.body.removeChild(ta);
    }
  }

  extraEntries(t: Toast): Array<{ key: string; value: string }> {
    const extra = t.errorContext?.extra;
    if (!extra) return [];
    return Object.entries(extra).map(([key, value]) => ({
      key,
      value: typeof value === 'string' ? value : JSON.stringify(value),
    }));
  }

  private markCopied(id: number): void {
    this.copiedId.set(id);
    if (this.copiedTimer) clearTimeout(this.copiedTimer);
    this.copiedTimer = setTimeout(() => {
      this.copiedId.set(null);
    }, 2000);
  }
}
