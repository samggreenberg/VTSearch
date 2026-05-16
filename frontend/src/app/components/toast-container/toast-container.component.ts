import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { Toast, ToastService } from '../../services/toast.service';

/**
 * Stacked toast renderer mounted once in ``AppComponent``. Subscribes
 * to ``ToastService.toasts$`` and renders each one. Toasts with a
 * rich ``errorContext`` get the expandable Details + Copy debug info
 * actions (preserves the old global-error-banner UX).
 */
@Component({
  selector: 'vt-toast-container',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toast-container.component.html',
  styleUrl: './toast-container.component.scss',
})
export class ToastContainerComponent implements OnInit, OnDestroy {
  toasts: Toast[] = [];
  expandedId: number | null = null;
  copiedId: number | null = null;

  private sub?: Subscription;
  private copiedTimer?: ReturnType<typeof setTimeout>;

  constructor(private toastService: ToastService) {}

  ngOnInit(): void {
    this.sub = this.toastService.toasts$.subscribe((toasts) => {
      this.toasts = toasts;
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
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
        // give up silently — details still visible in the toast
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
    this.copiedId = id;
    if (this.copiedTimer) clearTimeout(this.copiedTimer);
    this.copiedTimer = setTimeout(() => {
      this.copiedId = null;
    }, 2000);
  }
}
